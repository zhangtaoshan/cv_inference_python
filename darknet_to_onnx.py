from collections import OrderedDict
from mimetypes import init
from onnx import helper, TensorProto
import numpy as np
import onnx
import os  


class DarkNetParser:
    def __init__(self, supported_layers):
        self.layer_configs = OrderedDict()
        self.supported_layers = supported_layers
        self.layer_counter = 0

    def parser_cfg_file(self, cfg_file_path):
        with open(cfg_file_path) as cfg_file:
            remainder = cfg_file.read()
            while remainder is not None:
                # 处理当前块
                layer_dict, layer_name, remainder = self.next_layer(remainder)
                if layer_dict is not None:
                    self.layer_configs[layer_name] = layer_dict
        # 返回当前块的内容
        return self.layer_configs

    def next_layer(self, remainder):
        remainder = remainder.split("[", 1)
        if len(remainder) == 2:
            remainder = remainder[1]
        else:
            return None, None, None 
        remainder = remainder.split("]", 1)
        if len(remainder) == 2:
            layer_type, remainder = remainder
        else:
            return None, None, None 
        # 上面两个条件语句将当前 remainder 中 [xx] 的内容即层类型提取出来

        if remainder.replace(" ", "")[0] == "#":
            remainder = remainder.split("\n", 1)[1]
        
        # 每块参数后有个空行，使用 \n\n 提取当前参数块
        layer_param_block, remainder = remainder.split("\n\n", 1)
        # 生成列表后将参数存放到列表
        layer_param_lines = layer_param_block.split("\n")[1:]
        # 以层索引 + 层类型命名
        layer_name = str(self.layer_counter).zfill(3) + "_" + layer_type 
        layer_dict = dict(type=layer_type)
        if layer_type in self.supported_layers:
            for param_line in layer_param_lines:
                # 过滤注释
                if param_line[0] == "#":
                    continue
                # 返回参数类型及其对应的值 
                param_type, param_value = self.parse_params(param_line)
                layer_dict[param_type] = param_value
        self.layer_counter += 1
        # 返回 remainder 剩余内容
        return layer_dict, layer_name, remainder

    def parse_params(self, param_line):
        # param_line 类似于 xxx=xxx 的形式
        param_line = param_line.replace(" ", "")
        param_type, param_value_raw = param_line.split("=")
        param_value = None 
        # 解析 route 层
        if param_type == "layers":
            layer_indexes = list()
            for index in param_value_raw.split(","):
                layer_indexes.append(int(index))
            param_value = layer_indexes
        # 解析 xxx=digit 的形式，数字可能为正或负 (shortcut)
        elif isinstance(param_value_raw, str) and not param_value_raw.isalpha():
            condition_param_value_positive = param_value_raw.isdigit()
            condition_param_value_negative = \
                param_value_raw[0] == "-" and param_value_raw[1:].isdigit()
            # 处理整数
            if condition_param_value_positive or condition_param_value_negative:
                param_value = int(param_value_raw)
            # 处理浮点数
            else:
                param_value = float(param_value_raw)
        # 解析 xxx=string 的形式
        else:
            param_value = str(param_value_raw)
        return param_type, param_value


class GraphBuilderONNX:
    def __init__(self, output_tensors):
        self.output_tensors = output_tensors
        self.nodes = list()
        self.graph_def = None 
        self.input_tensor = None 
        self.epsilon_bn = 1e-5
        self.momentum_bn = 0.99
        self.alpha_lrelu = 0.1
        self.param_dict = OrderedDict()
        self.major_node_specs = list()
        self.bath_size = 1

    def build_onnx_graph(self, layer_configs, weights_file_path, verbose=True):
        # 遍历字典所有键，即网络的所有层
        for layer_name in layer_configs.keys():
            layer_dict = layer_configs[layer_name]
            # 根据层及其参数构建 ONNX 节点
            major_node_specs = self.make_onnx_node(layer_name, layer_dict)
            # 添加除输入层的其他层信息，包括层名及其通道数
            if major_node_specs is not None:
                self.major_node_specs.append(major_node_specs)

        # 处理 output 信息
        outputs = list()
        for tensor_name in self.output_tensors.keys():
            output_dims = [
                self.bath_size
            ] + self.output_tensors[tensor_name]
            output_tensor = helper.make_tensor_value_info(
                tensor_name, TensorProto.FLOAT, output_dims)
            outputs.append(output_tensor)

        # 加载模型权重信息
        inputs = [self.input_tensor]
        weight_loader = WeightLoader(weights_file_path)
        initializer = list()
        for layer_name in self.param_dict.keys():
            _, layer_type = layer_name.split("_", 1)
            params = self.param_dict[layer_name]

            # 加载卷积层权重
            if layer_type == "convolutional":
                initializer_layer, inputs_layer = \
                    weight_loader.load_conv_weights(params)
                initializer.extend(initializer_layer)
                inputs.extend(inputs_layer)

            # 加载上采样层权重
            elif layer_type == "upsample":
                initializer_layer, inputs_layer = \
                    weight_loader.load_resize_scales(params)
                initializer.extend(initializer_layer)
                inputs.extend(inputs_layer)
        del weight_loader

        self.graph_def = helper.make_graph(
            nodes=self.nodes, name="YOLOv3-608", inputs=inputs,
            outputs=outputs, initializer=initializer)
        if verbose:
            print(helper.printable_graph(self.graph_def))
        model_def = helper.make_model(self.graph_def, producer_name="Tao")
        return model_def 

    def make_onnx_node(self, layer_name, layer_dict):
        layer_type = layer_dict["type"]
        if self.input_tensor is None:
            # net 层对应输入层的参数
            if layer_type == "net":
                major_node_output_name, major_node_output_channels = \
                    self.make_input_tensor(layer_name, layer_dict)
                major_node_specs = MajorNodeSpecs(
                    major_node_output_name, major_node_output_channels)
            else:
                raise ValueError("The first node has to be of type 'net'.")
        # 处理除输入层的其他层
        else:
            node_creators = dict()
            node_creators["convolutional"] = self.make_conv_node
            node_creators["shortcut"]      = self.make_shortcut_node
            node_creators["route"]         = self.make_route_node
            node_creators["upsample"]      = self.make_resize_node 

            if layer_type in node_creators.keys():
                major_node_output_name, major_node_output_channels = \
                    node_creators[layer_type](layer_name, layer_dict)
                major_node_specs = MajorNodeSpecs(
                    major_node_output_name, major_node_output_channels)
            else:
                print(f"[INFO] Layer of type {layer_type} not supported.")
                major_node_specs = MajorNodeSpecs(layer_name, None)
        # 返回当前节点信息
        return major_node_specs

    def make_input_tensor(self, layer_name, layer_dict):
        batch_size = layer_dict["batch"]
        channels   = layer_dict["channels"]
        height     = layer_dict["height"]
        width      = layer_dict["width"]
        self.batch_size = batch_size
        # 创建输入
        input_tensor = helper.make_tensor_value_info(
            name=str(layer_name), 
            elem_type=TensorProto.FLOAT, 
            shape=[batch_size, channels, height, width])
        self.input_tensor = input_tensor
        # 返回层名和当前层通道数
        return layer_name, channels

    def get_previous_node_specs(self, target_index=-1):
        """ 获取当前节点的前驱节点 """
        previous_node = None 
        for node in self.major_node_specs[target_index::-1]:
            if node.created_onnx_node:
                previous_node = node 
                break 
        assert previous_node is not None 
        return previous_node

    def make_conv_node(self, layer_name, layer_dict):
        """
        在构建卷积节点时, 包含卷积 + BN + 激活函数层
        1. 对于卷积, 除上层输入外多一个输入为权重信息
        2. 对于 BN, 除上层输入外多四个输入输入为 BN 层的四个参数信息
        3. 对于激活函数层，只有上层输入
        """
        # 获取前驱节点的信息
        previous_node_specs = self.get_previous_node_specs()

        # 卷积的一个输入来自前驱节点
        inputs = [previous_node_specs.name]
        previous_channels = previous_node_specs.channels
        kernel_size = layer_dict["size"]
        stride      = layer_dict["stride"]
        filters     = layer_dict["filters"]

        # 检测是否存在 BN 层
        batch_normalize = False 
        if "batch_normalize" in layer_dict.keys() and \
            layer_dict["batch_normalize"] == 1:
            batch_normalize = True 
        kernel_shape = [kernel_size, kernel_size]
        weights_shape = [filters, previous_channels] + kernel_shape
        conv_params = ConvParams(layer_name, batch_normalize, weights_shape)

        # 卷积的另一个输入来自权重信息
        weights_name = conv_params.generate_param_name("conv", "weights")
        inputs.append(weights_name)

        # 如果后面没有 BN 层，则加上来自偏置项的输入
        if not batch_normalize:
            bias_name = conv_params.generate_param_name("conv", "bias")
            inputs.append(bias_name)

        # 构建一个卷积节点
        strides = [stride, stride]
        dilations = [1, 1]        
        conv_node = helper.make_node(
            "Conv",
            inputs=inputs,
            outputs=[layer_name],          # outputs 和 name 字段内容相同
            kernel_shape=kernel_shape,
            strides=strides,
            auto_pad="SAME_LOWER",
            dilations=dilations,
            name=layer_name)               # name 和 outputs 字段内容相同
        self.nodes.append(conv_node)

        # 紧接着构建卷积层后续的 BN 层，BN 层的前驱节点信息
        inputs = [layer_name]
        layer_name_output = layer_name

        if batch_normalize:
            layer_name_bn = layer_name + "_bn"

            # BN 层的四个参数信息
            bn_param_suffixes = ["scale", "bias", "mean", "var"]
            for suffix in bn_param_suffixes:
                bn_param_name = conv_params.generate_param_name("bn", suffix)
                inputs.append(bn_param_name)
            
            # 构建一个 BN 节点
            batchnorm_node = helper.make_node(
                "BatchNormalization",
                inputs=inputs,
                outputs=[layer_name_bn],
                epsilon=self.epsilon_bn,
                momentum=self.momentum_bn,
                name=layer_name_bn)
            self.nodes.append(batchnorm_node)

            # 激活函数层的输入仅来自上层
            inputs = [layer_name_bn]
            layer_name_output = layer_name_bn
        
        # 再紧接着构建激活函数层
        if layer_dict["activation"] == "leaky":
            layer_name_lrelu = layer_name + "_lrelu"
            lrelu_node = helper.make_node(
                "LeakyRelu", 
                inputs=inputs, 
                outputs=[layer_name_lrelu],
                name=layer_name_lrelu,
                alpha=self.alpha_lrelu)
            self.nodes.append(lrelu_node)
            inputs = [layer_name_lrelu]
            layer_name_output = layer_name_lrelu
        elif layer_dict["activation"] == "linear":
            pass 
        else:
            print("[INFO] Activation is not supported.")
        
        self.param_dict[layer_name] = conv_params

        # 返回当前卷积（卷积 + BN + 激活）的输入以及卷积核个数，相当于输出通道数
        return layer_name_output, filters 

    def make_shortcut_node(self, layer_name, layer_dict):
        shortcut_index = layer_dict["from"]
        activation     = layer_dict["activation"]
        assert activation == "linear"

        # 获取前驱节点
        first_node_specs  = self.get_previous_node_specs()

        # 获取 concat 的另一个节点
        second_node_spces = self.get_previous_node_specs(
            target_index=shortcut_index)
        assert first_node_specs.channels == second_node_spces.channels
 
        # concat 的两个输入
        inputs = [first_node_specs.name, second_node_spces.name]

        # 构建节点
        shortcut_node = helper.make_node(
            "Add",
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name)
        self.nodes.append(shortcut_node)
        
        channels = first_node_specs.channels
        return layer_name, channels

    def make_route_node(self, layer_name, layer_dict):
        route_node_indexes = layer_dict["layers"]

        # layers 值只有一个时为恒等映射，没啥实际功能
        if len(route_node_indexes) == 1:
            split_index = route_node_indexes[0]
            assert split_index < 0
            split_index += 1
            self.major_node_specs = self.major_node_specs[:split_index]
            layer_name = None 
            channels = None 
        else:
            # route 层有两个输入
            inputs = list()
            channels = 0
            for index in route_node_indexes:
                if index > 0:
                    index += 1
                route_node_specs = self.get_previous_node_specs(
                    target_index=index)
                inputs.append(route_node_specs.name)
                channels += route_node_specs.channels
            assert inputs 
            assert channels > 0

            # 构建节点
            route_node = helper.make_node(
                "Concat",
                axis=1,
                inputs=inputs,
                outputs=[layer_name],
                name=layer_name)
            self.nodes.append(route_node)
        return layer_name, channels

    def make_resize_node(self, layer_name, layer_dict):
        # 上采样倍数
        re_scale_factors = float(layer_dict["stride"])
        scales = np.array(
            [1.0, 1.0, re_scale_factors, re_scale_factors]).astype(np.float32)
        
        # 获取前驱节点信息
        previous_node_specs = self.get_previous_node_specs()
        inputs = [previous_node_specs.name]
        
        channels = previous_node_specs.channels
        assert channels > 0
        resize_params = ResizeParams(layer_name, scales)

        # upsample 节点有两个输入，一个是 roi 一个是 scale
        roi_name = resize_params.generate_roi_name()
        inputs.append(roi_name)
        scales_name = resize_params.generate_param_name()
        inputs.append(scales_name)

        # 构建节点
        resize_node = helper.make_node(
            "Resize",
            coordinate_transformation_mode="asymmetric",
            mode="nearest",
            nearest_mode="floor",
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name)
        self.nodes.append(resize_node)
        self.param_dict[layer_name] = resize_params
        return layer_name, channels


class MajorNodeSpecs:
    def __init__(self, name, channels):
        self.name = name 
        self.channels = channels 
        self.created_onnx_node = False 
        if name is not None and isinstance(channels, int) and channels > 0:
            self.created_onnx_node = True 


class ConvParams:
    def __init__(self, node_name, batch_normalize, conv_weight_dims):
        self.node_name = node_name 
        self.batch_normalize = batch_normalize
        assert len(conv_weight_dims) == 4
        self.conv_weight_dims = conv_weight_dims

    def generate_param_name(self, param_category, suffix):
        assert suffix
        assert param_category in ["bn", "conv"]
        assert suffix in ["scale", "mean", "var", "weights", "bias"]
        if param_category == "bn":
            assert self.batch_normalize
            assert suffix in ["scale", "bias", "mean", "var"]
        elif param_category == "conv":
            assert suffix in ["weights", "bias"]
            if suffix == "bias":
                assert not self.batch_normalize
        param_name = self.node_name + "_" + param_category + "_" + suffix
        return param_name 


class WeightLoader:
    def __init__(self, weights_file_path):
        self.weights_file = self.open_weights_file(weights_file_path)

    def open_weights_file(self, weights_file_path):
        weights_file = open(weights_file_path, "rb")
        length_header = 5
        np.ndarray(shape=(length_header,), 
                   dtype="int32", buffer=weights_file.read(length_header * 4))
        return weights_file

    def load_resize_scales(self, resize_params):
        initializer = list()
        inputs = list()
        name  = resize_params.generate_param_name()
        shape = resize_params.value.shape
        data  = resize_params.value 
        scale_init = helper.make_tensor(name, TensorProto.FLOAT, shape, data)
        scale_input = helper.make_tensor_value_info(
            name, TensorProto.FLOAT, shape)
        initializer.append(scale_init)
        inputs.append(scale_input)

        rank = 4
        roi_name = resize_params.generate_roi_name()
        roi_input = helper.make_tensor_value_info(
            roi_name, TensorProto.FLOAT, [rank])
        roi_init = helper.make_tensor(
            roi_name, TensorProto.FLOAT, [rank], [0, 0, 0, 0])
        initializer.append(roi_init)
        inputs.append(roi_input)

        return initializer, inputs
    
    def load_conv_weights(self, conv_params):
        initializer = list()
        inputs = list()
        if conv_params.batch_normalize:
            bias_init, bias_input         = self.create_param_tensors(
                conv_params, "bn", "bias")
            bn_scale_init, bn_scale_input = self.create_param_tensors(
                conv_params, "bn", "scale")
            bn_mean_init, bn_mean_input   = self.create_param_tensors(
                conv_params, "bn", "mean")
            bn_var_init, bn_var_input     = self.create_param_tensors(
                conv_params, "bn", "var")
            initializer.extend(
                [bn_scale_init, bias_init, bn_mean_init, bn_var_init])
            inputs.extend(
                [bn_scale_input, bias_input, bn_mean_input, bn_var_input])
        else:
            bias_init, bias_input = self.create_param_tensors(
                conv_params, "conv", "bias")
            initializer.append(bias_init)
            inputs.append(bias_input)
        conv_init, conv_input = self.create_param_tensors(
            conv_params, "conv", "weights")
        initializer.append(conv_init)
        inputs.append(conv_input)
        return initializer, inputs 

    def create_param_tensors(self, conv_params, param_category, suffix):
        param_name, param_data, param_data_shape = self.load_one_param_type(
            conv_params, param_category, suffix)

        # 加载指定节点名的权重
        initializer_tensor = helper.make_tensor(
            param_name, TensorProto.FLOAT, param_data_shape, param_data)

        # 得到指定输入的信息
        input_tensor = helper.make_tensor_value_info(
            param_name, TensorProto.FLOAT, param_data_shape)

        return initializer_tensor, input_tensor

    def load_one_param_type(self, conv_params, param_category, suffix):
        param_name = conv_params.generate_param_name(param_category, suffix)

        # 卷积核的维度
        channels_out, channels_in, filter_h, filter_w = \
            conv_params.conv_weight_dims
        
        # BN 层只有一个参数
        if param_category == "bn":
            param_shape = [channels_out]
        # 卷积层有四个或一个参数
        elif param_category == "conv":
            if suffix == "weights":
                param_shape = [channels_out, channels_in, filter_h, filter_w]
            elif suffix == "bias":
                param_shape = [channels_out]

        # 参数填充
        param_size = np.product(np.array(param_shape))
        param_data = np.ndarray(shape=param_shape, dtype="float32",
                                buffer=self.weights_file.read(param_size * 4))
        param_data = param_data.flatten().astype(float)
        return param_name, param_data, param_shape


class ResizeParams:
    def __init__(self, node_name, value):
        self.node_name = node_name 
        self.value = value 

    def generate_param_name(self):
        param_name = self.node_name + "_" + "scale"
        return param_name

    def generate_roi_name(self):
        param_name = self.node_name + "_" + "roi"
        return param_name


def main():
    # 当前路径
    curr_path = os.getcwd()
    # 配置文件
    cfg_file_path = "yolov3.cfg"
    # 转换过程中支持的层
    supported_layers = ["net", "convolutional", "shortcut", "route", "upsample"]
    # 构建解析器
    parser = DarkNetParser(supported_layers)
    layer_configs = parser.parser_cfg_file(
        os.path.join(curr_path, cfg_file_path))
    del parser

    # 三种尺度输出的节点名
    output_tensor_dims = OrderedDict()
    output_tensor_dims["082_convolutional"] = [255, 19, 19]
    output_tensor_dims["094_convolutional"] = [255, 38, 38]
    output_tensor_dims["106_convolutional"] = [255, 76, 76]

    # 新建构建器对象
    builder = GraphBuilderONNX(output_tensor_dims)

    # 权重文件
    weights_file_path = "yolov3.weights"

    # 生成 ONNX 模型
    yolov3_model_def = builder.build_onnx_graph(
        layer_configs=layer_configs, 
        weights_file_path=os.path.join(curr_path, weights_file_path))
    del builder 

    onnx.checker.check_model(yolov3_model_def)

    output_file_path = "yolov3.onnx"
    onnx.save(yolov3_model_def, output_file_path)


if __name__ == "__main__":
    main()
