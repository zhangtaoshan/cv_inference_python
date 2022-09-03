import os 
import tensorrt as trt 
from PIL import ImageDraw
import numpy as np 

from utils import PreprocessYOLO, PostprocessYOLO, ALL_CATES
from utils import allocate_buffers, do_inference


TRT_LOGGER = trt.Logger()


def get_engine(onnx_file_path, engine_file_path=""):
    # 如果不指定 engine_file_path 则通过 build_engine 生成 engine 文件
    def build_engine():
        # 基于 INetworkDefinition 构建 ICudaEngine
        builder = trt.Builder(TRT_LOGGER)
        # 基于 INetworkDefinition 和 IBuilderConfig 构建 engine
        network = builder.create_network(
            1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        # 构建 builder 的配置对象
        config = builder.create_builder_config()
        # 构建 ONNX 解析器
        parser = trt.OnnxParser(network, TRT_LOGGER)
        # 构建 TensorRT 运行时
        runtime = trt.Runtime(TRT_LOGGER)
        # 参数设置
        config.max_workspace_size = 1 << 28 # 256MiB
        builder.max_batch_size = 1
        # 解析 onnx 模型
        if not os.path.exists(onnx_file_path):
            print(
                f"[INFO] ONNX file {onnx_file_path} not found.")
        print(f"[INFO] Loading ONNX file from {onnx_file_path}.")
        with open(onnx_file_path, "rb") as model:
            print("[INFO] Beginning ONNX file parsing.")
            if not parser.parse(model.read()):
                print("[ERROR] Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None 
        # 根据 yolov3.onnx，reshape 输入数据的形状
        network.get_input(0).shape = [1, 3, 608, 608]
        print("[INFO] Completed parsing of ONNX file.")
        print(f"[INFO] Building an engine from {onnx_file_path}.")
        # 序列化模型
        plan = builder.build_serialized_network(network, config)
        # 反序列化
        engine = runtime.deserialize_cuda_engine(plan)
        print("[INFO] Completed creating engine.")
        # 写入文件
        with open(engine_file_path, "wb") as f:
            f.write(plan)
        return engine 

    if os.path.exists(engine_file_path):
        print(f"[INFO] Reading engine from {engine_file_path}.")
        with open(engine_file_path, "rb") as f:
            with trt.Runtime(TRT_LOGGER) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def draw_bboxes(image_raw, bboxes, confs, cates, all_cates, bbox_color="blue"):
    draw = ImageDraw.Draw(image_raw)
    for box, score, category in zip(bboxes, confs, cates):
        x_coord, y_coord, width, height = box 
        left   = max(0, np.floor(x_coord + 0.5).astype(int))
        top    = max(0, np.floor(y_coord + 0.5).astype(int))
        right  = min(
            image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(
            image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text(
            (left, top - 12), 
            "{0} {1:.2f}".format(all_cates[category], score), fill=bbox_color)
    return image_raw


def main():
    """
    1. 根据训练流程对输入图像预处理
    2. 生成 engine 对象。如果已存在则直接反序列化，否则先序列化存储到本地
    3. 分配内存，包括在主机和设备上
    4. 推理，数据：主机->设备（推理）->主机，调用异步推理函数接口后需同步
    5. 对网络输出结果后处理
    """
    onnx_file_path   = "yolov3.onnx"
    engine_file_path = "yolov3.trt"
    input_image_path = "sheep.jpg"
    input_resolution = (608, 608)
    curr_path = os.getcwd()

    # YOLOv3 的输入预处理
    preprocesser = PreprocessYOLO(input_resolution)
    image_raw, image = preprocesser.preprocess(
        os.path.join(curr_path, input_image_path))
    # 保留图像的原始尺寸，后续后处理会用到
    shape_origin = image_raw.size 
    
    # YOLOv3 的输出张量形状
    output_shapes = [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)]
    # 基于 TensorRT 的推理
    trt_outputs = []
    engine = get_engine(
        onnx_file_path=os.path.join(curr_path, onnx_file_path),
        engine_file_path=os.path.join(curr_path, engine_file_path))
    # 创建推理上下文
    context = engine.create_execution_context()
    # 分配内存
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    print(f"[INFO] Running inference on {input_image_path}.")
    # 为图像分配主机内存
    inputs[0].host = image 
    # 推理并获得输出
    trt_outputs = do_inference(context, bindings, inputs, outputs, stream)
    # 由于 trt_outputs 为展开的张量，这里将其 reshape
    trt_outputs = [output.reshape(shape) \
        for output, shape in zip(trt_outputs, output_shapes)]

    # YOLOv3 的输出后处理参数
    yolov3_postprocess_args = {
        "masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],
        "anchors": [
            (10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)],
        "obj_thred": 0.6,
        "nms_thred": 0.5,
        "input_resolution": input_resolution 
    }

    # YOLOv3 的输出预处理
    postprocesser = PostprocessYOLO(**yolov3_postprocess_args)

    # 获得后处理的框信息
    boxes, classes, scores = postprocesser.process(trt_outputs, (shape_origin))
    
    # 绘制和保存
    obj_detected_img = draw_bboxes(image_raw, boxes, scores, classes, ALL_CATES)
    output_image_path = "sheep_bboxes.png"
    obj_detected_img.save(output_image_path, "PNG")
    print(f"[INFO] Saved image with bounding boxes to {output_image_path}.")


if __name__ == "__main__":
    main()
