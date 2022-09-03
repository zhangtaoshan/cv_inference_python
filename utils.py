from PIL import Image
import math 
import numpy as np 
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


GATEGORY_NUM = 80


def load_label_categories(label_file_path):
    categories = [line.rstrip("\n") for line in open(label_file_path)]
    return categories

ALL_CATES = load_label_categories("coco_labels.txt")


class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host   = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs   = []
    outputs  = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size  = trt.volume(
            engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(
            engine.get_binding_dtype(binding))
        # 分配主机内存和设备内存
        host_mem   = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # 绑定设备内存
        bindings.append(int(device_mem))
        # 输入输出绑定
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream 


def do_inference(context, bindings, inputs, outputs, stream):
    # 将输入数据从主机拷贝到设备
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # 推理
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # 将输出数据从设备拷贝到主机
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # 同步流
    stream.synchronize()
    # 仅返回主机上的输出
    return [out.host for out in outputs]


class PreprocessYOLO:
    def __init__(self, input_resolution):
        self.input_resolution = input_resolution

    def preprocess(self, image_path):
        image_raw, image_resized = self.load_and_resize(image_path)
        image_preprocesed = self.shuffle_and_normalize(image_resized)
        return image_raw, image_preprocesed

    def load_and_resize(self, image_path):
        image_raw = Image.open(image_path)
        new_resolution = (self.input_resolution[1], self.input_resolution[0])
        image_resized = image_raw.resize(new_resolution, resample=Image.CUBIC)
        image_resized = np.array(image_resized, dtype=np.float32, order="C")
        return image_raw, image_resized

    def shuffle_and_normalize(self, image):
        # 归一化
        image /= 255.0
        # (w,h,c) -> (c,h,w)
        image = np.transpose(image, [2, 0, 1])
        # (c,h,w) -> (n,c,h,w)
        image = np.expand_dims(image, axis=0)
        image = np.array(image, dtype=np.float32, order="C")
        return image 


class PostprocessYOLO:
    def __init__(self, masks, anchors, obj_thred, nms_thred, input_resolution):
        self.masks = masks 
        self.anchors = anchors
        self.obj_threshold = obj_thred
        self.nms_threshold = nms_thred
        self.innput_resolution = input_resolution

    def process(self, outputs, resolution_raw):
        outputs_reshaped = list()
        for output in outputs:
            outputs_reshaped.append(self.reshape_output(output))
        
        boxes, categories, confidences = self.process_output(
            outputs_reshaped, resolution_raw)

        return boxes, categories, confidences

    def reshape_output(self, output):
        # (n, c, h, w) -> (n, h, w, c)
        output = np.transpose(output, [0, 2, 3, 1])
        _, height, width, _ = output.shape
        dim1, dim2 = height, width 
        dim3 = 3
        dim4 = 4 + 1 + GATEGORY_NUM
        return np.reshape(output, (dim1, dim2, dim3, dim4))

    def process_output(self, outputs_reshaped, resolution_raw):
        boxes, categories, confidences = list(), list(), list()
        for output, mask in zip(outputs_reshaped, self.masks):
            box, category, confidence = self.process_feats(output, mask)
            box, category, confidence = self.filter_boxes(
                box, category, confidence)
            boxes.append(box)
            categories.append(category)
            confidences.append(confidence)
        
        boxes       = np.concatenate(boxes)
        categories  = np.concatenate(categories)
        confidences = np.concatenate(confidences)

        # 将框还原为原图尺寸
        width, height = resolution_raw
        image_dims = [width, height, width, height]
        boxes = boxes * image_dims

        # 执行 nms 操作
        nms_boxes, nms_categories, nscores = list(), list(), list()
        for category in set(categories):
            idxs = np.where(categories == category)
            box = boxes[idxs]
            category = categories[idxs]
            confidence = confidences[idxs]

            keep = self.nms_boxes(box, confidence)

            nms_boxes.append(box[keep])
            nms_categories.append(category[keep])
            nscores.append(confidence[keep])
        
        if not nms_categories and not nscores:
            return None, None, None 

        boxes       = np.concatenate(nms_boxes)
        categories  = np.concatenate(nms_categories)
        confidences = np.concatenate(nscores)

        return boxes, categories, confidences

    def process_feats(self, output_reshaped, mask):
        def sigmoid(value):
            return 1.0 / (1.0 + math.exp(-value))

        def exponential(value):
            return math.exp(value)

        sigmoid_v     = np.vectorize(sigmoid) 
        exponential_v = np.vectorize(exponential)

        grid_h, grid_w, _, _ = output_reshaped.shape 

        anchors = [self.anchors[i] for i in mask]

        # reshape 成 (n,h,w,num_anchors,box_params)
        anchors_tensor = np.reshape(anchors, [1, 1, len(anchors), 2])
        box_xy = sigmoid_v(output_reshaped[..., :2])
        box_wh = exponential_v(output_reshaped[..., 2:4]) * anchors_tensor
        box_confidence = sigmoid_v(output_reshaped[..., 4])

        box_confidence  = np.expand_dims(box_confidence, axis=-1)
        box_class_probs = sigmoid_v(output_reshaped[..., 5:])

        col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

        col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)

        box_xy += grid 
        box_xy /= (grid_w, grid_h)
        box_wh /= self.innput_resolution
        box_xy -= box_wh / 2.0
        boxes = np.concatenate((box_xy, box_wh), axis=-1)

        return boxes, box_confidence, box_class_probs


    def filter_boxes(self, boxes, box_confidence, box_class_probs):
        box_scores  = box_confidence * box_class_probs
        box_classes = np.argmax(box_scores, axis=-1)
        box_class_scores = np.max(box_scores, axis=-1)
        pos = np.where(box_class_scores >= self.obj_threshold)

        boxes   = boxes[pos]
        classes = box_classes[pos]
        scores  = box_class_scores[pos] 

        return boxes, classes, scores 

    def nms_boxes(self, boxes, box_confidences):
        """ nms """
        x_coord = boxes[:, 0]
        y_coord = boxes[:, 1]
        width   = boxes[:, 2] 
        height  = boxes[:, 3]

        areas = width * height 
        ordered = box_confidences.argsort()[::-1]

        keep = list()
        while ordered.size > 0:
            i = ordered[0]
            keep.append(i)
            xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
            yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
            xx2 = np.minimum(x_coord[i] + width[i], 
                             x_coord[ordered[1:]] + width[ordered[1:]])
            yy2 = np.minimum(y_coord[i] + height[i],
                             y_coord[ordered[1:]] + height[ordered[1:]])

            width1  = np.maximum(0.0, xx2 - xx1 + 1)
            height1 = np.maximum(0.0, yy2 - yy1 + 1) 
            intersection = width1 * height1
            union = areas[i] + areas[ordered[1:]] - intersection

            iou = intersection / union

            indexes = np.where(iou <= self.nms_threshold)[0]
            ordered = ordered[indexes + 1]

        keep = np.array(keep)
        return keep 
