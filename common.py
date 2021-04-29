import cv2
import numpy as np
import tensorrt as trt
import torch
import struct
import pycuda.driver as cuda
import pycuda.autoinit

import sys
sys.path.append("../")
from backbone import EfficientDetBackbone

def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    canvas = np.zeros((height, height, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h:
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h),
                               interpolation=interpolation)

    padding_h = height - new_h
    padding_w = width - new_w

    if c > 1:
        canvas[:new_h, :new_w] = image
    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image

    return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h,


def preprocess(*image_path, max_size=512, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    ori_imgs = [cv2.imread(img_path) for img_path in image_path]
    normalized_imgs = [(img[..., ::-1] / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img, max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas


def convBlock(network, weights, inputTensor, num_output_maps, ksize, stride, group, lname, pre_padding = (0, 0), post_padding = (0, 0)):
    conv_w = weights[lname + '.weight'].numpy()
    conv_b = None
    if lname + '.bias' in weights:
        conv_b = weights[lname + '.bias'].numpy()
        
    conv = network.add_convolution(input=inputTensor, num_output_maps=num_output_maps, kernel_shape=(
        ksize, ksize), kernel=conv_w, bias=conv_b)
    conv.stride_nd  = ((stride, stride))
    conv.pre_padding  = pre_padding
    conv.post_padding  = post_padding
    conv.num_groups  = group
    return conv

def maxPoolingBlock(network, weights, inputTensor,kernel_size, stride, pre_padding = (0, 0), post_padding = (0,0)):
    padding = network.add_padding_nd(inputTensor, pre_padding, post_padding)
    max_pool=network.add_pooling_nd(padding.get_output(0),trt.PoolingType.MAX,(kernel_size,kernel_size))
    max_pool.stride = (stride,stride)
    return max_pool

def bi_fpn_test():
    print("test")

def get_weights(weight_path):
    compound_coef = 3
    obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush']
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                 ratios=anchor_ratios, scales=anchor_scales)
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    model.requires_grad_(False)
    model.eval()
    return model.state_dict()

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def load_image(img, pagelocked_buffer):
    # Select an image at random to be the test case.
    np.copyto(pagelocked_buffer, img)

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def GiB(val):
    return val * 1 << 30

def get_output_message(output):
    #print('Output->', output)
    print('Output Shape->', output.shape)
    print('Output Mean->', np.mean(output))
    print('Output var->', np.var(output))
    print('Output std->', np.std(output))

def addBatchNorm2d(network,  weights, inputTensor, lname, eps = 1e-03): 
    in_gamma = weights[lname + '.weight'].numpy()        # in gamma
    in_beta  = weights[lname + '.bias'].numpy()          # in beta
    in_mean  = weights[lname + '.running_mean'].numpy()  # in mean
    in_var   = weights[lname + '.running_var'].numpy()   # in var sqrt
    len = list(weights[lname + ".running_var"].size())[0]
    in_var1   = np.sqrt(in_var + eps)
    in_scale = in_gamma / in_var1
    in_shift = - in_mean / in_var1 * in_gamma + in_beta
    power = np.ones((len),dtype=np.float32)
    bn       = network.add_scale(inputTensor, mode=trt.ScaleMode.CHANNEL, shift=in_shift, scale=in_scale)
    return bn

def addSwish(network, inputTensor):
    sigmoid = network.add_activation(inputTensor, trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(inputTensor, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
    return swish

def addUpsample(network, inputTensor,scales, is_align_corners = False):
    upsample = network.add_resize(inputTensor)
    upsample.resize_mode = trt.ResizeMode.NEAREST
    upsample.scales = scales
    upsample.align_corners = is_align_corners
    return upsample

'''
def BBoxTransform(anchors, regression):
    y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
    x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
    ha = anchors[..., 2] - anchors[..., 0]
    wa = anchors[..., 3] - anchors[..., 1]

    w = regression[..., 3].exp() * wa
    h = regression[..., 2].exp() * ha

    y_centers = regression[..., 0] * ha + y_centers_a
    x_centers = regression[..., 1] * wa + x_centers_a

    ymin = y_centers - h / 2.
    xmin = x_centers - w / 2.
    ymax = y_centers + h / 2.
    xmax = x_centers + w / 2.

    return [xmin, ymin, xmax, ymax]


def Anchors(image, anchor_scale=4.0):
    strides = [8, 16, 32, 64, 128]
    scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    last_shape = None
    image_shape = image.shape[2:]

    if last_shape is None or last_shape != image_shape:
        last_shape = image_shape

    boxes_all = []
    for stride in strides:
        boxes_level = []
        for scale, ratio in (scales, ratios):
            if image_shape[1] % stride != 0:
                raise ValueError('input size must be divided by the stride.')
            base_anchor_size = anchor_scale * stride * scale
            anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0
            anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0

            x = np.arange(stride / 2, image_shape[1], stride)
            y = np.arange(stride / 2, image_shape[0], stride)
            xv, yv = np.meshgrid(x, y)
            xv = xv.reshape(-1)
            yv = yv.reshape(-1)

            # y1,x1,y2,x2
            boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                               yv + anchor_size_y_2, xv + anchor_size_x_2))
            boxes = np.swapaxes(boxes, 0, 1)
            boxes_level.append(np.expand_dims(boxes, axis=1))
        # concat anchors on the same level to the reshape NxAx4
        boxes_level = np.concatenate(boxes_level, axis=1)
        boxes_all.append(boxes_level.reshape([-1, 4]))

    anchor_boxes = np.vstack(boxes_all)
    return anchor_boxes
'''

