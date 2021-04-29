import numpy as np
import tensorrt as trt
import argparse
import time
import os
import torch
import cv2

from common import preprocess, load_image, do_inference, get_output_message, allocate_buffers

import sys
sys.path.append("../")
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes, Anchors
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

input_size = 896
logger = trt.Logger(trt.Logger.INFO)
compound_coef = 3
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
threshold = 0.2
iou_threshold = 0.2
color_list = standard_to_bgr(STANDARD_COLORS)

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


def display(preds, imgs, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])


        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            cv2.imwrite(f'test/img_inferred_d{compound_coef}_this_repo_{i}.jpg', imgs[i])

def get_engine(engine_file_path):
    if os.path.isfile(engine_file_path):
        with open(engine_file_path, 'rb') as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine( f.read() )
            if engine == None:
                exit()
        return engine

def run(framed_imgs, engine, inputs, outputs, bindings, stream, context, batch_size):
    load_image(framed_imgs, pagelocked_buffer=inputs[0].host)
    x, regression, classification = do_inference(context, bindings=bindings, inputs=inputs,outputs=outputs,stream=stream, batch_size=batch_size)
    x_shape = tuple([batch_size]) + tuple(engine.get_binding_shape(1))
    x = x.reshape(x_shape)

    regression_shape = tuple([batch_size]) + tuple(engine.get_binding_shape(2))
    regression = regression.reshape(regression_shape)
    classification_shape = tuple([batch_size]) + tuple(engine.get_binding_shape(3))
    classification = classification.reshape(classification_shape)
    return x, regression, classification

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='test/images/img.png', help='initial image path')
    parser.add_argument('--engine_file_path', type=str, default='tensorrt_engine/efficientdet_fp32_1.engine', help='initial engine_file_path')
    parser.add_argument('--batch_size', type=int, default=1, help='total batch size for all GPUs')
    opt = parser.parse_args()

    img_path = opt.img_path
    engine_file_path = opt.engine_file_path
    batch_size = opt.batch_size
    engine = get_engine(engine_file_path)
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    context = engine.create_execution_context()

    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)
    framed_imgs = np.array([framed_imgs] * batch_size).reshape(-1)
    framed_imgs = np.array(framed_imgs).reshape(-1)

    for i in range(100):
        t1 = int(round(time.time() * 1000))
        x, regression, classification = run(framed_imgs, engine, inputs, outputs, bindings, stream, context, batch_size)
        t2 = int(round(time.time() * 1000))
        print('modeltime: ', t2-t1, 'ms')

        anchors = Anchors(anchor_scale=4.0,
                                   pyramid_levels=(torch.arange(5) + 3).tolist(),
                                   compound_coef=compound_coef, num_classes=len(obj_list),
                                   ratios=anchor_ratios, scales=anchor_scales)

        x = torch.unsqueeze(torch.from_numpy(x[0]), 0)
        anchors = anchors(x, torch.float32)
        
        regression = torch.from_numpy(regression[0])
        classification = torch.from_numpy(classification[0])
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)

        out = invert_affine(framed_metas, out)
        
        if i == 0:
            display(out, ori_imgs, imshow=False, imwrite=True)
    