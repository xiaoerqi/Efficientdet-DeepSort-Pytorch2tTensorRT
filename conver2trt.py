import tensorrt as trt
import argparse
import torch
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import time

from common import get_weights, GiB
from effientdet_api import EfficientNetBackBone, BiFPN, Regressor, Classifier

use_cuda = True
use_float16 = False
MAX_BATCH_SIZE=4

class ModelData(object):
    INPUT_NAME = "data"
    INPUT_SHAPE = (896, 896, 3)   #(h , w, c)
    DTYPE = trt.float32

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
def efficientdet_network(network, weights):
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)
    print("inputTensor->", input_tensor.shape)

    #HWC to CHW
    shuffleLayer = network.add_shuffle(input_tensor)  
    shuffleLayer.first_transpose = (2, 0, 1)
    
    p3, p4, p5 = EfficientNetBackBone(network, weights, shuffleLayer.get_output(0))
    p3_in, p4_in, p5_in, p6_in, p7_in = BiFPN(network, weights, [p3, p4, p5])

    regressor_cat_feats = Regressor(network, weights, [p3_in, p4_in, p5_in, p6_in, p7_in])
    classifier_cat_feats = Classifier(network, weights, [p3_in, p4_in, p5_in, p6_in, p7_in])
    network.mark_output(shuffleLayer.get_output(0))
    network.mark_output(regressor_cat_feats.get_output(0))
    network.mark_output(classifier_cat_feats.get_output(0))

def build_engine(weights, engine_file_path, precision, batch_size):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, builder.create_builder_config() as config:
        config.max_workspace_size = GiB(1)
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        config.set_flag(trt.BuilderFlag.FP16)
        if precision == 'int8':
            from Calibrator import MyCalibrator
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = MyCalibrator(20,(1,896,896,3),"data/tensorrtx-int8calib-data/coco_calib","cache.txt")
        # Populate the network using weights from the PyTorch model.
        efficientdet_network(network, weights)
        # Build and return an engine.
        builder.max_batch_size = batch_size
        if precision == 'fp16':
            builder.fp16_mode = True  # alternative: builder.platform_has_fast_fp16
        engine = builder.build_engine(network, config)
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine_file_path', type=str, default='tensorrt_engine/efficientdet.engine', help='initial engine_file_path')
    parser.add_argument('--weight_path', type=str, default='weights/efficientdet-d3.pth', help='initial weights path')
    parser.add_argument('--precision', type=str, default='fp32', help='initial precision fp32, fp16, int8')
    parser.add_argument('--batch_size', type=int, default=1, help='total batch size for all GPUs')

    opt = parser.parse_args()
    engine_file_path = opt.engine_file_path
    weight_path = opt.weight_path
    precision = opt.precision
    batch_size = opt.batch_size

    weights = get_weights(weight_path)
    engine_name = '_' + precision + '_' + str(batch_size) + '.engine'
    # build_engine(weights, engine_file_path)
    build_engine(weights, engine_file_path.replace(".engine",engine_name), precision, batch_size)