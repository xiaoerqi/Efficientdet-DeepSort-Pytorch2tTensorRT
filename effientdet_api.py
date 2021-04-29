import numpy as np
import tensorrt as trt

from common import addUpsample, maxPoolingBlock, addSwish, preprocess, convBlock, get_weights, allocate_buffers, load_image, do_inference, GiB, get_output_message, addBatchNorm2d

def EfficientNetBackBone(network, weights, input_tenor):
    #-----------------------------------------------------------------------------------------------# EfficientDetBackbone
    #-----------------------------------------------------------------------------------------------# EfficientNet
    #-----------------------------------------------------------------------------------------------# extract_features
    # Stem
    backbone_net_model_conv_stem_conv = convBlock(network = network, weights = weights, inputTensor = input_tenor, num_output_maps = 40, ksize = 3, stride = 2, pre_padding = (0, 0), post_padding = (1, 1), group = 1, lname = 'backbone_net.model._conv_stem.conv')
    backbone_net_model_bn0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_conv_stem_conv.get_output(0), lname = 'backbone_net.model._bn0')
    backbone_net_model_swish0 = addSwish(network, inputTensor = backbone_net_model_bn0.get_output(0))

    # Blocks
    #-----------------------------------------------------------------------------------------------# MBConvBlock0
    backbone_net_model_blocks0_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_swish0.get_output(0), num_output_maps = 40, ksize = 3, stride = 1, group = 40, lname = 'backbone_net.model._blocks.0._depthwise_conv.conv', pre_padding = (1, 1), post_padding = (1, 1))
    backbone_net_model_blocks0_bn1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks0_depthwise_conv.get_output(0), lname = 'backbone_net.model._blocks.0._bn1')
    backbone_net_model_blocks0_swish1 = addSwish(network, inputTensor = backbone_net_model_blocks0_bn1.get_output(0))
    backbone_net_model_blocks0_average_pooling0 = network.add_pooling_nd(backbone_net_model_blocks0_swish1.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (224, 224))
    backbone_net_model_blocks0_average_pooling0.stride = (224, 224)
    backbone_net_model_blocks0_average_pooling1 = network.add_pooling_nd(backbone_net_model_blocks0_average_pooling0.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (2, 2))
    backbone_net_model_blocks0_average_pooling1.stride = (2, 2)
    backbone_net_model_blocks0_se_reduce_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks0_average_pooling1.get_output(0), num_output_maps = 10, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.0._se_reduce.conv')
    backbone_net_model_blocks0_swish2 = addSwish(network, inputTensor = backbone_net_model_blocks0_se_reduce_conv.get_output(0))
    backbone_net_model_blocks0_se_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks0_swish2.get_output(0), num_output_maps = 40, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.0._se_expand.conv')
    backbone_net_model_blocks0_sigmoid0 = network.add_activation(input = backbone_net_model_blocks0_se_expand_conv.get_output(0), type=trt.ActivationType.SIGMOID)
    backbone_net_model_blocks0_prod0 = network.add_elementwise(backbone_net_model_blocks0_swish1.get_output(0), backbone_net_model_blocks0_sigmoid0.get_output(0), trt.ElementWiseOperation.PROD)
    backbone_net_model_blocks0_project_conv_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks0_prod0.get_output(0), num_output_maps = 24, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.0._project_conv.conv')
    backbone_net_model_blocks0_bn2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks0_project_conv_conv.get_output(0), lname = 'backbone_net.model._blocks.0._bn2')
    #-----------------------------------------------------------------------------------------------# MBConvBlock0
    #-----------------------------------------------------------------------------------------------# MBconvBlock
    
    backbone_net_model_blocks1_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks0_bn2.get_output(0), num_output_maps = 24, ksize = 3, stride = 1, group = 24, lname = 'backbone_net.model._blocks.1._depthwise_conv.conv',pre_padding = (1, 1), post_padding = (1, 1))
    
    backbone_net_model_blocks1_bn1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = 
    backbone_net_model_blocks1_depthwise_conv.get_output(0), lname = 'backbone_net.model._blocks.1._bn1')

    backbone_net_model_blocks1_swish1 = addSwish(network, inputTensor = backbone_net_model_blocks1_bn1.get_output(0))
    backbone_net_model_blocks1_average_pooling0 = network.add_pooling_nd(backbone_net_model_blocks1_swish1.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (224, 224))
    backbone_net_model_blocks1_average_pooling0.stride = (224, 224)
    backbone_net_model_blocks1_average_pooling1 = network.add_pooling_nd(backbone_net_model_blocks1_average_pooling0.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (2, 2))
    backbone_net_model_blocks1_average_pooling1.stride = (2, 2)
    backbone_net_model_blocks1_se_reduce_conv = convBlock(network = network, weights = weights, 
                                                          inputTensor = backbone_net_model_blocks1_average_pooling1.get_output(0), 
                                                          num_output_maps = 6, ksize = 1, stride = 1, group = 1, 
                                                          lname = 'backbone_net.model._blocks.1._se_reduce.conv')
    backbone_net_model_blocks1_swish2 = addSwish(network, inputTensor = backbone_net_model_blocks1_se_reduce_conv.get_output(0))
    backbone_net_model_blocks1_se_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks1_swish2.get_output(0), num_output_maps = 24, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.1._se_expand.conv')
    backbone_net_model_blocks1_sigmoid0 = network.add_activation(input = backbone_net_model_blocks1_se_expand_conv.get_output(0), type=trt.ActivationType.SIGMOID)
    backbone_net_model_blocks1_prod0 = network.add_elementwise(backbone_net_model_blocks1_swish1.get_output(0), backbone_net_model_blocks1_sigmoid0.get_output(0), trt.ElementWiseOperation.PROD)
    backbone_net_model_blocks1_project_conv_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks1_prod0.get_output(0), num_output_maps = 24, ksize = 1, stride = 1,  group = 1, lname = 'backbone_net.model._blocks.1._project_conv.conv')
    backbone_net_model_blocks1_bn2 = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks1_project_conv_conv.get_output(0), lname = 'backbone_net.model._blocks.1._bn2')
    backbone_net_model_blocks1_bn2 = network.add_elementwise(backbone_net_model_blocks1_bn2.get_output(0), backbone_net_model_blocks0_bn2.get_output(0), trt.ElementWiseOperation.SUM)
    #-----------------------------------------------------------------------------------------------# MBconvBlock
    #-----------------------------------------------------------------------------------------------# MBConvBlock2
    backbone_net_model_blocks2_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks1_bn2.get_output(0), 
                                                       num_output_maps = 144, ksize = 1, stride = 1, group = 1, 
                                                       lname = 'backbone_net.model._blocks.2._expand_conv.conv')
    backbone_net_model_blocks2_bn0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks2_expand_conv.get_output(0), lname = 'backbone_net.model._blocks.2._bn0')
    backbone_net_model_blocks2_swish0 = addSwish(network, inputTensor = backbone_net_model_blocks2_bn0.get_output(0))
    backbone_net_model_blocks2_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks2_swish0.get_output(0), 
                                                          num_output_maps = 144, ksize = 3, stride = 2, group = 144, 
                                                          lname = 'backbone_net.model._blocks.2._depthwise_conv.conv',pre_padding = (0, 0), post_padding = (1, 1))
    backbone_net_model_blocks2_bn1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = 
                                                     backbone_net_model_blocks2_depthwise_conv.get_output(0), lname = 'backbone_net.model._blocks.2._bn1')
    backbone_net_model_blocks2_swish1 = addSwish(network, inputTensor = backbone_net_model_blocks2_bn1.get_output(0))
    backbone_net_model_blocks2_average_pooling0 = network.add_pooling_nd(backbone_net_model_blocks2_swish1.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (224, 224))
    backbone_net_model_blocks2_average_pooling0.stride = (224, 224)
    backbone_net_model_blocks2_average_pooling1 = network.add_pooling_nd(backbone_net_model_blocks2_average_pooling0.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (2, 2))
    backbone_net_model_blocks2_average_pooling1.stride = (2, 2)
    backbone_net_model_blocks2_se_reduce_conv = convBlock(network = network, weights = weights, 
                                                          inputTensor = backbone_net_model_blocks2_average_pooling1.get_output(0), 
                                                          num_output_maps = 6, ksize = 1, stride = 1, group = 1, 
                                                          lname = 'backbone_net.model._blocks.2._se_reduce.conv')
    backbone_net_model_blocks2_swish2 = addSwish(network, inputTensor = backbone_net_model_blocks2_se_reduce_conv.get_output(0))
    backbone_net_model_blocks2_se_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks2_swish2.get_output(0), num_output_maps = 144, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.2._se_expand.conv')
    backbone_net_model_blocks2_sigmoid0 = network.add_activation(input = backbone_net_model_blocks2_se_expand_conv.get_output(0), type=trt.ActivationType.SIGMOID)
    backbone_net_model_blocks2_prod0 = network.add_elementwise(backbone_net_model_blocks2_swish1.get_output(0), backbone_net_model_blocks2_sigmoid0.get_output(0), trt.ElementWiseOperation.PROD)
    backbone_net_model_blocks2_project_conv_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks2_prod0.get_output(0), num_output_maps = 32, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.2._project_conv.conv')
    backbone_net_model_blocks2_bn2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks2_project_conv_conv.get_output(0), lname = 'backbone_net.model._blocks.2._bn2')
    #-----------------------------------------------------------------------------------------------# MBConvBlock2
    #-----------------------------------------------------------------------------------------------# extract_features
    #-----------------------------------------------------------------------------------------------# MBConvBlock3
    backbone_net_model_blocks3_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks2_bn2.get_output(0), 
                                                       num_output_maps = 192, ksize = 1, stride = 1, group = 1, 
                                                       lname = 'backbone_net.model._blocks.3._expand_conv.conv')
    backbone_net_model_blocks3_bn0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks3_expand_conv.get_output(0), lname = 'backbone_net.model._blocks.3._bn0')
    backbone_net_model_blocks3_swish0 = addSwish(network, inputTensor = backbone_net_model_blocks3_bn0.get_output(0))
    backbone_net_model_blocks3_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks3_swish0.get_output(0), 
                                                          num_output_maps = 192, ksize = 3, stride = 1, group = 192, 
                                                          lname = 'backbone_net.model._blocks.3._depthwise_conv.conv',pre_padding = (1, 1), post_padding = (1, 1))
    backbone_net_model_blocks3_bn1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = 
                                                     backbone_net_model_blocks3_depthwise_conv.get_output(0), lname = 'backbone_net.model._blocks.3._bn1')
    backbone_net_model_blocks3_swish1 = addSwish(network, inputTensor = backbone_net_model_blocks3_bn1.get_output(0))
    backbone_net_model_blocks3_average_pooling0 = network.add_pooling_nd(backbone_net_model_blocks3_swish1.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (224, 224))
    backbone_net_model_blocks3_average_pooling0.stride = (224, 224)
    backbone_net_model_blocks3_average_pooling1 = network.add_pooling_nd(backbone_net_model_blocks3_average_pooling0.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (2, 2))
    backbone_net_model_blocks3_average_pooling1.stride = (2, 2)
    backbone_net_model_blocks3_se_reduce_conv = convBlock(network = network, weights = weights, 
                                                          inputTensor = backbone_net_model_blocks3_average_pooling1.get_output(0), 
                                                          num_output_maps = 8, ksize = 1, stride = 1, group = 1, 
                                                          lname = 'backbone_net.model._blocks.3._se_reduce.conv')
    backbone_net_model_blocks3_swish2 = addSwish(network, inputTensor = backbone_net_model_blocks3_se_reduce_conv.get_output(0))
    backbone_net_model_blocks3_se_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks3_swish2.get_output(0), num_output_maps = 192, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.3._se_expand.conv')
    backbone_net_model_blocks3_sigmoid0 = network.add_activation(input = backbone_net_model_blocks3_se_expand_conv.get_output(0), type=trt.ActivationType.SIGMOID)
    backbone_net_model_blocks3_prod0 = network.add_elementwise(backbone_net_model_blocks3_swish1.get_output(0), backbone_net_model_blocks3_sigmoid0.get_output(0), trt.ElementWiseOperation.PROD)
    backbone_net_model_blocks3_project_conv_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks3_prod0.get_output(0), num_output_maps = 32, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.3._project_conv.conv')
    backbone_net_model_blocks3_bn2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks3_project_conv_conv.get_output(0), lname = 'backbone_net.model._blocks.3._bn2')
    backbone_net_model_blocks3_bn2 = network.add_elementwise(backbone_net_model_blocks3_bn2.get_output(0), backbone_net_model_blocks2_bn2.get_output(0), trt.ElementWiseOperation.SUM)
    #-----------------------------------------------------------------------------------------------# MBConvBlock3
    #-----------------------------------------------------------------------------------------------# MBConvBlock4
    backbone_net_model_blocks4_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks3_bn2.get_output(0), 
                                                       num_output_maps = 192, ksize = 1, stride = 1, group = 1, 
                                                       lname = 'backbone_net.model._blocks.4._expand_conv.conv')
    backbone_net_model_blocks4_bn0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks4_expand_conv.get_output(0), lname = 'backbone_net.model._blocks.4._bn0')
    backbone_net_model_blocks4_swish0 = addSwish(network, inputTensor = backbone_net_model_blocks4_bn0.get_output(0))
    backbone_net_model_blocks4_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks4_swish0.get_output(0), 
                                                          num_output_maps = 192, ksize = 3, stride = 1, group = 192, 
                                                          lname = 'backbone_net.model._blocks.4._depthwise_conv.conv',pre_padding = (1, 1), post_padding = (1, 1))
    backbone_net_model_blocks4_bn1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = 
                                                     backbone_net_model_blocks4_depthwise_conv.get_output(0), lname = 'backbone_net.model._blocks.4._bn1')
    backbone_net_model_blocks4_swish1 = addSwish(network, inputTensor = backbone_net_model_blocks4_bn1.get_output(0))
    backbone_net_model_blocks4_average_pooling0 = network.add_pooling_nd(backbone_net_model_blocks4_swish1.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (224, 224))
    backbone_net_model_blocks4_average_pooling0.stride = (224, 224)
    backbone_net_model_blocks4_average_pooling1 = network.add_pooling_nd(backbone_net_model_blocks4_average_pooling0.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (2, 2))
    backbone_net_model_blocks4_average_pooling1.stride = (2, 2)
    backbone_net_model_blocks4_se_reduce_conv = convBlock(network = network, weights = weights, 
                                                          inputTensor = backbone_net_model_blocks4_average_pooling1.get_output(0), 
                                                          num_output_maps = 8, ksize = 1, stride = 1, group = 1, 
                                                          lname = 'backbone_net.model._blocks.4._se_reduce.conv')
    backbone_net_model_blocks4_swish2 = addSwish(network, inputTensor = backbone_net_model_blocks4_se_reduce_conv.get_output(0))
    backbone_net_model_blocks4_se_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks4_swish2.get_output(0), num_output_maps = 192, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.4._se_expand.conv')
    backbone_net_model_blocks4_sigmoid0 = network.add_activation(input = backbone_net_model_blocks4_se_expand_conv.get_output(0), type=trt.ActivationType.SIGMOID)
    backbone_net_model_blocks4_prod0 = network.add_elementwise(backbone_net_model_blocks4_swish1.get_output(0), backbone_net_model_blocks4_sigmoid0.get_output(0), trt.ElementWiseOperation.PROD)
    backbone_net_model_blocks4_project_conv_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks4_prod0.get_output(0), num_output_maps = 32, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.4._project_conv.conv')
    backbone_net_model_blocks4_bn2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks4_project_conv_conv.get_output(0), lname = 'backbone_net.model._blocks.4._bn2')
    backbone_net_model_blocks4_bn2 = network.add_elementwise(backbone_net_model_blocks4_bn2.get_output(0), backbone_net_model_blocks3_bn2.get_output(0), trt.ElementWiseOperation.SUM)
    #-----------------------------------------------------------------------------------------------# MBConvBlock4
    #-----------------------------------------------------------------------------------------------# MBConvBlock5
    backbone_net_model_blocks5_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks4_bn2.get_output(0), 
                                                       num_output_maps = 192, ksize = 1, stride = 1, group = 1, 
                                                       lname = 'backbone_net.model._blocks.5._expand_conv.conv')
    backbone_net_model_blocks5_bn0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks5_expand_conv.get_output(0), lname = 'backbone_net.model._blocks.5._bn0')
    backbone_net_model_blocks5_swish0 = addSwish(network, inputTensor = backbone_net_model_blocks5_bn0.get_output(0))
    backbone_net_model_blocks5_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks5_swish0.get_output(0), 
                                                          num_output_maps = 192, ksize = 5, stride = 2, group = 192, 
                                                          lname = 'backbone_net.model._blocks.5._depthwise_conv.conv',pre_padding = (1, 1), post_padding = (2, 2))
    backbone_net_model_blocks5_bn1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = 
                                                     backbone_net_model_blocks5_depthwise_conv.get_output(0), lname = 'backbone_net.model._blocks.5._bn1')
    backbone_net_model_blocks5_swish1 = addSwish(network, inputTensor = backbone_net_model_blocks5_bn1.get_output(0))
    backbone_net_model_blocks5_average_pooling0 = network.add_pooling_nd(backbone_net_model_blocks5_swish1.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (224, 224))
    backbone_net_model_blocks5_average_pooling0.stride = (224, 224)
    backbone_net_model_blocks5_average_pooling1 = network.add_pooling_nd(backbone_net_model_blocks5_average_pooling0.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (2, 2))
    backbone_net_model_blocks5_average_pooling1.stride = (2, 2)
    backbone_net_model_blocks5_se_reduce_conv = convBlock(network = network, weights = weights, 
                                                          inputTensor = backbone_net_model_blocks5_average_pooling1.get_output(0), 
                                                          num_output_maps = 8, ksize = 1, stride = 1, group = 1, 
                                                          lname = 'backbone_net.model._blocks.5._se_reduce.conv')
    backbone_net_model_blocks5_swish2 = addSwish(network, inputTensor = backbone_net_model_blocks5_se_reduce_conv.get_output(0))
    backbone_net_model_blocks5_se_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks5_swish2.get_output(0), num_output_maps = 192, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.5._se_expand.conv')
    backbone_net_model_blocks5_sigmoid0 = network.add_activation(input = backbone_net_model_blocks5_se_expand_conv.get_output(0), type=trt.ActivationType.SIGMOID)
    backbone_net_model_blocks5_prod0 = network.add_elementwise(backbone_net_model_blocks5_swish1.get_output(0), backbone_net_model_blocks5_sigmoid0.get_output(0), trt.ElementWiseOperation.PROD)
    backbone_net_model_blocks5_project_conv_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks5_prod0.get_output(0), num_output_maps = 48, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.5._project_conv.conv')
    backbone_net_model_blocks5_bn2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks5_project_conv_conv.get_output(0), lname = 'backbone_net.model._blocks.5._bn2')
    #-----------------------------------------------------------------------------------------------# MBConvBlock5
    #-----------------------------------------------------------------------------------------------# MBConvBlock6
    backbone_net_model_blocks6_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks5_bn2.get_output(0), 
                                                       num_output_maps = 288, ksize = 1, stride = 1, group = 1, 
                                                       lname = 'backbone_net.model._blocks.6._expand_conv.conv')
    backbone_net_model_blocks6_bn0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks6_expand_conv.get_output(0), lname = 'backbone_net.model._blocks.6._bn0')
    backbone_net_model_blocks6_swish0 = addSwish(network, inputTensor = backbone_net_model_blocks6_bn0.get_output(0))
    backbone_net_model_blocks6_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks6_swish0.get_output(0), 
                                                          num_output_maps = 288, ksize = 5, stride = 1, group = 288, 
                                                          lname = 'backbone_net.model._blocks.6._depthwise_conv.conv',pre_padding = (2, 2), post_padding = (2, 2))
    backbone_net_model_blocks6_bn1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = 
                                                     backbone_net_model_blocks6_depthwise_conv.get_output(0), lname = 'backbone_net.model._blocks.6._bn1')
    backbone_net_model_blocks6_swish1 = addSwish(network, inputTensor = backbone_net_model_blocks6_bn1.get_output(0))
    backbone_net_model_blocks6_average_pooling0 = network.add_pooling_nd(backbone_net_model_blocks6_swish1.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (224, 224))
    backbone_net_model_blocks6_average_pooling0.stride = (224, 224)
    backbone_net_model_blocks6_average_pooling1 = network.add_pooling_nd(backbone_net_model_blocks6_average_pooling0.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (2, 2))
    backbone_net_model_blocks6_average_pooling1.stride = (2, 2)
    backbone_net_model_blocks6_se_reduce_conv = convBlock(network = network, weights = weights, 
                                                          inputTensor = backbone_net_model_blocks6_average_pooling1.get_output(0), 
                                                          num_output_maps = 12, ksize = 1, stride = 1, group = 1, 
                                                          lname = 'backbone_net.model._blocks.6._se_reduce.conv')
    backbone_net_model_blocks6_swish2 = addSwish(network, inputTensor = backbone_net_model_blocks6_se_reduce_conv.get_output(0))
    backbone_net_model_blocks6_se_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks6_swish2.get_output(0), num_output_maps = 288, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.6._se_expand.conv')
    backbone_net_model_blocks6_sigmoid0 = network.add_activation(input = backbone_net_model_blocks6_se_expand_conv.get_output(0), type=trt.ActivationType.SIGMOID)
    backbone_net_model_blocks6_prod0 = network.add_elementwise(backbone_net_model_blocks6_swish1.get_output(0), backbone_net_model_blocks6_sigmoid0.get_output(0), trt.ElementWiseOperation.PROD)
    backbone_net_model_blocks6_project_conv_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks6_prod0.get_output(0), num_output_maps = 48, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.6._project_conv.conv')
    backbone_net_model_blocks6_bn2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks6_project_conv_conv.get_output(0), lname = 'backbone_net.model._blocks.6._bn2')
    backbone_net_model_blocks6_bn2 = network.add_elementwise(backbone_net_model_blocks6_bn2.get_output(0), backbone_net_model_blocks5_bn2.get_output(0), trt.ElementWiseOperation.SUM)
    #-----------------------------------------------------------------------------------------------# MBConvBlock6
    #-----------------------------------------------------------------------------------------------# MBConvBlock7
    backbone_net_model_blocks7_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks6_bn2.get_output(0), 
                                                       num_output_maps = 288, ksize = 1, stride = 1, group = 1, 
                                                       lname = 'backbone_net.model._blocks.7._expand_conv.conv')
    backbone_net_model_blocks7_bn0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks7_expand_conv.get_output(0), lname = 'backbone_net.model._blocks.7._bn0')
    backbone_net_model_blocks7_swish0 = addSwish(network, inputTensor = backbone_net_model_blocks7_bn0.get_output(0))
    backbone_net_model_blocks7_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks7_swish0.get_output(0), 
                                                          num_output_maps = 288, ksize = 5, stride = 1, group = 288, 
                                                          lname = 'backbone_net.model._blocks.7._depthwise_conv.conv',pre_padding = (2, 2), post_padding = (2, 2))
    backbone_net_model_blocks7_bn1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = 
                                                     backbone_net_model_blocks7_depthwise_conv.get_output(0), lname = 'backbone_net.model._blocks.7._bn1')
    backbone_net_model_blocks7_swish1 = addSwish(network, inputTensor = backbone_net_model_blocks7_bn1.get_output(0))
    backbone_net_model_blocks7_average_pooling0 = network.add_pooling_nd(backbone_net_model_blocks7_swish1.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (224, 224))
    backbone_net_model_blocks7_average_pooling0.stride = (224, 224)
    backbone_net_model_blocks7_average_pooling1 = network.add_pooling_nd(backbone_net_model_blocks7_average_pooling0.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (2, 2))
    backbone_net_model_blocks7_average_pooling1.stride = (2, 2)
    backbone_net_model_blocks7_se_reduce_conv = convBlock(network = network, weights = weights, 
                                                          inputTensor = backbone_net_model_blocks7_average_pooling1.get_output(0), 
                                                          num_output_maps = 12, ksize = 1, stride = 1, group = 1, 
                                                          lname = 'backbone_net.model._blocks.7._se_reduce.conv')
    backbone_net_model_blocks7_swish2 = addSwish(network, inputTensor = backbone_net_model_blocks7_se_reduce_conv.get_output(0))
    backbone_net_model_blocks7_se_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks7_swish2.get_output(0), num_output_maps = 288, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.7._se_expand.conv')
    backbone_net_model_blocks7_sigmoid0 = network.add_activation(input = backbone_net_model_blocks7_se_expand_conv.get_output(0), type=trt.ActivationType.SIGMOID)
    backbone_net_model_blocks7_prod0 = network.add_elementwise(backbone_net_model_blocks7_swish1.get_output(0), backbone_net_model_blocks7_sigmoid0.get_output(0), trt.ElementWiseOperation.PROD)
    backbone_net_model_blocks7_project_conv_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks7_prod0.get_output(0), num_output_maps = 48, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.7._project_conv.conv')
    backbone_net_model_blocks7_bn2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks7_project_conv_conv.get_output(0), lname = 'backbone_net.model._blocks.7._bn2')
    backbone_net_model_blocks7_bn2 = network.add_elementwise(backbone_net_model_blocks7_bn2.get_output(0), backbone_net_model_blocks6_bn2.get_output(0), trt.ElementWiseOperation.SUM)
    #-----------------------------------------------------------------------------------------------# MBConvBlock7
    #-----------------------------------------------------------------------------------------------# MBConvBlock8
    backbone_net_model_blocks8_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks7_bn2.get_output(0), 
                                                       num_output_maps = 288, ksize = 1, stride = 1, group = 1, 
                                                       lname = 'backbone_net.model._blocks.8._expand_conv.conv')
    backbone_net_model_blocks8_bn0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks8_expand_conv.get_output(0), lname = 'backbone_net.model._blocks.8._bn0')
    backbone_net_model_blocks8_swish0 = addSwish(network, inputTensor = backbone_net_model_blocks8_bn0.get_output(0))
    backbone_net_model_blocks8_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks8_swish0.get_output(0), 
                                                          num_output_maps = 288, ksize = 3, stride = 2, group = 288, 
                                                          lname = 'backbone_net.model._blocks.8._depthwise_conv.conv',pre_padding = (0, 0), post_padding = (1, 1))
    backbone_net_model_blocks8_bn1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = 
                                                     backbone_net_model_blocks8_depthwise_conv.get_output(0), lname = 'backbone_net.model._blocks.8._bn1')
    backbone_net_model_blocks8_swish1 = addSwish(network, inputTensor = backbone_net_model_blocks8_bn1.get_output(0))
    backbone_net_model_blocks8_average_pooling0 = network.add_pooling_nd(backbone_net_model_blocks8_swish1.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (224, 224))
    backbone_net_model_blocks8_average_pooling0.stride = (224, 224)
    backbone_net_model_blocks8_average_pooling1 = network.add_pooling_nd(backbone_net_model_blocks8_average_pooling0.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (2, 2))
    backbone_net_model_blocks8_average_pooling1.stride = (2, 2)
    backbone_net_model_blocks8_se_reduce_conv = convBlock(network = network, weights = weights, 
                                                          inputTensor = backbone_net_model_blocks8_average_pooling1.get_output(0), 
                                                          num_output_maps = 12, ksize = 1, stride = 1, group = 1, 
                                                          lname = 'backbone_net.model._blocks.8._se_reduce.conv')
    backbone_net_model_blocks8_swish2 = addSwish(network, inputTensor = backbone_net_model_blocks8_se_reduce_conv.get_output(0))
    backbone_net_model_blocks8_se_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks8_swish2.get_output(0), num_output_maps = 288, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.8._se_expand.conv')
    backbone_net_model_blocks8_sigmoid0 = network.add_activation(input = backbone_net_model_blocks8_se_expand_conv.get_output(0), type=trt.ActivationType.SIGMOID)
    backbone_net_model_blocks8_prod0 = network.add_elementwise(backbone_net_model_blocks8_swish1.get_output(0), backbone_net_model_blocks8_sigmoid0.get_output(0), trt.ElementWiseOperation.PROD)
    backbone_net_model_blocks8_project_conv_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks8_prod0.get_output(0), num_output_maps = 96, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.8._project_conv.conv')
    backbone_net_model_blocks8_bn2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks8_project_conv_conv.get_output(0), lname = 'backbone_net.model._blocks.8._bn2')
    #-----------------------------------------------------------------------------------------------# MBConvBlock8
    #-----------------------------------------------------------------------------------------------# MBConvBlock9
    backbone_net_model_blocks9_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks8_bn2.get_output(0), 
                                                       num_output_maps = 576, ksize = 1, stride = 1, group = 1, 
                                                       lname = 'backbone_net.model._blocks.9._expand_conv.conv')
    backbone_net_model_blocks9_bn0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks9_expand_conv.get_output(0), lname = 'backbone_net.model._blocks.9._bn0')
    backbone_net_model_blocks9_swish0 = addSwish(network, inputTensor = backbone_net_model_blocks9_bn0.get_output(0))
    backbone_net_model_blocks9_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks9_swish0.get_output(0), 
                                                          num_output_maps = 576, ksize = 3, stride = 1, group = 576, 
                                                          lname = 'backbone_net.model._blocks.9._depthwise_conv.conv',pre_padding = (1, 1), post_padding = (1, 1))
    backbone_net_model_blocks9_bn1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = 
                                                     backbone_net_model_blocks9_depthwise_conv.get_output(0), lname = 'backbone_net.model._blocks.9._bn1')
    backbone_net_model_blocks9_swish1 = addSwish(network, inputTensor = backbone_net_model_blocks9_bn1.get_output(0))
    backbone_net_model_blocks9_average_pooling0 = network.add_pooling_nd(backbone_net_model_blocks9_swish1.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (224, 224))
    backbone_net_model_blocks9_average_pooling0.stride = (224, 224)
    backbone_net_model_blocks9_average_pooling1 = network.add_pooling_nd(backbone_net_model_blocks9_average_pooling0.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (2, 2))
    backbone_net_model_blocks9_average_pooling1.stride = (2, 2)
    backbone_net_model_blocks9_se_reduce_conv = convBlock(network = network, weights = weights, 
                                                          inputTensor = backbone_net_model_blocks9_average_pooling1.get_output(0), 
                                                          num_output_maps = 24, ksize = 1, stride = 1, group = 1, 
                                                          lname = 'backbone_net.model._blocks.9._se_reduce.conv')
    backbone_net_model_blocks9_swish2 = addSwish(network, inputTensor = backbone_net_model_blocks9_se_reduce_conv.get_output(0))
    backbone_net_model_blocks9_se_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks9_swish2.get_output(0), num_output_maps = 576, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.9._se_expand.conv')
    backbone_net_model_blocks9_sigmoid0 = network.add_activation(input = backbone_net_model_blocks9_se_expand_conv.get_output(0), type=trt.ActivationType.SIGMOID)
    backbone_net_model_blocks9_prod0 = network.add_elementwise(backbone_net_model_blocks9_swish1.get_output(0), backbone_net_model_blocks9_sigmoid0.get_output(0), trt.ElementWiseOperation.PROD)
    backbone_net_model_blocks9_project_conv_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks9_prod0.get_output(0), num_output_maps = 96, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.9._project_conv.conv')
    backbone_net_model_blocks9_bn2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks9_project_conv_conv.get_output(0), lname = 'backbone_net.model._blocks.9._bn2')
    backbone_net_model_blocks9_bn2 = network.add_elementwise(backbone_net_model_blocks9_bn2.get_output(0), backbone_net_model_blocks8_bn2.get_output(0), trt.ElementWiseOperation.SUM)
    #-----------------------------------------------------------------------------------------------# MBConvBlock9
    #-----------------------------------------------------------------------------------------------# MBconvBlock0
    backbone_net_model_blocks10_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks9_bn2.get_output(0), 
                                                       num_output_maps = 576, ksize = 1, stride = 1, group = 1, 
                                                       lname = 'backbone_net.model._blocks.10._expand_conv.conv')
    backbone_net_model_blocks10_bn0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks10_expand_conv.get_output(0), lname = 'backbone_net.model._blocks.10._bn0')
    backbone_net_model_blocks10_swish0 = addSwish(network, inputTensor = backbone_net_model_blocks10_bn0.get_output(0))
    backbone_net_model_blocks10_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks10_swish0.get_output(0), 
                                                          num_output_maps = 576, ksize = 3, stride = 1, group = 576, 
                                                          lname = 'backbone_net.model._blocks.10._depthwise_conv.conv',pre_padding = (1, 1), post_padding = (1, 1))
    backbone_net_model_blocks10_bn1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = 
                                                     backbone_net_model_blocks10_depthwise_conv.get_output(0), lname = 'backbone_net.model._blocks.10._bn1')
    backbone_net_model_blocks10_swish1 = addSwish(network, inputTensor = backbone_net_model_blocks10_bn1.get_output(0))
    backbone_net_model_blocks10_average_pooling0 = network.add_pooling_nd(backbone_net_model_blocks10_swish1.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (224, 224))
    backbone_net_model_blocks10_average_pooling0.stride = (224, 224)
    backbone_net_model_blocks10_average_pooling1 = network.add_pooling_nd(backbone_net_model_blocks10_average_pooling0.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (2, 2))
    backbone_net_model_blocks10_average_pooling1.stride = (2, 2)
    backbone_net_model_blocks10_se_reduce_conv = convBlock(network = network, weights = weights, 
                                                          inputTensor = backbone_net_model_blocks10_average_pooling1.get_output(0), 
                                                          num_output_maps = 24, ksize = 1, stride = 1, group = 1, 
                                                          lname = 'backbone_net.model._blocks.10._se_reduce.conv')
    backbone_net_model_blocks10_swish2 = addSwish(network, inputTensor = backbone_net_model_blocks10_se_reduce_conv.get_output(0))
    backbone_net_model_blocks10_se_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks10_swish2.get_output(0), num_output_maps = 576, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.10._se_expand.conv')
    backbone_net_model_blocks10_sigmoid0 = network.add_activation(input = backbone_net_model_blocks10_se_expand_conv.get_output(0), type=trt.ActivationType.SIGMOID)
    backbone_net_model_blocks10_prod0 = network.add_elementwise(backbone_net_model_blocks10_swish1.get_output(0), backbone_net_model_blocks10_sigmoid0.get_output(0), trt.ElementWiseOperation.PROD)
    backbone_net_model_blocks10_project_conv_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks10_prod0.get_output(0), num_output_maps = 96, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.10._project_conv.conv')
    backbone_net_model_blocks10_bn2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks10_project_conv_conv.get_output(0), lname = 'backbone_net.model._blocks.10._bn2')
    backbone_net_model_blocks10_bn2 = network.add_elementwise(backbone_net_model_blocks10_bn2.get_output(0), backbone_net_model_blocks9_bn2.get_output(0), trt.ElementWiseOperation.SUM)
    #-----------------------------------------------------------------------------------------------# MBconvBlock0
    #-----------------------------------------------------------------------------------------------# MBconvBlock1
    backbone_net_model_blocks11_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks10_bn2.get_output(0), 
                                                       num_output_maps = 576, ksize = 1, stride = 1, group = 1, 
                                                       lname = 'backbone_net.model._blocks.11._expand_conv.conv')
    backbone_net_model_blocks11_bn0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks11_expand_conv.get_output(0), lname = 'backbone_net.model._blocks.11._bn0')
    backbone_net_model_blocks11_swish0 = addSwish(network, inputTensor = backbone_net_model_blocks11_bn0.get_output(0))
    backbone_net_model_blocks11_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks11_swish0.get_output(0), 
                                                          num_output_maps = 576, ksize = 3, stride = 1, group = 576, 
                                                          lname = 'backbone_net.model._blocks.11._depthwise_conv.conv',pre_padding = (1, 1), post_padding = (1, 1))
    backbone_net_model_blocks11_bn1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = 
                                                     backbone_net_model_blocks11_depthwise_conv.get_output(0), lname = 'backbone_net.model._blocks.11._bn1')
    backbone_net_model_blocks11_swish1 = addSwish(network, inputTensor = backbone_net_model_blocks11_bn1.get_output(0))
    backbone_net_model_blocks11_average_pooling0 = network.add_pooling_nd(backbone_net_model_blocks11_swish1.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (224, 224))
    backbone_net_model_blocks11_average_pooling0.stride = (224, 224)
    backbone_net_model_blocks11_average_pooling1 = network.add_pooling_nd(backbone_net_model_blocks11_average_pooling0.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (2, 2))
    backbone_net_model_blocks11_average_pooling1.stride = (2, 2)
    backbone_net_model_blocks11_se_reduce_conv = convBlock(network = network, weights = weights, 
                                                          inputTensor = backbone_net_model_blocks11_average_pooling1.get_output(0), 
                                                          num_output_maps = 24, ksize = 1, stride = 1, group = 1, 
                                                          lname = 'backbone_net.model._blocks.11._se_reduce.conv')
    backbone_net_model_blocks11_swish2 = addSwish(network, inputTensor = backbone_net_model_blocks11_se_reduce_conv.get_output(0))
    backbone_net_model_blocks11_se_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks11_swish2.get_output(0), num_output_maps = 576, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.11._se_expand.conv')
    backbone_net_model_blocks11_sigmoid0 = network.add_activation(input = backbone_net_model_blocks11_se_expand_conv.get_output(0), type=trt.ActivationType.SIGMOID)
    backbone_net_model_blocks11_prod0 = network.add_elementwise(backbone_net_model_blocks11_swish1.get_output(0), backbone_net_model_blocks11_sigmoid0.get_output(0), trt.ElementWiseOperation.PROD)
    backbone_net_model_blocks11_project_conv_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks11_prod0.get_output(0), num_output_maps = 96, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.11._project_conv.conv')
    backbone_net_model_blocks11_bn2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks11_project_conv_conv.get_output(0), lname = 'backbone_net.model._blocks.11._bn2')
    backbone_net_model_blocks11_bn2 = network.add_elementwise(backbone_net_model_blocks11_bn2.get_output(0), backbone_net_model_blocks10_bn2.get_output(0), trt.ElementWiseOperation.SUM)
    #-----------------------------------------------------------------------------------------------# MBconvBlock1
    #-----------------------------------------------------------------------------------------------# MBconvBlock2
    backbone_net_model_blocks12_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks11_bn2.get_output(0), 
                                                       num_output_maps = 576, ksize = 1, stride = 1, group = 1, 
                                                       lname = 'backbone_net.model._blocks.12._expand_conv.conv')
    backbone_net_model_blocks12_bn0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks12_expand_conv.get_output(0), lname = 'backbone_net.model._blocks.12._bn0')
    backbone_net_model_blocks12_swish0 = addSwish(network, inputTensor = backbone_net_model_blocks12_bn0.get_output(0))
    backbone_net_model_blocks12_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks12_swish0.get_output(0), 
                                                          num_output_maps = 576, ksize = 3, stride = 1, group = 576, 
                                                          lname = 'backbone_net.model._blocks.12._depthwise_conv.conv',pre_padding = (1, 1), post_padding = (1, 1))
    backbone_net_model_blocks12_bn1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = 
                                                     backbone_net_model_blocks12_depthwise_conv.get_output(0), lname = 'backbone_net.model._blocks.12._bn1')
    backbone_net_model_blocks12_swish1 = addSwish(network, inputTensor = backbone_net_model_blocks12_bn1.get_output(0))
    backbone_net_model_blocks12_average_pooling0 = network.add_pooling_nd(backbone_net_model_blocks12_swish1.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (224, 224))
    backbone_net_model_blocks12_average_pooling0.stride = (224, 224)
    backbone_net_model_blocks12_average_pooling1 = network.add_pooling_nd(backbone_net_model_blocks12_average_pooling0.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (2, 2))
    backbone_net_model_blocks12_average_pooling1.stride = (2, 2)
    backbone_net_model_blocks12_se_reduce_conv = convBlock(network = network, weights = weights, 
                                                          inputTensor = backbone_net_model_blocks12_average_pooling1.get_output(0), 
                                                          num_output_maps = 24, ksize = 1, stride = 1, group = 1, 
                                                          lname = 'backbone_net.model._blocks.12._se_reduce.conv')
    backbone_net_model_blocks12_swish2 = addSwish(network, inputTensor = backbone_net_model_blocks12_se_reduce_conv.get_output(0))
    backbone_net_model_blocks12_se_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks12_swish2.get_output(0), num_output_maps = 576, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.12._se_expand.conv')
    backbone_net_model_blocks12_sigmoid0 = network.add_activation(input = backbone_net_model_blocks12_se_expand_conv.get_output(0), type=trt.ActivationType.SIGMOID)
    backbone_net_model_blocks12_prod0 = network.add_elementwise(backbone_net_model_blocks12_swish1.get_output(0), backbone_net_model_blocks12_sigmoid0.get_output(0), trt.ElementWiseOperation.PROD)
    backbone_net_model_blocks12_project_conv_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks12_prod0.get_output(0), num_output_maps = 96, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.12._project_conv.conv')
    backbone_net_model_blocks12_bn2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks12_project_conv_conv.get_output(0), lname = 'backbone_net.model._blocks.12._bn2')
    backbone_net_model_blocks12_bn2 = network.add_elementwise(backbone_net_model_blocks12_bn2.get_output(0), backbone_net_model_blocks11_bn2.get_output(0), trt.ElementWiseOperation.SUM)
    #-----------------------------------------------------------------------------------------------# MBconvBlock2
    #-----------------------------------------------------------------------------------------------# MBconvBlock3
    backbone_net_model_blocks13_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks12_bn2.get_output(0), 
                                                       num_output_maps = 576, ksize = 1, stride = 1, group = 1, 
                                                       lname = 'backbone_net.model._blocks.13._expand_conv.conv')
    backbone_net_model_blocks13_bn0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks13_expand_conv.get_output(0), lname = 'backbone_net.model._blocks.13._bn0')
    backbone_net_model_blocks13_swish0 = addSwish(network, inputTensor = backbone_net_model_blocks13_bn0.get_output(0))
    backbone_net_model_blocks13_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks13_swish0.get_output(0), 
                                                          num_output_maps = 576, ksize = 5, stride = 1, group = 576, 
                                                          lname = 'backbone_net.model._blocks.13._depthwise_conv.conv',pre_padding = (2, 2), post_padding = (2, 2))
    backbone_net_model_blocks13_bn1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = 
                                                     backbone_net_model_blocks13_depthwise_conv.get_output(0), lname = 'backbone_net.model._blocks.13._bn1')
    backbone_net_model_blocks13_swish1 = addSwish(network, inputTensor = backbone_net_model_blocks13_bn1.get_output(0))
    backbone_net_model_blocks13_average_pooling0 = network.add_pooling_nd(backbone_net_model_blocks13_swish1.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (224, 224))
    backbone_net_model_blocks13_average_pooling0.stride = (224, 224)
    backbone_net_model_blocks13_average_pooling1 = network.add_pooling_nd(backbone_net_model_blocks13_average_pooling0.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (2, 2))
    backbone_net_model_blocks13_average_pooling1.stride = (2, 2)
    backbone_net_model_blocks13_se_reduce_conv = convBlock(network = network, weights = weights, 
                                                          inputTensor = backbone_net_model_blocks13_average_pooling1.get_output(0), 
                                                          num_output_maps = 24, ksize = 1, stride = 1, group = 1, 
                                                          lname = 'backbone_net.model._blocks.13._se_reduce.conv')
    backbone_net_model_blocks13_swish2 = addSwish(network, inputTensor = backbone_net_model_blocks13_se_reduce_conv.get_output(0))
    backbone_net_model_blocks13_se_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks13_swish2.get_output(0), num_output_maps = 576, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.13._se_expand.conv')
    backbone_net_model_blocks13_sigmoid0 = network.add_activation(input = backbone_net_model_blocks13_se_expand_conv.get_output(0), type=trt.ActivationType.SIGMOID)
    backbone_net_model_blocks13_prod0 = network.add_elementwise(backbone_net_model_blocks13_swish1.get_output(0), backbone_net_model_blocks13_sigmoid0.get_output(0), trt.ElementWiseOperation.PROD)
    backbone_net_model_blocks13_project_conv_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks13_prod0.get_output(0), num_output_maps = 136, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.13._project_conv.conv')
    backbone_net_model_blocks13_bn2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks13_project_conv_conv.get_output(0), lname = 'backbone_net.model._blocks.13._bn2')
    #backbone_net_model_blocks13_bn2 = network.add_elementwise(backbone_net_model_blocks13_bn2.get_output(0), backbone_net_model_blocks12_bn2.get_output(0), trt.ElementWiseOperation.SUM)
    #-----------------------------------------------------------------------------------------------# MBconvBlock3
    #-----------------------------------------------------------------------------------------------# MBconvBlock4
    backbone_net_model_blocks14_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks13_bn2.get_output(0), 
                                                       num_output_maps = 816, ksize = 1, stride = 1, group = 1, 
                                                       lname = 'backbone_net.model._blocks.14._expand_conv.conv')
    backbone_net_model_blocks14_bn0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks14_expand_conv.get_output(0), lname = 'backbone_net.model._blocks.14._bn0')
    backbone_net_model_blocks14_swish0 = addSwish(network, inputTensor = backbone_net_model_blocks14_bn0.get_output(0))
    backbone_net_model_blocks14_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks14_swish0.get_output(0), 
                                                          num_output_maps = 816, ksize = 5, stride = 1, group = 816, 
                                                          lname = 'backbone_net.model._blocks.14._depthwise_conv.conv',pre_padding = (2, 2), post_padding = (2, 2))
    backbone_net_model_blocks14_bn1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = 
                                                     backbone_net_model_blocks14_depthwise_conv.get_output(0), lname = 'backbone_net.model._blocks.14._bn1')
    backbone_net_model_blocks14_swish1 = addSwish(network, inputTensor = backbone_net_model_blocks14_bn1.get_output(0))
    backbone_net_model_blocks14_average_pooling0 = network.add_pooling_nd(backbone_net_model_blocks14_swish1.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (224, 224))
    backbone_net_model_blocks14_average_pooling0.stride = (224, 224)
    backbone_net_model_blocks14_average_pooling1 = network.add_pooling_nd(backbone_net_model_blocks14_average_pooling0.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (2, 2))
    backbone_net_model_blocks14_average_pooling1.stride = (2, 2)
    backbone_net_model_blocks14_se_reduce_conv = convBlock(network = network, weights = weights, 
                                                          inputTensor = backbone_net_model_blocks14_average_pooling1.get_output(0), 
                                                          num_output_maps = 34, ksize = 1, stride = 1, group = 1, 
                                                          lname = 'backbone_net.model._blocks.14._se_reduce.conv')
    backbone_net_model_blocks14_swish2 = addSwish(network, inputTensor = backbone_net_model_blocks14_se_reduce_conv.get_output(0))
    backbone_net_model_blocks14_se_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks14_swish2.get_output(0), num_output_maps = 816, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.14._se_expand.conv')
    backbone_net_model_blocks14_sigmoid0 = network.add_activation(input = backbone_net_model_blocks14_se_expand_conv.get_output(0), type=trt.ActivationType.SIGMOID)
    backbone_net_model_blocks14_prod0 = network.add_elementwise(backbone_net_model_blocks14_swish1.get_output(0), backbone_net_model_blocks14_sigmoid0.get_output(0), trt.ElementWiseOperation.PROD)
    backbone_net_model_blocks14_project_conv_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks14_prod0.get_output(0), num_output_maps = 136, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.14._project_conv.conv')
    backbone_net_model_blocks14_bn2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks14_project_conv_conv.get_output(0), lname = 'backbone_net.model._blocks.14._bn2')
    backbone_net_model_blocks14_bn2 = network.add_elementwise(backbone_net_model_blocks14_bn2.get_output(0), backbone_net_model_blocks13_bn2.get_output(0), trt.ElementWiseOperation.SUM)
    #-----------------------------------------------------------------------------------------------# MBconvBlock4
    #-----------------------------------------------------------------------------------------------# MBconvBlock5
    backbone_net_model_blocks15_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks14_bn2.get_output(0), 
                                                       num_output_maps = 816, ksize = 1, stride = 1, group = 1, 
                                                       lname = 'backbone_net.model._blocks.15._expand_conv.conv')
    backbone_net_model_blocks15_bn0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks15_expand_conv.get_output(0), lname = 'backbone_net.model._blocks.15._bn0')
    backbone_net_model_blocks15_swish0 = addSwish(network, inputTensor = backbone_net_model_blocks15_bn0.get_output(0))
    backbone_net_model_blocks15_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks15_swish0.get_output(0), 
                                                          num_output_maps = 816, ksize = 5, stride = 1, group = 816, 
                                                          lname = 'backbone_net.model._blocks.15._depthwise_conv.conv',pre_padding = (2, 2), post_padding = (2, 2))
    backbone_net_model_blocks15_bn1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = 
                                                     backbone_net_model_blocks15_depthwise_conv.get_output(0), lname = 'backbone_net.model._blocks.15._bn1')
    backbone_net_model_blocks15_swish1 = addSwish(network, inputTensor = backbone_net_model_blocks15_bn1.get_output(0))
    backbone_net_model_blocks15_average_pooling0 = network.add_pooling_nd(backbone_net_model_blocks15_swish1.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (224, 224))
    backbone_net_model_blocks15_average_pooling0.stride = (224, 224)
    backbone_net_model_blocks15_average_pooling1 = network.add_pooling_nd(backbone_net_model_blocks15_average_pooling0.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (2, 2))
    backbone_net_model_blocks15_average_pooling1.stride = (2, 2)
    backbone_net_model_blocks15_se_reduce_conv = convBlock(network = network, weights = weights, 
                                                          inputTensor = backbone_net_model_blocks15_average_pooling1.get_output(0), 
                                                          num_output_maps = 34, ksize = 1, stride = 1, group = 1, 
                                                          lname = 'backbone_net.model._blocks.15._se_reduce.conv')
    backbone_net_model_blocks15_swish2 = addSwish(network, inputTensor = backbone_net_model_blocks15_se_reduce_conv.get_output(0))
    backbone_net_model_blocks15_se_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks15_swish2.get_output(0), num_output_maps = 816, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.15._se_expand.conv')
    backbone_net_model_blocks15_sigmoid0 = network.add_activation(input = backbone_net_model_blocks15_se_expand_conv.get_output(0), type=trt.ActivationType.SIGMOID)
    backbone_net_model_blocks15_prod0 = network.add_elementwise(backbone_net_model_blocks15_swish1.get_output(0), backbone_net_model_blocks15_sigmoid0.get_output(0), trt.ElementWiseOperation.PROD)
    backbone_net_model_blocks15_project_conv_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks15_prod0.get_output(0), num_output_maps = 136, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.15._project_conv.conv')
    backbone_net_model_blocks15_bn2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks15_project_conv_conv.get_output(0), lname = 'backbone_net.model._blocks.15._bn2')
    backbone_net_model_blocks15_bn2 = network.add_elementwise(backbone_net_model_blocks15_bn2.get_output(0), backbone_net_model_blocks14_bn2.get_output(0), trt.ElementWiseOperation.SUM)
    #-----------------------------------------------------------------------------------------------# MBconvBlock5
    #-----------------------------------------------------------------------------------------------# MBconvBlock6
    backbone_net_model_blocks16_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks15_bn2.get_output(0), 
                                                       num_output_maps = 816, ksize = 1, stride = 1, group = 1, 
                                                       lname = 'backbone_net.model._blocks.16._expand_conv.conv')
    backbone_net_model_blocks16_bn0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks16_expand_conv.get_output(0), lname = 'backbone_net.model._blocks.16._bn0')
    backbone_net_model_blocks16_swish0 = addSwish(network, inputTensor = backbone_net_model_blocks16_bn0.get_output(0))
    backbone_net_model_blocks16_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks16_swish0.get_output(0), 
                                                          num_output_maps = 816, ksize = 5, stride = 1, group = 816, 
                                                          lname = 'backbone_net.model._blocks.16._depthwise_conv.conv',pre_padding = (2, 2), post_padding = (2, 2))
    backbone_net_model_blocks16_bn1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = 
                                                     backbone_net_model_blocks16_depthwise_conv.get_output(0), lname = 'backbone_net.model._blocks.16._bn1')
    backbone_net_model_blocks16_swish1 = addSwish(network, inputTensor = backbone_net_model_blocks16_bn1.get_output(0))
    backbone_net_model_blocks16_average_pooling0 = network.add_pooling_nd(backbone_net_model_blocks16_swish1.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (224, 224))
    backbone_net_model_blocks16_average_pooling0.stride = (224, 224)
    backbone_net_model_blocks16_average_pooling1 = network.add_pooling_nd(backbone_net_model_blocks16_average_pooling0.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (2, 2))
    backbone_net_model_blocks16_average_pooling1.stride = (2, 2)
    backbone_net_model_blocks16_se_reduce_conv = convBlock(network = network, weights = weights, 
                                                          inputTensor = backbone_net_model_blocks16_average_pooling1.get_output(0), 
                                                          num_output_maps = 34, ksize = 1, stride = 1, group = 1, 
                                                          lname = 'backbone_net.model._blocks.16._se_reduce.conv')
    backbone_net_model_blocks16_swish2 = addSwish(network, inputTensor = backbone_net_model_blocks16_se_reduce_conv.get_output(0))
    backbone_net_model_blocks16_se_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks16_swish2.get_output(0), num_output_maps = 816, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.16._se_expand.conv')
    backbone_net_model_blocks16_sigmoid0 = network.add_activation(input = backbone_net_model_blocks16_se_expand_conv.get_output(0), type=trt.ActivationType.SIGMOID)
    backbone_net_model_blocks16_prod0 = network.add_elementwise(backbone_net_model_blocks16_swish1.get_output(0), backbone_net_model_blocks16_sigmoid0.get_output(0), trt.ElementWiseOperation.PROD)
    backbone_net_model_blocks16_project_conv_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks16_prod0.get_output(0), num_output_maps = 136, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.16._project_conv.conv')
    backbone_net_model_blocks16_bn2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks16_project_conv_conv.get_output(0), lname = 'backbone_net.model._blocks.16._bn2')
    backbone_net_model_blocks16_bn2 = network.add_elementwise(backbone_net_model_blocks16_bn2.get_output(0), backbone_net_model_blocks15_bn2.get_output(0), trt.ElementWiseOperation.SUM)
    #-----------------------------------------------------------------------------------------------# MBconvBlock6
    #-----------------------------------------------------------------------------------------------# MBconvBlock7
    backbone_net_model_blocks17_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks16_bn2.get_output(0), 
                                                       num_output_maps = 816, ksize = 1, stride = 1, group = 1, 
                                                       lname = 'backbone_net.model._blocks.17._expand_conv.conv')
    backbone_net_model_blocks17_bn0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks17_expand_conv.get_output(0), lname = 'backbone_net.model._blocks.17._bn0')
    backbone_net_model_blocks17_swish0 = addSwish(network, inputTensor = backbone_net_model_blocks17_bn0.get_output(0))
    backbone_net_model_blocks17_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks17_swish0.get_output(0), 
                                                          num_output_maps = 816, ksize = 5, stride = 1, group = 816, 
                                                          lname = 'backbone_net.model._blocks.17._depthwise_conv.conv',pre_padding = (2, 2), post_padding = (2, 2))
    backbone_net_model_blocks17_bn1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = 
                                                     backbone_net_model_blocks17_depthwise_conv.get_output(0), lname = 'backbone_net.model._blocks.17._bn1')
    backbone_net_model_blocks17_swish1 = addSwish(network, inputTensor = backbone_net_model_blocks17_bn1.get_output(0))
    backbone_net_model_blocks17_average_pooling0 = network.add_pooling_nd(backbone_net_model_blocks17_swish1.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (224, 224))
    backbone_net_model_blocks17_average_pooling0.stride = (224, 224)
    backbone_net_model_blocks17_average_pooling1 = network.add_pooling_nd(backbone_net_model_blocks17_average_pooling0.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (2, 2))
    backbone_net_model_blocks17_average_pooling1.stride = (2, 2)
    backbone_net_model_blocks17_se_reduce_conv = convBlock(network = network, weights = weights, 
                                                          inputTensor = backbone_net_model_blocks17_average_pooling1.get_output(0), 
                                                          num_output_maps = 34, ksize = 1, stride = 1, group = 1, 
                                                          lname = 'backbone_net.model._blocks.17._se_reduce.conv')
    backbone_net_model_blocks17_swish2 = addSwish(network, inputTensor = backbone_net_model_blocks17_se_reduce_conv.get_output(0))
    backbone_net_model_blocks17_se_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks17_swish2.get_output(0), num_output_maps = 816, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.17._se_expand.conv')
    backbone_net_model_blocks17_sigmoid0 = network.add_activation(input = backbone_net_model_blocks17_se_expand_conv.get_output(0), type=trt.ActivationType.SIGMOID)
    backbone_net_model_blocks17_prod0 = network.add_elementwise(backbone_net_model_blocks17_swish1.get_output(0), backbone_net_model_blocks17_sigmoid0.get_output(0), trt.ElementWiseOperation.PROD)
    backbone_net_model_blocks17_project_conv_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks17_prod0.get_output(0), num_output_maps = 136, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.17._project_conv.conv')
    backbone_net_model_blocks17_bn2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks17_project_conv_conv.get_output(0), lname = 'backbone_net.model._blocks.17._bn2')
    backbone_net_model_blocks17_bn2 = network.add_elementwise(backbone_net_model_blocks17_bn2.get_output(0), backbone_net_model_blocks16_bn2.get_output(0), trt.ElementWiseOperation.SUM)
    #-----------------------------------------------------------------------------------------------# MBconvBlock7
    #-----------------------------------------------------------------------------------------------# MBconvBlock8
    backbone_net_model_blocks18_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks17_bn2.get_output(0), 
                                                       num_output_maps = 816, ksize = 1, stride = 1, group = 1, 
                                                       lname = 'backbone_net.model._blocks.18._expand_conv.conv')
    backbone_net_model_blocks18_bn0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks18_expand_conv.get_output(0), lname = 'backbone_net.model._blocks.18._bn0')
    backbone_net_model_blocks18_swish0 = addSwish(network, inputTensor = backbone_net_model_blocks18_bn0.get_output(0))
    backbone_net_model_blocks18_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks18_swish0.get_output(0), 
                                                          num_output_maps = 816, ksize = 5, stride = 2, group = 816, 
                                                          lname = 'backbone_net.model._blocks.18._depthwise_conv.conv',pre_padding = (1, 1), post_padding = (2, 2))
    backbone_net_model_blocks18_bn1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = 
                                                     backbone_net_model_blocks18_depthwise_conv.get_output(0), lname = 'backbone_net.model._blocks.18._bn1')
    backbone_net_model_blocks18_swish1 = addSwish(network, inputTensor = backbone_net_model_blocks18_bn1.get_output(0))
    backbone_net_model_blocks18_average_pooling0 = network.add_pooling_nd(backbone_net_model_blocks18_swish1.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (224, 224))
    backbone_net_model_blocks18_average_pooling0.stride = (224, 224)
    backbone_net_model_blocks18_average_pooling1 = network.add_pooling_nd(backbone_net_model_blocks18_average_pooling0.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (2, 2))
    backbone_net_model_blocks18_average_pooling1.stride = (2, 2)
    backbone_net_model_blocks18_se_reduce_conv = convBlock(network = network, weights = weights, 
                                                          inputTensor = backbone_net_model_blocks18_average_pooling1.get_output(0), 
                                                          num_output_maps = 34, ksize = 1, stride = 1, group = 1, 
                                                          lname = 'backbone_net.model._blocks.18._se_reduce.conv')
    backbone_net_model_blocks18_swish2 = addSwish(network, inputTensor = backbone_net_model_blocks18_se_reduce_conv.get_output(0))
    backbone_net_model_blocks18_se_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks18_swish2.get_output(0), num_output_maps = 816, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.18._se_expand.conv')
    backbone_net_model_blocks18_sigmoid0 = network.add_activation(input = backbone_net_model_blocks18_se_expand_conv.get_output(0), type=trt.ActivationType.SIGMOID)
    backbone_net_model_blocks18_prod0 = network.add_elementwise(backbone_net_model_blocks18_swish1.get_output(0), backbone_net_model_blocks18_sigmoid0.get_output(0), trt.ElementWiseOperation.PROD)
    backbone_net_model_blocks18_project_conv_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks18_prod0.get_output(0), num_output_maps = 232, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.18._project_conv.conv')
    backbone_net_model_blocks18_bn2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks18_project_conv_conv.get_output(0), lname = 'backbone_net.model._blocks.18._bn2')
    #backbone_net_model_blocks18_bn2 = network.add_elementwise(backbone_net_model_blocks18_bn2.get_output(0), backbone_net_model_blocks17_bn2.get_output(0), trt.ElementWiseOperation.SUM)
    #-----------------------------------------------------------------------------------------------# MBconvBlock8
    #-----------------------------------------------------------------------------------------------# MBconvBlock9
    backbone_net_model_blocks19_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks18_bn2.get_output(0), 
                                                       num_output_maps = 1392, ksize = 1, stride = 1, group = 1, 
                                                       lname = 'backbone_net.model._blocks.19._expand_conv.conv')
    backbone_net_model_blocks19_bn0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks19_expand_conv.get_output(0), lname = 'backbone_net.model._blocks.19._bn0')
    backbone_net_model_blocks19_swish0 = addSwish(network, inputTensor = backbone_net_model_blocks19_bn0.get_output(0))
    backbone_net_model_blocks19_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks19_swish0.get_output(0), 
                                                          num_output_maps = 1392, ksize = 5, stride = 1, group = 1392, 
                                                          lname = 'backbone_net.model._blocks.19._depthwise_conv.conv',pre_padding = (2, 2), post_padding = (2, 2))
    backbone_net_model_blocks19_bn1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = 
                                                     backbone_net_model_blocks19_depthwise_conv.get_output(0), lname = 'backbone_net.model._blocks.19._bn1')
    backbone_net_model_blocks19_swish1 = addSwish(network, inputTensor = backbone_net_model_blocks19_bn1.get_output(0))
    backbone_net_model_blocks19_average_pooling0 = network.add_pooling_nd(backbone_net_model_blocks19_swish1.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (224, 224))
    backbone_net_model_blocks19_average_pooling0.stride = (224, 224)
    backbone_net_model_blocks19_average_pooling1 = network.add_pooling_nd(backbone_net_model_blocks19_average_pooling0.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (2, 2))
    backbone_net_model_blocks19_average_pooling1.stride = (2, 2)
    backbone_net_model_blocks19_se_reduce_conv = convBlock(network = network, weights = weights, 
                                                          inputTensor = backbone_net_model_blocks19_average_pooling1.get_output(0), 
                                                          num_output_maps = 58, ksize = 1, stride = 1, group = 1, 
                                                          lname = 'backbone_net.model._blocks.19._se_reduce.conv')
    backbone_net_model_blocks19_swish2 = addSwish(network, inputTensor = backbone_net_model_blocks19_se_reduce_conv.get_output(0))
    backbone_net_model_blocks19_se_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks19_swish2.get_output(0), num_output_maps = 1392, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.19._se_expand.conv')
    backbone_net_model_blocks19_sigmoid0 = network.add_activation(input = backbone_net_model_blocks19_se_expand_conv.get_output(0), type=trt.ActivationType.SIGMOID)
    backbone_net_model_blocks19_prod0 = network.add_elementwise(backbone_net_model_blocks19_swish1.get_output(0), backbone_net_model_blocks19_sigmoid0.get_output(0), trt.ElementWiseOperation.PROD)
    backbone_net_model_blocks19_project_conv_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks19_prod0.get_output(0), num_output_maps = 232, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.19._project_conv.conv')
    backbone_net_model_blocks19_bn2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks19_project_conv_conv.get_output(0), lname = 'backbone_net.model._blocks.19._bn2')
    backbone_net_model_blocks19_bn2 = network.add_elementwise(backbone_net_model_blocks19_bn2.get_output(0), backbone_net_model_blocks18_bn2.get_output(0), trt.ElementWiseOperation.SUM)
    #-----------------------------------------------------------------------------------------------# MBconvBlock9
    #-----------------------------------------------------------------------------------------------# MBConvBlock20
    backbone_net_model_blocks20_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks19_bn2.get_output(0), 
                                                       num_output_maps = 1392, ksize = 1, stride = 1, group = 1, 
                                                       lname = 'backbone_net.model._blocks.20._expand_conv.conv')
    backbone_net_model_blocks20_bn0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks20_expand_conv.get_output(0), lname = 'backbone_net.model._blocks.20._bn0')
    backbone_net_model_blocks20_swish0 = addSwish(network, inputTensor = backbone_net_model_blocks20_bn0.get_output(0))
    backbone_net_model_blocks20_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks20_swish0.get_output(0), 
                                                          num_output_maps = 1392, ksize = 5, stride = 1, group = 1392, 
                                                          lname = 'backbone_net.model._blocks.20._depthwise_conv.conv',pre_padding = (2, 2), post_padding = (2, 2))
    backbone_net_model_blocks20_bn1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = 
                                                     backbone_net_model_blocks20_depthwise_conv.get_output(0), lname = 'backbone_net.model._blocks.20._bn1')
    backbone_net_model_blocks20_swish1 = addSwish(network, inputTensor = backbone_net_model_blocks20_bn1.get_output(0))
    backbone_net_model_blocks20_average_pooling0 = network.add_pooling_nd(backbone_net_model_blocks20_swish1.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (224, 224))
    backbone_net_model_blocks20_average_pooling0.stride = (224, 224)
    backbone_net_model_blocks20_average_pooling1 = network.add_pooling_nd(backbone_net_model_blocks20_average_pooling0.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (2, 2))
    backbone_net_model_blocks20_average_pooling1.stride = (2, 2)
    backbone_net_model_blocks20_se_reduce_conv = convBlock(network = network, weights = weights, 
                                                          inputTensor = backbone_net_model_blocks20_average_pooling1.get_output(0), 
                                                          num_output_maps = 58, ksize = 1, stride = 1, group = 1, 
                                                          lname = 'backbone_net.model._blocks.20._se_reduce.conv')
    backbone_net_model_blocks20_swish2 = addSwish(network, inputTensor = backbone_net_model_blocks20_se_reduce_conv.get_output(0))
    backbone_net_model_blocks20_se_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks20_swish2.get_output(0), num_output_maps = 1392, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.20._se_expand.conv')
    backbone_net_model_blocks20_sigmoid0 = network.add_activation(input = backbone_net_model_blocks20_se_expand_conv.get_output(0), type=trt.ActivationType.SIGMOID)
    backbone_net_model_blocks20_prod0 = network.add_elementwise(backbone_net_model_blocks20_swish1.get_output(0), backbone_net_model_blocks20_sigmoid0.get_output(0), trt.ElementWiseOperation.PROD)
    backbone_net_model_blocks20_project_conv_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks20_prod0.get_output(0), num_output_maps = 232, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.20._project_conv.conv')
    backbone_net_model_blocks20_bn2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks20_project_conv_conv.get_output(0), lname = 'backbone_net.model._blocks.20._bn2')
    backbone_net_model_blocks20_bn2 = network.add_elementwise(backbone_net_model_blocks20_bn2.get_output(0), backbone_net_model_blocks19_bn2.get_output(0), trt.ElementWiseOperation.SUM)
    #-----------------------------------------------------------------------------------------------# MBConvBlock20
    #-----------------------------------------------------------------------------------------------# MBConvBlock21
    backbone_net_model_blocks21_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks20_bn2.get_output(0), 
                                                       num_output_maps = 1392, ksize = 1, stride = 1, group = 1, 
                                                       lname = 'backbone_net.model._blocks.21._expand_conv.conv')
    backbone_net_model_blocks21_bn0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks21_expand_conv.get_output(0), lname = 'backbone_net.model._blocks.21._bn0')
    backbone_net_model_blocks21_swish0 = addSwish(network, inputTensor = backbone_net_model_blocks21_bn0.get_output(0))
    backbone_net_model_blocks21_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks21_swish0.get_output(0), 
                                                          num_output_maps = 1392, ksize = 5, stride = 1, group = 1392, 
                                                          lname = 'backbone_net.model._blocks.21._depthwise_conv.conv',pre_padding = (2, 2), post_padding = (2, 2))
    backbone_net_model_blocks21_bn1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = 
                                                     backbone_net_model_blocks21_depthwise_conv.get_output(0), lname = 'backbone_net.model._blocks.21._bn1')
    backbone_net_model_blocks21_swish1 = addSwish(network, inputTensor = backbone_net_model_blocks21_bn1.get_output(0))
    backbone_net_model_blocks21_average_pooling0 = network.add_pooling_nd(backbone_net_model_blocks21_swish1.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (224, 224))
    backbone_net_model_blocks21_average_pooling0.stride = (224, 224)
    backbone_net_model_blocks21_average_pooling1 = network.add_pooling_nd(backbone_net_model_blocks21_average_pooling0.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (2, 2))
    backbone_net_model_blocks21_average_pooling1.stride = (2, 2)
    backbone_net_model_blocks21_se_reduce_conv = convBlock(network = network, weights = weights, 
                                                          inputTensor = backbone_net_model_blocks21_average_pooling1.get_output(0), 
                                                          num_output_maps = 58, ksize = 1, stride = 1, group = 1, 
                                                          lname = 'backbone_net.model._blocks.21._se_reduce.conv')
    backbone_net_model_blocks21_swish2 = addSwish(network, inputTensor = backbone_net_model_blocks21_se_reduce_conv.get_output(0))
    backbone_net_model_blocks21_se_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks21_swish2.get_output(0), num_output_maps = 1392, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.21._se_expand.conv')
    backbone_net_model_blocks21_sigmoid0 = network.add_activation(input = backbone_net_model_blocks21_se_expand_conv.get_output(0), type=trt.ActivationType.SIGMOID)
    backbone_net_model_blocks21_prod0 = network.add_elementwise(backbone_net_model_blocks21_swish1.get_output(0), backbone_net_model_blocks21_sigmoid0.get_output(0), trt.ElementWiseOperation.PROD)
    backbone_net_model_blocks21_project_conv_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks21_prod0.get_output(0), num_output_maps = 232, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.21._project_conv.conv')
    backbone_net_model_blocks21_bn2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks21_project_conv_conv.get_output(0), lname = 'backbone_net.model._blocks.21._bn2')
    backbone_net_model_blocks21_bn2 = network.add_elementwise(backbone_net_model_blocks21_bn2.get_output(0), backbone_net_model_blocks20_bn2.get_output(0), trt.ElementWiseOperation.SUM)
    #-----------------------------------------------------------------------------------------------# MBConvBlock21
    #-----------------------------------------------------------------------------------------------# MBConvBlock22
    backbone_net_model_blocks22_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks21_bn2.get_output(0), 
                                                       num_output_maps = 1392, ksize = 1, stride = 1, group = 1, 
                                                       lname = 'backbone_net.model._blocks.22._expand_conv.conv')
    backbone_net_model_blocks22_bn0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks22_expand_conv.get_output(0), lname = 'backbone_net.model._blocks.22._bn0')
    backbone_net_model_blocks22_swish0 = addSwish(network, inputTensor = backbone_net_model_blocks22_bn0.get_output(0))
    backbone_net_model_blocks22_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks22_swish0.get_output(0), 
                                                          num_output_maps = 1392, ksize = 5, stride = 1, group = 1392, 
                                                          lname = 'backbone_net.model._blocks.22._depthwise_conv.conv',pre_padding = (2, 2), post_padding = (2, 2))
    backbone_net_model_blocks22_bn1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = 
                                                     backbone_net_model_blocks22_depthwise_conv.get_output(0), lname = 'backbone_net.model._blocks.22._bn1')
    backbone_net_model_blocks22_swish1 = addSwish(network, inputTensor = backbone_net_model_blocks22_bn1.get_output(0))
    backbone_net_model_blocks22_average_pooling0 = network.add_pooling_nd(backbone_net_model_blocks22_swish1.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (224, 224))
    backbone_net_model_blocks22_average_pooling0.stride = (224, 224)
    backbone_net_model_blocks22_average_pooling1 = network.add_pooling_nd(backbone_net_model_blocks22_average_pooling0.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (2, 2))
    backbone_net_model_blocks22_average_pooling1.stride = (2, 2)
    backbone_net_model_blocks22_se_reduce_conv = convBlock(network = network, weights = weights, 
                                                          inputTensor = backbone_net_model_blocks22_average_pooling1.get_output(0), 
                                                          num_output_maps = 58, ksize = 1, stride = 1, group = 1, 
                                                          lname = 'backbone_net.model._blocks.22._se_reduce.conv')
    backbone_net_model_blocks22_swish2 = addSwish(network, inputTensor = backbone_net_model_blocks22_se_reduce_conv.get_output(0))
    backbone_net_model_blocks22_se_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks22_swish2.get_output(0), num_output_maps = 1392, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.22._se_expand.conv')
    backbone_net_model_blocks22_sigmoid0 = network.add_activation(input = backbone_net_model_blocks22_se_expand_conv.get_output(0), type=trt.ActivationType.SIGMOID)
    backbone_net_model_blocks22_prod0 = network.add_elementwise(backbone_net_model_blocks22_swish1.get_output(0), backbone_net_model_blocks22_sigmoid0.get_output(0), trt.ElementWiseOperation.PROD)
    backbone_net_model_blocks22_project_conv_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks22_prod0.get_output(0), num_output_maps = 232, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.22._project_conv.conv')
    backbone_net_model_blocks22_bn2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks22_project_conv_conv.get_output(0), lname = 'backbone_net.model._blocks.22._bn2')
    backbone_net_model_blocks22_bn2 = network.add_elementwise(backbone_net_model_blocks22_bn2.get_output(0), backbone_net_model_blocks21_bn2.get_output(0), trt.ElementWiseOperation.SUM)
    #-----------------------------------------------------------------------------------------------# MBConvBlock22
    #-----------------------------------------------------------------------------------------------# MBConvBlock23
    backbone_net_model_blocks23_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks22_bn2.get_output(0), 
                                                       num_output_maps = 1392, ksize = 1, stride = 1, group = 1, 
                                                       lname = 'backbone_net.model._blocks.23._expand_conv.conv')
    backbone_net_model_blocks23_bn0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks23_expand_conv.get_output(0), lname = 'backbone_net.model._blocks.23._bn0')
    backbone_net_model_blocks23_swish0 = addSwish(network, inputTensor = backbone_net_model_blocks23_bn0.get_output(0))
    backbone_net_model_blocks23_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks23_swish0.get_output(0), 
                                                          num_output_maps = 1392, ksize = 5, stride = 1, group = 1392, 
                                                          lname = 'backbone_net.model._blocks.23._depthwise_conv.conv',pre_padding = (2, 2), post_padding = (2, 2))
    backbone_net_model_blocks23_bn1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = 
                                                     backbone_net_model_blocks23_depthwise_conv.get_output(0), lname = 'backbone_net.model._blocks.23._bn1')
    backbone_net_model_blocks23_swish1 = addSwish(network, inputTensor = backbone_net_model_blocks23_bn1.get_output(0))
    backbone_net_model_blocks23_average_pooling0 = network.add_pooling_nd(backbone_net_model_blocks23_swish1.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (224, 224))
    backbone_net_model_blocks23_average_pooling0.stride = (224, 224)
    backbone_net_model_blocks23_average_pooling1 = network.add_pooling_nd(backbone_net_model_blocks23_average_pooling0.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (2, 2))
    backbone_net_model_blocks23_average_pooling1.stride = (2, 2)
    backbone_net_model_blocks23_se_reduce_conv = convBlock(network = network, weights = weights, 
                                                          inputTensor = backbone_net_model_blocks23_average_pooling1.get_output(0), 
                                                          num_output_maps = 58, ksize = 1, stride = 1, group = 1, 
                                                          lname = 'backbone_net.model._blocks.23._se_reduce.conv')
    backbone_net_model_blocks23_swish2 = addSwish(network, inputTensor = backbone_net_model_blocks23_se_reduce_conv.get_output(0))
    backbone_net_model_blocks23_se_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks23_swish2.get_output(0), num_output_maps = 1392, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.23._se_expand.conv')
    backbone_net_model_blocks23_sigmoid0 = network.add_activation(input = backbone_net_model_blocks23_se_expand_conv.get_output(0), type=trt.ActivationType.SIGMOID)
    backbone_net_model_blocks23_prod0 = network.add_elementwise(backbone_net_model_blocks23_swish1.get_output(0), backbone_net_model_blocks23_sigmoid0.get_output(0), trt.ElementWiseOperation.PROD)
    backbone_net_model_blocks23_project_conv_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks23_prod0.get_output(0), num_output_maps = 232, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.23._project_conv.conv')
    backbone_net_model_blocks23_bn2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks23_project_conv_conv.get_output(0), lname = 'backbone_net.model._blocks.23._bn2')
    backbone_net_model_blocks23_bn2 = network.add_elementwise(backbone_net_model_blocks23_bn2.get_output(0), backbone_net_model_blocks22_bn2.get_output(0), trt.ElementWiseOperation.SUM)
    #-----------------------------------------------------------------------------------------------# MBConvBlock23
    #-----------------------------------------------------------------------------------------------# MBConvBlock24
    backbone_net_model_blocks24_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks23_bn2.get_output(0), 
                                                       num_output_maps = 1392, ksize = 1, stride = 1, group = 1, 
                                                       lname = 'backbone_net.model._blocks.24._expand_conv.conv')
    backbone_net_model_blocks24_bn0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks24_expand_conv.get_output(0), lname = 'backbone_net.model._blocks.24._bn0')
    backbone_net_model_blocks24_swish0 = addSwish(network, inputTensor = backbone_net_model_blocks24_bn0.get_output(0))
    backbone_net_model_blocks24_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks24_swish0.get_output(0), 
                                                          num_output_maps = 1392, ksize = 3, stride = 1, group = 1392, 
                                                          lname = 'backbone_net.model._blocks.24._depthwise_conv.conv',pre_padding = (1, 1), post_padding = (1, 1))
    backbone_net_model_blocks24_bn1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = 
                                                     backbone_net_model_blocks24_depthwise_conv.get_output(0), lname = 'backbone_net.model._blocks.24._bn1')
    backbone_net_model_blocks24_swish1 = addSwish(network, inputTensor = backbone_net_model_blocks24_bn1.get_output(0))
    backbone_net_model_blocks24_average_pooling0 = network.add_pooling_nd(backbone_net_model_blocks24_swish1.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (224, 224))
    backbone_net_model_blocks24_average_pooling0.stride = (224, 224)
    backbone_net_model_blocks24_average_pooling1 = network.add_pooling_nd(backbone_net_model_blocks24_average_pooling0.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (2, 2))
    backbone_net_model_blocks24_average_pooling1.stride = (2, 2)
    backbone_net_model_blocks24_se_reduce_conv = convBlock(network = network, weights = weights, 
                                                          inputTensor = backbone_net_model_blocks24_average_pooling1.get_output(0), 
                                                          num_output_maps = 58, ksize = 1, stride = 1, group = 1, 
                                                          lname = 'backbone_net.model._blocks.24._se_reduce.conv')
    backbone_net_model_blocks24_swish2 = addSwish(network, inputTensor = backbone_net_model_blocks24_se_reduce_conv.get_output(0))
    backbone_net_model_blocks24_se_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks24_swish2.get_output(0), num_output_maps = 1392, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.24._se_expand.conv')
    backbone_net_model_blocks24_sigmoid0 = network.add_activation(input = backbone_net_model_blocks24_se_expand_conv.get_output(0), type=trt.ActivationType.SIGMOID)
    backbone_net_model_blocks24_prod0 = network.add_elementwise(backbone_net_model_blocks24_swish1.get_output(0), backbone_net_model_blocks24_sigmoid0.get_output(0), trt.ElementWiseOperation.PROD)
    backbone_net_model_blocks24_project_conv_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks24_prod0.get_output(0), num_output_maps = 384, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.24._project_conv.conv')
    backbone_net_model_blocks24_bn2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks24_project_conv_conv.get_output(0), lname = 'backbone_net.model._blocks.24._bn2')
    #backbone_net_model_blocks24_bn2 = network.add_elementwise(backbone_net_model_blocks24_bn2.get_output(0), backbone_net_model_blocks23_bn2.get_output(0), trt.ElementWiseOperation.SUM)
    #-----------------------------------------------------------------------------------------------# MBConvBlock24
    #-----------------------------------------------------------------------------------------------# MBConvBlock25
    backbone_net_model_blocks25_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks24_bn2.get_output(0), 
                                                       num_output_maps = 2304, ksize = 1, stride = 1, group = 1, 
                                                       lname = 'backbone_net.model._blocks.25._expand_conv.conv')
    backbone_net_model_blocks25_bn0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks25_expand_conv.get_output(0), lname = 'backbone_net.model._blocks.25._bn0')
    backbone_net_model_blocks25_swish0 = addSwish(network, inputTensor = backbone_net_model_blocks25_bn0.get_output(0))
    backbone_net_model_blocks25_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks25_swish0.get_output(0), 
                                                          num_output_maps = 2304, ksize = 3, stride = 1, group = 2304, 
                                                          lname = 'backbone_net.model._blocks.25._depthwise_conv.conv',pre_padding = (1, 1), post_padding = (1, 1))
    backbone_net_model_blocks25_bn1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = 
                                                     backbone_net_model_blocks25_depthwise_conv.get_output(0), lname = 'backbone_net.model._blocks.25._bn1')
    backbone_net_model_blocks25_swish1 = addSwish(network, inputTensor = backbone_net_model_blocks25_bn1.get_output(0))
    backbone_net_model_blocks25_average_pooling0 = network.add_pooling_nd(backbone_net_model_blocks25_swish1.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (224, 224))
    backbone_net_model_blocks25_average_pooling0.stride = (224, 224)
    backbone_net_model_blocks25_average_pooling1 = network.add_pooling_nd(backbone_net_model_blocks25_average_pooling0.get_output(0), type = trt.PoolingType.AVERAGE, window_size = (2, 2))
    backbone_net_model_blocks25_average_pooling1.stride = (2, 2)
    backbone_net_model_blocks25_se_reduce_conv = convBlock(network = network, weights = weights, 
                                                          inputTensor = backbone_net_model_blocks25_average_pooling1.get_output(0), 
                                                          num_output_maps = 96, ksize = 1, stride = 1, group = 1, 
                                                          lname = 'backbone_net.model._blocks.25._se_reduce.conv')
    backbone_net_model_blocks25_swish2 = addSwish(network, inputTensor = backbone_net_model_blocks25_se_reduce_conv.get_output(0))
    backbone_net_model_blocks25_se_expand_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks25_swish2.get_output(0), num_output_maps = 2304, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.25._se_expand.conv')
    backbone_net_model_blocks25_sigmoid0 = network.add_activation(input = backbone_net_model_blocks25_se_expand_conv.get_output(0), type=trt.ActivationType.SIGMOID)
    backbone_net_model_blocks25_prod0 = network.add_elementwise(backbone_net_model_blocks25_swish1.get_output(0), backbone_net_model_blocks25_sigmoid0.get_output(0), trt.ElementWiseOperation.PROD)
    backbone_net_model_blocks25_project_conv_conv = convBlock(network = network, weights = weights, inputTensor = backbone_net_model_blocks25_prod0.get_output(0), num_output_maps = 384, ksize = 1, stride = 1, group = 1, lname = 'backbone_net.model._blocks.25._project_conv.conv')
    backbone_net_model_blocks25_bn2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = backbone_net_model_blocks25_project_conv_conv.get_output(0), lname = 'backbone_net.model._blocks.25._bn2')
    backbone_net_model_blocks25_bn2 = network.add_elementwise(backbone_net_model_blocks25_bn2.get_output(0), backbone_net_model_blocks24_bn2.get_output(0), trt.ElementWiseOperation.SUM)
    #-----------------------------------------------------------------------------------------------# MBConvBlock25
    #-----------------------------------------------------------------------------------------------# EfficientNet
    #-----------------------------------------------------------------------------------------------# EfficientDetBackbone
    p3=backbone_net_model_blocks7_bn2
    p4=backbone_net_model_blocks17_bn2
    p5=backbone_net_model_blocks25_bn2

    return p3, p4, p5

def BiFPN(network, weights, input_tenors):
    p3, p4, p5 = input_tenors
    # ----------------------------p5_to_p6----------------------------
    bifpn0_p5_to_p6_0_conv = convBlock(network = network, 
                        weights = weights, 
                        inputTensor = p5.get_output(0), 
                        num_output_maps = 160, 
                        ksize = 1, 
                        stride = 1, 
                        group = 1, 
                        lname = 'bifpn.0.p5_to_p6.0.conv',
                        pre_padding = (0, 0),
                        post_padding = (0, 0))
    bifpn0_p5_to_p6_1_bn = addBatchNorm2d(network = network, 
                       weights = weights, 
                       inputTensor = bifpn0_p5_to_p6_0_conv.get_output(0),
                       lname = 'bifpn.0.p5_to_p6.1')
    p6_in = maxPoolingBlock(network = network, 
                       weights = weights,
                       inputTensor=bifpn0_p5_to_p6_1_bn.get_output(0),
                       kernel_size=3,
                       stride=2,
                       pre_padding=(0,0),
                       post_padding=(1,1))
# ----------------------------p6_to_p7----------------------------
    p7_in = maxPoolingBlock(network = network, 
                        weights = weights,
                        inputTensor=p6_in.get_output(0),
                        kernel_size=3,
                        stride=2,
                        pre_padding=(0,0),
                        post_padding=(1,1))
# ----------------------------p6_to_p7----------------------------
# ----------------------------p3_down_channel----------------------------
    bifpn0_p3_down_channel0_conv = convBlock(network = network,
                        weights = weights,
                        inputTensor = p3.get_output(0), 
                        num_output_maps = 160, 
                        ksize = 1, 
                        stride = 1, 
                        group = 1, 
                        lname = 'bifpn.0.p3_down_channel.0.conv',
                        pre_padding = (0, 0),
                        post_padding = (0, 0))
    p3_in  = addBatchNorm2d(network = network,
                            weights = weights, 
                            inputTensor = bifpn0_p3_down_channel0_conv.get_output(0), 
                            lname = 'bifpn.0.p3_down_channel.1')
# ----------------------------p3_down_channel----------------------------
# ----------------------------p4_down_channel----------------------------
    bifpn0_p4_down_channel0_conv = convBlock(network = network, 
                        weights = weights, 
                        inputTensor = p4.get_output(0), 
                        num_output_maps = 160, 
                        ksize = 1, 
                        stride = 1, 
                        group = 1, 
                        lname = 'bifpn.0.p4_down_channel.0.conv',
                        pre_padding = (0, 0), 
                        post_padding = (0, 0))
    p4_in  = addBatchNorm2d(network = network,
                        weights = weights, 
                        inputTensor = bifpn0_p4_down_channel0_conv.get_output(0), 
                        lname = 'bifpn.0.p4_down_channel.1')    
# ----------------------------p5_down_channel----------------------------
    bifpn0_p5_down_channel0_conv = convBlock(network = network, 
                        weights = weights, 
                        inputTensor = p5.get_output(0), 
                        num_output_maps = 160, 
                        ksize = 1, 
                        stride = 1, 
                        group = 1, 
                        lname = 'bifpn.0.p5_down_channel.0.conv',
                        pre_padding = (0, 0), 
                        post_padding = (0, 0))
    p5_in  = addBatchNorm2d(network = network,
                        weights = weights, 
                        inputTensor = bifpn0_p5_down_channel0_conv.get_output(0),
                        lname = 'bifpn.0.p5_down_channel.1')

    lname="bifpn.0"
    p6_w1=weights[lname+".p6_w1"]
    p5_w1=weights[lname+".p5_w1"]
    p4_w1=weights[lname+".p4_w1"]
    p3_w1=weights[lname+".p3_w1"]

    p4_w2=weights[lname+".p4_w2"]
    p5_w2=weights[lname+".p5_w2"]
    p6_w2=weights[lname+".p6_w2"]
    p7_w2=weights[lname+".p7_w2"]
    
    epsilon=1e-4
    # ----------------------- P6_0 and P7_0 to P6_1-----------------------------
    p6_w1_relu=p6_w1*(p6_w1>0)
    p6_w1_sum=np.sum(p6_w1_relu.numpy())+epsilon
    weight6=(p6_w1_relu/p6_w1_sum).numpy()
    weight0_p6_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p6_in_scale = np.full((160,),weight6[0])
    weight0_p6_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_in_w0=network.add_scale(p6_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p6_in_shift,scale=weight0_p6_in_scale)
    #p7_in_upsample = addUpsample(network, p7_in.get_output(0))

    p6_upsample = addUpsample(network, p7_in.get_output(0), [1, 2, 2])

    weight1_p6p7_in_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p6p7_in_scale = np.full((160,),weight6[1])
    weight1_p6p7_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_in_w1=network.add_scale(p6_upsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p6p7_in_shift,scale=weight1_p6p7_in_scale)

    p6_in_w = network.add_elementwise(p6_in_w0.get_output(0), p6_in_w1.get_output(0), trt.ElementWiseOperation.SUM)
    p6_up_swish = addSwish(network = network,inputTensor = p6_in_w.get_output(0))

    p6_up_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p6_up_swish.get_output(0), num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = 'bifpn.0.conv6_up.depthwise_conv.conv', pre_padding = (1, 1), post_padding = (1, 1))
    p6_up_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p6_up_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'bifpn.0.conv6_up.pointwise_conv.conv')
    p6_up_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p6_up_pointwise_conv.get_output(0),
                         lname = 'bifpn.0.conv6_up.bn')
    # ----------------------- P5_0 and P6_1 to P5_1-----------------------------
    p5_w1_relu=p5_w1*(p5_w1>0)
    p5_w1_sum=np.sum(p5_w1_relu.numpy())+epsilon
    weight5=(p5_w1_relu/p5_w1_sum).numpy()
    weight0_p5_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p5_in_scale = np.full((160,),weight5[0])
    weight0_p5_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p5_in_w0=network.add_scale(p5_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p5_in_shift,scale=weight0_p5_in_scale)
    #p7_in_upsample = addUpsample(network, p7_in.get_output(0))

    p6_up_upsample = addUpsample(network, p6_up_bn.get_output(0), [1, 2, 2])

    weight1_p5p6up_in_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p5p6up_in_scale = np.full((160,),weight5[1])
    weight1_p5p6up_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_up_w=network.add_scale(p6_up_upsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p5p6up_in_shift,scale=weight1_p5p6up_in_scale)

    p5p6up_in_w = network.add_elementwise(p5_in_w0.get_output(0), p6_up_w.get_output(0), trt.ElementWiseOperation.SUM)
    p5p6up_swish = addSwish(network = network,inputTensor = p5p6up_in_w.get_output(0))

    p5_up_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p5p6up_swish.get_output(0), num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = 'bifpn.0.conv5_up.depthwise_conv.conv', pre_padding = (1, 1), post_padding = (1, 1))
    p5_up_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p5_up_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'bifpn.0.conv5_up.pointwise_conv.conv')
    p5_up_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p5_up_pointwise_conv.get_output(0),
                         lname = 'bifpn.0.conv5_up.bn')    

    # ----------------------- P4_0 and P5_1 to P4_1-----------------------------
    p4_w1_relu=p4_w1*(p4_w1>0)
    p4_w1_sum=np.sum(p4_w1_relu.numpy())+epsilon
    weight4=(p4_w1_relu/p4_w1_sum).numpy()
    weight0_p4_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p4_in_scale = np.full((160,),weight4[0])
    weight0_p4_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p4_in_w0=network.add_scale(p4_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p4_in_shift,scale=weight0_p4_in_scale)
    #p7_in_upsample = addUpsample(network, p7_in.get_output(0))

    p5_up_upsample = addUpsample(network, p5_up_bn.get_output(0), [1, 2, 2])

    weight1_p4p5up_in_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p4p5up_in_scale = np.full((160,),weight4[1])
    weight1_p4p5up_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p5_up_w=network.add_scale(p5_up_upsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p4p5up_in_shift,scale=weight1_p4p5up_in_scale)

    p4p5up_in_w = network.add_elementwise(p4_in_w0.get_output(0), p5_up_w.get_output(0), trt.ElementWiseOperation.SUM)
    p4p5up_swish = addSwish(network = network,inputTensor = p4p5up_in_w.get_output(0))

    p4_up_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p4p5up_swish.get_output(0), num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = 'bifpn.0.conv4_up.depthwise_conv.conv', pre_padding = (1, 1), post_padding = (1, 1))
    p4_up_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p4_up_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'bifpn.0.conv4_up.pointwise_conv.conv')
    p4_up_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p4_up_pointwise_conv.get_output(0),
                         lname = 'bifpn.0.conv4_up.bn') 

# ----------------------- P3_0 and P4_1 to P3_2-----------------------------
    p3_w1_relu=p3_w1*(p3_w1>0)
    p3_w1_sum=np.sum(p3_w1_relu.numpy())+epsilon
    weight3=(p3_w1_relu/p3_w1_sum).numpy()
    weight0_p3_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p3_in_scale = np.full((160,),weight3[0])
    weight0_p3_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p3_in_w0=network.add_scale(p3_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p3_in_shift,scale=weight0_p3_in_scale)
    #p7_in_upsample = addUpsample(network, p7_in.get_output(0))

    p4_up_upsample = addUpsample(network, p4_up_bn.get_output(0), [1, 2, 2])

    weight1_p3p4up_in_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p3p4up_in_scale = np.full((160,),weight3[1])
    weight1_p3p4up_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p4_up_w=network.add_scale(p4_up_upsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p3p4up_in_shift,scale=weight1_p3p4up_in_scale)

    p3p4up_in_w = network.add_elementwise(p3_in_w0.get_output(0), p4_up_w.get_output(0), trt.ElementWiseOperation.SUM)
    p3p4up_swish = addSwish(network = network,inputTensor = p3p4up_in_w.get_output(0))

    p3_out_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p3p4up_swish.get_output(0), num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = 'bifpn.0.conv3_up.depthwise_conv.conv', pre_padding = (1, 1), post_padding = (1, 1))
    p3_out_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p3_out_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'bifpn.0.conv3_up.pointwise_conv.conv')
    p3_out_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p3_out_pointwise_conv.get_output(0),
                         lname = 'bifpn.0.conv3_up.bn') 

    p4_down_channel_2_conv = convBlock(network = network, weights = weights, inputTensor = p4.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'bifpn.0.p4_down_channel_2.0.conv')

    p4_in_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p4_down_channel_2_conv.get_output(0),
                         lname = 'bifpn.0.p4_down_channel_2.1') 

    p5_down_channel_2_conv = convBlock(network = network, weights = weights, inputTensor = p5.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'bifpn.0.p5_down_channel_2.0.conv')

    p5_in_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p5_down_channel_2_conv.get_output(0),
                         lname = 'bifpn.0.p5_down_channel_2.1') 
    # ----------------------- P4_0, P4_1 and p3_2 to P4_2-----------------------------
    p4_w2_relu=p4_w2*(p4_w2>0)
    p4_w2_sum=np.sum(p4_w2_relu.numpy())+epsilon
    weight4_2=(p4_w2_relu/p4_w2_sum).numpy()
    weight0_p4_2_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p4_2_in_scale = np.full((160,),weight4_2[0])
    weight0_p4_2_in_scale.astype(np.float32)
    #weight_p4_2_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p4_2_in_w0=network.add_scale(p4_in_bn.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p4_2_in_shift,scale=weight0_p4_2_in_scale)
    
    
    weight0_p4_2_up_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p4_2_up_scale = np.full((160,),weight4_2[1])
    weight0_p4_2_up_scale.astype(np.float32)
    p4_2_up_w0=network.add_scale(p4_up_bn.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p4_2_up_shift,scale=weight0_p4_2_up_scale)
    
    p3_out_downsample = maxPoolingBlock(network, weights, p3_out_bn.get_output(0),3, 2,  pre_padding = (0, 0), post_padding = (1,1))

    
    weight1_p3_out_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p3_out_scale = np.full((160,),weight4_2[2])
    weight1_p3_out_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p3_out_down_w=network.add_scale(p3_out_downsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p3_out_shift,scale=weight1_p3_out_scale)
    
    p4_0_p4_2_sum = network.add_elementwise(p4_2_in_w0.get_output(0), p4_2_up_w0.get_output(0), trt.ElementWiseOperation.SUM)
    
    p4_0_p4_2_p3_2_sum = network.add_elementwise(p4_0_p4_2_sum.get_output(0), p3_out_down_w.get_output(0), trt.ElementWiseOperation.SUM)

    p4_0_p4_2_p3_2_swish = addSwish(network = network,inputTensor = p4_0_p4_2_p3_2_sum.get_output(0))

    
    p4_out_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p4_0_p4_2_p3_2_swish.get_output(0), num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = 'bifpn.0.conv4_down.depthwise_conv.conv', pre_padding = (1, 1), post_padding = (1, 1))
    p4_out_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p4_out_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'bifpn.0.conv4_down.pointwise_conv.conv')
    p4_out_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p4_out_pointwise_conv.get_output(0),
                         lname = 'bifpn.0.conv4_down.bn') 

    # -----------------------  Weights for P5_0, P5_1 and P4_2 to P5_2-----------------------------
    p5_w2_relu=p5_w2*(p5_w2>0)
    p5_w2_sum=np.sum(p5_w2_relu.numpy())+epsilon

    weight5_2=(p5_w2_relu/p5_w2_sum).numpy()
    weight0_p5_2_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p5_2_in_scale = np.full((160,),weight5_2[0])
    weight0_p5_2_in_scale.astype(np.float32)
    #weight_p4_2_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p5_in_w0=network.add_scale(p5_in_bn.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p5_2_in_shift,scale=weight0_p5_2_in_scale)
    
    
    weight0_p5_2_up_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p5_2_up_scale = np.full((160,),weight5_2[1])
    weight0_p5_2_up_scale.astype(np.float32)
    p5_up_w1=network.add_scale(p5_up_bn.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p5_2_up_shift,scale=weight0_p5_2_up_scale)
    

    p4_out_downsample = maxPoolingBlock(network, weights, p4_out_bn.get_output(0),3, 2,  pre_padding = (0, 0), post_padding = (1,1))
    weight1_p4_out_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p4_out_scale = np.full((160,),weight4_2[2])
    weight1_p4_out_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p4_out_down_w=network.add_scale(p4_out_downsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p4_out_shift,scale=weight1_p4_out_scale)
    

    p5_0_p5_2_sum = network.add_elementwise(p5_in_w0.get_output(0), p5_up_w1.get_output(0), trt.ElementWiseOperation.SUM)
    
    p5_0_p5_2_p4_2_sum = network.add_elementwise(p5_0_p5_2_sum.get_output(0), p4_out_down_w.get_output(0), trt.ElementWiseOperation.SUM)

    p5_0_p5_2_p4_2_swish = addSwish(network = network,inputTensor = p5_0_p5_2_p4_2_sum.get_output(0))

    
    p5_out_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p5_0_p5_2_p4_2_swish.get_output(0), num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = 'bifpn.0.conv5_down.depthwise_conv.conv', pre_padding = (1, 1), post_padding = (1, 1))
    p5_out_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p5_out_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'bifpn.0.conv5_down.pointwise_conv.conv')
    p5_out_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p5_out_pointwise_conv.get_output(0),
                         lname = 'bifpn.0.conv5_down.bn') 
    # -----------------------  #  Weights for P6_0, P6_1 and P5_2 to P6_2-----------------------------
    p6_w2_relu=p6_w2*(p6_w2>0)
    p6_w2_sum=np.sum(p6_w2_relu.numpy())+epsilon

    weight6_2=(p6_w2_relu/p6_w2_sum).numpy()
    weight0_p6_2_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p6_2_in_scale = np.full((160,),weight6_2[0])
    weight0_p6_2_in_scale.astype(np.float32)
    #weight_p4_2_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_in_w0=network.add_scale(p6_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p6_2_in_shift,scale=weight0_p6_2_in_scale)
    
    
    weight0_p6_2_up_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p6_2_up_scale = np.full((160,),weight6_2[1])
    weight0_p6_2_up_scale.astype(np.float32)
    p6_up_w1=network.add_scale(p6_up_bn.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p6_2_up_shift,scale=weight0_p6_2_up_scale)
    

    p5_out_downsample = maxPoolingBlock(network, weights, p5_out_bn.get_output(0),3, 2,  pre_padding = (0, 0), post_padding = (1,1))
    weight1_p5_out_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p5_out_scale = np.full((160,),weight6_2[2])
    weight1_p5_out_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p5_out_down_w=network.add_scale(p5_out_downsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p5_out_shift,scale=weight1_p5_out_scale)
    

    p6_0_p6_2_sum = network.add_elementwise(p6_in_w0.get_output(0), p6_up_w1.get_output(0), trt.ElementWiseOperation.SUM)
    
    p6_0_p6_2_p5_2_sum = network.add_elementwise(p6_0_p6_2_sum.get_output(0), p5_out_down_w.get_output(0), trt.ElementWiseOperation.SUM)

    p6_0_p6_2_p5_2_swish = addSwish(network = network,inputTensor = p6_0_p6_2_p5_2_sum.get_output(0))

    
    p6_out_depthwise_conv = convBlock(network = network, weights = weights, 
                                inputTensor = p6_0_p6_2_p5_2_swish.get_output(0), 
                                num_output_maps = 160, ksize = 3, stride = 1, group = 160,
                                lname = 'bifpn.0.conv6_down.depthwise_conv.conv',
                                pre_padding = (1, 1), post_padding = (1, 1))
    p6_out_pointwise_conv = convBlock(network = network, weights = weights, 
                                inputTensor = p6_out_depthwise_conv.get_output(0),
                                num_output_maps = 160, ksize = 1, stride = 1, group = 1,
                                lname = 'bifpn.0.conv6_down.pointwise_conv.conv')
    p6_out_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p6_out_pointwise_conv.get_output(0),
                         lname = 'bifpn.0.conv6_down.bn') 
    # -----------------------  # # Weights for P7_0 and P6_2 to P7_2-----------------------------
    p7_w2_relu=p7_w2*(p7_w2>0)
    p7_w2_sum=np.sum(p7_w2_relu.numpy())+epsilon

    weight7_2=(p7_w2_relu/p7_w2_sum).numpy()
    weight0_p7_2_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p7_2_in_scale = np.full((160,),weight7_2[0])
    weight0_p7_2_in_scale.astype(np.float32)
    #weight_p4_2_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p7_in_w0=network.add_scale(p7_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p7_2_in_shift,scale=weight0_p7_2_in_scale)
   
    p6_out_downsample = maxPoolingBlock(network, weights, p6_out_bn.get_output(0),3, 2,  pre_padding = (0, 0), post_padding = (1,1))
    weight1_p6_out_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p6_out_scale = np.full((160,),weight7_2[1])
    weight1_p6_out_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_out_down_w=network.add_scale(p6_out_downsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p6_out_shift,scale=weight1_p6_out_scale)
        
    p7_0_p6_1_sum = network.add_elementwise(p7_in_w0.get_output(0), p6_out_down_w.get_output(0), trt.ElementWiseOperation.SUM)

    p7_0_p6_1_swish = addSwish(network = network,inputTensor = p7_0_p6_1_sum.get_output(0))

    p7_out_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p7_0_p6_1_swish.get_output(0), num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = 'bifpn.0.conv7_down.depthwise_conv.conv', pre_padding = (1, 1), post_padding = (1, 1))
    p7_out_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p7_out_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'bifpn.0.conv7_down.pointwise_conv.conv')
    p7_out_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p7_out_pointwise_conv.get_output(0),
                         lname = 'bifpn.0.conv7_down.bn')

# --------------------------------------------bifpn1--------------------------------
    lname="bifpn.1"
    p3_in=p3_out_bn
    p4_in=p4_out_bn
    p5_in=p5_out_bn
    p6_in=p6_out_bn
    p7_in=p7_out_bn

    p6_w1=weights[lname+".p6_w1"]
    p5_w1=weights[lname+".p5_w1"]
    p4_w1=weights[lname+".p4_w1"]
    p3_w1=weights[lname+".p3_w1"]

    p4_w2=weights[lname+".p4_w2"]
    p5_w2=weights[lname+".p5_w2"]
    p6_w2=weights[lname+".p6_w2"]
    p7_w2=weights[lname+".p7_w2"]
    
    epsilon=1e-4
    # ----------------------- P6_0 and P7_0 to P6_1-----------------------------
    p6_w1_relu=p6_w1*(p6_w1>0)
    p6_w1_sum=np.sum(p6_w1_relu.numpy())+epsilon
    weight6=(p6_w1_relu/p6_w1_sum).numpy()
    weight0_p6_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p6_in_scale = np.full((160,),weight6[0])
    weight0_p6_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_in_w0=network.add_scale(p6_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p6_in_shift,scale=weight0_p6_in_scale)
    #p7_in_upsample = addUpsample(network, p7_in.get_output(0))

    p6_upsample = addUpsample(network, p7_in.get_output(0), [1, 2, 2])

    weight1_p6p7_in_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p6p7_in_scale = np.full((160,),weight6[1])
    weight1_p6p7_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_in_w1=network.add_scale(p6_upsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p6p7_in_shift,scale=weight1_p6p7_in_scale)

    p6_in_w = network.add_elementwise(p6_in_w0.get_output(0), p6_in_w1.get_output(0), trt.ElementWiseOperation.SUM)
    p6_up_swish = addSwish(network = network,inputTensor = p6_in_w.get_output(0))

    p6_up_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p6_up_swish.get_output(0), 
                        num_output_maps = 160, ksize = 3, stride = 1, group = 160, 
                        lname = lname+'.conv6_up.depthwise_conv.conv', 
                        pre_padding = (1, 1), post_padding = (1, 1))
    p6_up_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p6_up_depthwise_conv.get_output(0), 
                        num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                        lname = lname+'.conv6_up.pointwise_conv.conv')
    p6_up_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p6_up_pointwise_conv.get_output(0),
                         lname = lname+'.conv6_up.bn')
    # 已验证
    # ----------------------- P5_0 and P6_1 to P5_1-----------------------------
    p5_w1_relu=p5_w1*(p5_w1>0)
    p5_w1_sum=np.sum(p5_w1_relu.numpy())+epsilon
    weight5=(p5_w1_relu/p5_w1_sum).numpy()
    weight0_p5_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p5_in_scale = np.full((160,),weight5[0])
    weight0_p5_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p5_in_w0=network.add_scale(p5_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p5_in_shift,scale=weight0_p5_in_scale)
    #p7_in_upsample = addUpsample(network, p7_in.get_output(0))

    p6_up_upsample = addUpsample(network, p6_up_bn.get_output(0), [1, 2, 2])

    weight1_p5p6up_in_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p5p6up_in_scale = np.full((160,),weight5[1])
    weight1_p5p6up_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_up_w=network.add_scale(p6_up_upsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p5p6up_in_shift,scale=weight1_p5p6up_in_scale)

    p5p6up_in_w = network.add_elementwise(p5_in_w0.get_output(0), p6_up_w.get_output(0), trt.ElementWiseOperation.SUM)
    p5p6up_swish = addSwish(network = network,inputTensor = p5p6up_in_w.get_output(0))

    p5_up_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p5p6up_swish.get_output(0), 
                num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv5_up.depthwise_conv.conv', 
                pre_padding = (1, 1), post_padding = (1, 1))
    p5_up_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p5_up_depthwise_conv.get_output(0), 
                num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                lname = lname+'.conv5_up.pointwise_conv.conv')
    p5_up_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p5_up_pointwise_conv.get_output(0),
                         lname = lname+'.conv5_up.bn')    
    # 已验证
    # ----------------------- P4_0 and P5_1 to P4_1-----------------------------
    p4_w1_relu=p4_w1*(p4_w1>0)
    p4_w1_sum=np.sum(p4_w1_relu.numpy())+epsilon
    weight4=(p4_w1_relu/p4_w1_sum).numpy()
    weight0_p4_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p4_in_scale = np.full((160,),weight4[0])
    weight0_p4_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p4_in_w0=network.add_scale(p4_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p4_in_shift,scale=weight0_p4_in_scale)
    #p7_in_upsample = addUpsample(network, p7_in.get_output(0))

    p5_up_upsample = addUpsample(network, p5_up_bn.get_output(0), [1, 2, 2])

    weight1_p4p5up_in_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p4p5up_in_scale = np.full((160,),weight4[1])
    weight1_p4p5up_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p5_up_w=network.add_scale(p5_up_upsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p4p5up_in_shift,scale=weight1_p4p5up_in_scale)

    p4p5up_in_w = network.add_elementwise(p4_in_w0.get_output(0), p5_up_w.get_output(0), trt.ElementWiseOperation.SUM)
    p4p5up_swish = addSwish(network = network,inputTensor = p4p5up_in_w.get_output(0))

    p4_up_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p4p5up_swish.get_output(0), 
                        num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv4_up.depthwise_conv.conv', 
                        pre_padding = (1, 1), post_padding = (1, 1))
    p4_up_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p4_up_depthwise_conv.get_output(0), 
                        num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                        lname = lname+'.conv4_up.pointwise_conv.conv')
    p4_up_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p4_up_pointwise_conv.get_output(0),
                         lname = lname+'.conv4_up.bn') 
    # 已验证
    # ----------------------- P3_0 and P4_1 to P3_2-----------------------------
    p3_w1_relu=p3_w1*(p3_w1>0)
    p3_w1_sum=np.sum(p3_w1_relu.numpy())+epsilon
    weight3=(p3_w1_relu/p3_w1_sum).numpy()
    weight0_p3_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p3_in_scale = np.full((160,),weight3[0])
    weight0_p3_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p3_in_w0=network.add_scale(p3_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p3_in_shift,scale=weight0_p3_in_scale)
    #p7_in_upsample = addUpsample(network, p7_in.get_output(0))

    p4_up_upsample = addUpsample(network, p4_up_bn.get_output(0), [1, 2, 2])

    weight1_p3p4up_in_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p3p4up_in_scale = np.full((160,),weight3[1])
    weight1_p3p4up_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p4_up_w=network.add_scale(p4_up_upsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p3p4up_in_shift,scale=weight1_p3p4up_in_scale)

    p3p4up_in_w = network.add_elementwise(p3_in_w0.get_output(0), p4_up_w.get_output(0), trt.ElementWiseOperation.SUM)
    p3p4up_swish = addSwish(network = network,inputTensor = p3p4up_in_w.get_output(0))

    p3_out_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p3p4up_swish.get_output(0),
                    num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv3_up.depthwise_conv.conv',
                    pre_padding = (1, 1), post_padding = (1, 1))
    p3_out_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p3_out_depthwise_conv.get_output(0), 
                    num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = lname+'.conv3_up.pointwise_conv.conv')
    p3_out_bn = addBatchNorm2d(network = network,
                    weights = weights, 
                    inputTensor = p3_out_pointwise_conv.get_output(0),
                    lname = lname+'.conv3_up.bn') 
    # 已验证
    # # ----------------------- P4_0, P4_1 and p3_2 to P4_2-----------------------------
    p4_w2_relu=p4_w2*(p4_w2>0)
    p4_w2_sum=np.sum(p4_w2_relu.numpy())+epsilon
    weight4_2=(p4_w2_relu/p4_w2_sum).numpy()
    weight0_p4_2_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p4_2_in_scale = np.full((160,),weight4_2[0])
    weight0_p4_2_in_scale.astype(np.float32)
    #weight_p4_2_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))

    p4_2_in_w0=network.add_scale(p4_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p4_2_in_shift,scale=weight0_p4_2_in_scale)
    weight0_p4_2_up_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p4_2_up_scale = np.full((160,),weight4_2[1])
    weight0_p4_2_up_scale.astype(np.float32)
    p4_2_up_w0=network.add_scale(p4_up_bn.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p4_2_up_shift,scale=weight0_p4_2_up_scale)
    
    p3_out_downsample = maxPoolingBlock(network, weights, p3_out_bn.get_output(0),3, 2,  pre_padding = (0, 0), post_padding = (1,1))

    
    weight1_p3_out_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p3_out_scale = np.full((160,),weight4_2[2])
    weight1_p3_out_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p3_out_down_w=network.add_scale(p3_out_downsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p3_out_shift,scale=weight1_p3_out_scale)
    
    p4_0_p4_2_sum = network.add_elementwise(p4_2_in_w0.get_output(0), p4_2_up_w0.get_output(0), trt.ElementWiseOperation.SUM)
    
    p4_0_p4_2_p3_2_sum = network.add_elementwise(p4_0_p4_2_sum.get_output(0), p3_out_down_w.get_output(0), trt.ElementWiseOperation.SUM)

    p4_0_p4_2_p3_2_swish = addSwish(network = network,inputTensor = p4_0_p4_2_p3_2_sum.get_output(0))

    
    p4_out_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p4_0_p4_2_p3_2_swish.get_output(0), 
                    num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv4_down.depthwise_conv.conv', 
                    pre_padding = (1, 1), post_padding = (1, 1))
    p4_out_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p4_out_depthwise_conv.get_output(0), 
                    num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                    lname = lname+'.conv4_down.pointwise_conv.conv')
    p4_out_bn = addBatchNorm2d(network = network,
                    weights = weights, 
                    inputTensor = p4_out_pointwise_conv.get_output(0),
                    lname = lname+'.conv4_down.bn') 
    # 已验证
    # -----------------------  Weights for P5_0, P5_1 and P4_2 to P5_2-----------------------------
    p5_w2_relu=p5_w2*(p5_w2>0)
    p5_w2_sum=np.sum(p5_w2_relu.numpy())+epsilon

    weight5_2=(p5_w2_relu/p5_w2_sum).numpy()
    weight0_p5_2_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p5_2_in_scale = np.full((160,),weight5_2[0])
    weight0_p5_2_in_scale.astype(np.float32)
    #weight_p4_2_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p5_in_w0=network.add_scale(p5_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p5_2_in_shift,scale=weight0_p5_2_in_scale)
    
    
    weight0_p5_2_up_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p5_2_up_scale = np.full((160,),weight5_2[1])
    weight0_p5_2_up_scale.astype(np.float32)
    p5_up_w1=network.add_scale(p5_up_bn.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p5_2_up_shift,scale=weight0_p5_2_up_scale)
    

    p4_out_downsample = maxPoolingBlock(network, weights, p4_out_bn.get_output(0),3, 2,  pre_padding = (0, 0), post_padding = (1,1))
    weight1_p4_out_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p4_out_scale = np.full((160,),weight5_2[2])
    weight1_p4_out_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p4_out_down_w=network.add_scale(p4_out_downsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p4_out_shift,scale=weight1_p4_out_scale)
    

    p5_0_p5_2_sum = network.add_elementwise(p5_in_w0.get_output(0), p5_up_w1.get_output(0), trt.ElementWiseOperation.SUM)
    
    p5_0_p5_2_p4_2_sum = network.add_elementwise(p5_0_p5_2_sum.get_output(0), p4_out_down_w.get_output(0), trt.ElementWiseOperation.SUM)

    p5_0_p5_2_p4_2_swish = addSwish(network = network,inputTensor = p5_0_p5_2_p4_2_sum.get_output(0))

    
    p5_out_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p5_0_p5_2_p4_2_swish.get_output(0), 
                    num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv5_down.depthwise_conv.conv', 
                    pre_padding = (1, 1), post_padding = (1, 1))
    p5_out_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p5_out_depthwise_conv.get_output(0), 
                    num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                    lname = lname+'.conv5_down.pointwise_conv.conv')
    p5_out_bn = addBatchNorm2d(network = network,
                    weights = weights, 
                    inputTensor = p5_out_pointwise_conv.get_output(0),
                    lname = lname+'.conv5_down.bn') 
    # -----------------------  #  Weights for P6_0, P6_1 and P5_2 to P6_2-----------------------------
    p6_w2_relu=p6_w2*(p6_w2>0)
    p6_w2_sum=np.sum(p6_w2_relu.numpy())+epsilon

    weight6_2=(p6_w2_relu/p6_w2_sum).numpy()
    weight0_p6_2_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p6_2_in_scale = np.full((160,),weight6_2[0])
    weight0_p6_2_in_scale.astype(np.float32)
    #weight_p4_2_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_in_w0=network.add_scale(p6_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p6_2_in_shift,scale=weight0_p6_2_in_scale)
    
    
    weight0_p6_2_up_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p6_2_up_scale = np.full((160,),weight6_2[1])
    weight0_p6_2_up_scale.astype(np.float32)
    p6_up_w1=network.add_scale(p6_up_bn.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p6_2_up_shift,scale=weight0_p6_2_up_scale)
    

    p5_out_downsample = maxPoolingBlock(network, weights, p5_out_bn.get_output(0),3, 2,  pre_padding = (0, 0), post_padding = (1,1))
    weight1_p5_out_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p5_out_scale = np.full((160,),weight6_2[2])
    weight1_p5_out_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p5_out_down_w=network.add_scale(p5_out_downsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p5_out_shift,scale=weight1_p5_out_scale)
    

    p6_0_p6_2_sum = network.add_elementwise(p6_in_w0.get_output(0), p6_up_w1.get_output(0), trt.ElementWiseOperation.SUM)
    
    p6_0_p6_2_p5_2_sum = network.add_elementwise(p6_0_p6_2_sum.get_output(0), p5_out_down_w.get_output(0), trt.ElementWiseOperation.SUM)

    p6_0_p6_2_p5_2_swish = addSwish(network = network,inputTensor = p6_0_p6_2_p5_2_sum.get_output(0))

    
    p6_out_depthwise_conv = convBlock(network = network, weights = weights, 
                            inputTensor = p6_0_p6_2_p5_2_swish.get_output(0), 
                            num_output_maps = 160, ksize = 3, stride = 1, group = 160,
                            lname = lname+'.conv6_down.depthwise_conv.conv',
                            pre_padding = (1, 1), post_padding = (1, 1))
    p6_out_pointwise_conv = convBlock(network = network, weights = weights, 
                            inputTensor = p6_out_depthwise_conv.get_output(0),
                            num_output_maps = 160, ksize = 1, stride = 1, group = 1,
                            lname = lname+'.conv6_down.pointwise_conv.conv')
    p6_out_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p6_out_pointwise_conv.get_output(0),
                         lname = lname+'.conv6_down.bn') 
    # -----------------------  # # Weights for P7_0 and P6_2 to P7_2-----------------------------
    p7_w2_relu=p7_w2*(p7_w2>0)
    p7_w2_sum=np.sum(p7_w2_relu.numpy())+epsilon

    weight7_2=(p7_w2_relu/p7_w2_sum).numpy()
    weight0_p7_2_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p7_2_in_scale = np.full((160,),weight7_2[0])
    weight0_p7_2_in_scale.astype(np.float32)
    #weight_p4_2_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p7_in_w0=network.add_scale(p7_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p7_2_in_shift,scale=weight0_p7_2_in_scale)
   
    p6_out_downsample = maxPoolingBlock(network, weights, p6_out_bn.get_output(0),3, 2,  pre_padding = (0, 0), post_padding = (1,1))
    weight1_p6_out_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p6_out_scale = np.full((160,),weight7_2[1])
    weight1_p6_out_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_out_down_w=network.add_scale(p6_out_downsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p6_out_shift,scale=weight1_p6_out_scale)
        
    p7_0_p6_1_sum = network.add_elementwise(p7_in_w0.get_output(0), p6_out_down_w.get_output(0), trt.ElementWiseOperation.SUM)

    p7_0_p6_1_swish = addSwish(network = network,inputTensor = p7_0_p6_1_sum.get_output(0))

    p7_out_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p7_0_p6_1_swish.get_output(0), 
                    num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv7_down.depthwise_conv.conv', 
                    pre_padding = (1, 1), post_padding = (1, 1))
    p7_out_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p7_out_depthwise_conv.get_output(0), 
                    num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                    lname = lname+'.conv7_down.pointwise_conv.conv')
    p7_out_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p7_out_pointwise_conv.get_output(0),
                         lname = lname+'.conv7_down.bn') 
# --------------------------------------------bifpn2--------------------------------
    lname="bifpn.2"
    p3_in=p3_out_bn
    p4_in=p4_out_bn
    p5_in=p5_out_bn
    p6_in=p6_out_bn
    p7_in=p7_out_bn

    p6_w1=weights[lname+".p6_w1"]
    p5_w1=weights[lname+".p5_w1"]
    p4_w1=weights[lname+".p4_w1"]
    p3_w1=weights[lname+".p3_w1"]

    p4_w2=weights[lname+".p4_w2"]
    p5_w2=weights[lname+".p5_w2"]
    p6_w2=weights[lname+".p6_w2"]
    p7_w2=weights[lname+".p7_w2"]
    
    epsilon=1e-4
    # ----------------------- P6_0 and P7_0 to P6_1-----------------------------
    p6_w1_relu=p6_w1*(p6_w1>0)
    p6_w1_sum=np.sum(p6_w1_relu.numpy())+epsilon
    weight6=(p6_w1_relu/p6_w1_sum).numpy()
    weight0_p6_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p6_in_scale = np.full((160,),weight6[0])
    weight0_p6_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_in_w0=network.add_scale(p6_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p6_in_shift,scale=weight0_p6_in_scale)
    #p7_in_upsample = addUpsample(network, p7_in.get_output(0))

    p6_upsample = addUpsample(network, p7_in.get_output(0), [1, 2, 2])

    weight1_p6p7_in_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p6p7_in_scale = np.full((160,),weight6[1])
    weight1_p6p7_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_in_w1=network.add_scale(p6_upsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p6p7_in_shift,scale=weight1_p6p7_in_scale)

    p6_in_w = network.add_elementwise(p6_in_w0.get_output(0), p6_in_w1.get_output(0), trt.ElementWiseOperation.SUM)
    p6_up_swish = addSwish(network = network,inputTensor = p6_in_w.get_output(0))

    p6_up_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p6_up_swish.get_output(0), 
                        num_output_maps = 160, ksize = 3, stride = 1, group = 160, 
                        lname = lname+'.conv6_up.depthwise_conv.conv', 
                        pre_padding = (1, 1), post_padding = (1, 1))
    p6_up_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p6_up_depthwise_conv.get_output(0), 
                        num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                        lname = lname+'.conv6_up.pointwise_conv.conv')
    p6_up_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p6_up_pointwise_conv.get_output(0),
                         lname = lname+'.conv6_up.bn')
    # 已验证
    # ----------------------- P5_0 and P6_1 to P5_1-----------------------------
    p5_w1_relu=p5_w1*(p5_w1>0)
    p5_w1_sum=np.sum(p5_w1_relu.numpy())+epsilon
    weight5=(p5_w1_relu/p5_w1_sum).numpy()
    weight0_p5_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p5_in_scale = np.full((160,),weight5[0])
    weight0_p5_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p5_in_w0=network.add_scale(p5_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p5_in_shift,scale=weight0_p5_in_scale)
    #p7_in_upsample = addUpsample(network, p7_in.get_output(0))

    p6_up_upsample = addUpsample(network, p6_up_bn.get_output(0), [1, 2, 2])

    weight1_p5p6up_in_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p5p6up_in_scale = np.full((160,),weight5[1])
    weight1_p5p6up_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_up_w=network.add_scale(p6_up_upsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p5p6up_in_shift,scale=weight1_p5p6up_in_scale)

    p5p6up_in_w = network.add_elementwise(p5_in_w0.get_output(0), p6_up_w.get_output(0), trt.ElementWiseOperation.SUM)
    p5p6up_swish = addSwish(network = network,inputTensor = p5p6up_in_w.get_output(0))

    p5_up_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p5p6up_swish.get_output(0), 
                num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv5_up.depthwise_conv.conv', 
                pre_padding = (1, 1), post_padding = (1, 1))
    p5_up_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p5_up_depthwise_conv.get_output(0), 
                num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                lname = lname+'.conv5_up.pointwise_conv.conv')
    p5_up_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p5_up_pointwise_conv.get_output(0),
                         lname = lname+'.conv5_up.bn')    
    # 已验证
    # ----------------------- P4_0 and P5_1 to P4_1-----------------------------
    p4_w1_relu=p4_w1*(p4_w1>0)
    p4_w1_sum=np.sum(p4_w1_relu.numpy())+epsilon
    weight4=(p4_w1_relu/p4_w1_sum).numpy()
    weight0_p4_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p4_in_scale = np.full((160,),weight4[0])
    weight0_p4_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p4_in_w0=network.add_scale(p4_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p4_in_shift,scale=weight0_p4_in_scale)
    #p7_in_upsample = addUpsample(network, p7_in.get_output(0))

    p5_up_upsample = addUpsample(network, p5_up_bn.get_output(0), [1, 2, 2])

    weight1_p4p5up_in_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p4p5up_in_scale = np.full((160,),weight4[1])
    weight1_p4p5up_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p5_up_w=network.add_scale(p5_up_upsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p4p5up_in_shift,scale=weight1_p4p5up_in_scale)

    p4p5up_in_w = network.add_elementwise(p4_in_w0.get_output(0), p5_up_w.get_output(0), trt.ElementWiseOperation.SUM)
    p4p5up_swish = addSwish(network = network,inputTensor = p4p5up_in_w.get_output(0))

    p4_up_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p4p5up_swish.get_output(0), 
                        num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv4_up.depthwise_conv.conv', 
                        pre_padding = (1, 1), post_padding = (1, 1))
    p4_up_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p4_up_depthwise_conv.get_output(0), 
                        num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                        lname = lname+'.conv4_up.pointwise_conv.conv')
    p4_up_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p4_up_pointwise_conv.get_output(0),
                         lname = lname+'.conv4_up.bn') 
    # 已验证
    # ----------------------- P3_0 and P4_1 to P3_2-----------------------------
    p3_w1_relu=p3_w1*(p3_w1>0)
    p3_w1_sum=np.sum(p3_w1_relu.numpy())+epsilon
    weight3=(p3_w1_relu/p3_w1_sum).numpy()
    weight0_p3_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p3_in_scale = np.full((160,),weight3[0])
    weight0_p3_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p3_in_w0=network.add_scale(p3_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p3_in_shift,scale=weight0_p3_in_scale)
    #p7_in_upsample = addUpsample(network, p7_in.get_output(0))

    p4_up_upsample = addUpsample(network, p4_up_bn.get_output(0), [1, 2, 2])

    weight1_p3p4up_in_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p3p4up_in_scale = np.full((160,),weight3[1])
    weight1_p3p4up_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p4_up_w=network.add_scale(p4_up_upsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p3p4up_in_shift,scale=weight1_p3p4up_in_scale)

    p3p4up_in_w = network.add_elementwise(p3_in_w0.get_output(0), p4_up_w.get_output(0), trt.ElementWiseOperation.SUM)
    p3p4up_swish = addSwish(network = network,inputTensor = p3p4up_in_w.get_output(0))

    p3_out_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p3p4up_swish.get_output(0),
                    num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv3_up.depthwise_conv.conv',
                    pre_padding = (1, 1), post_padding = (1, 1))
    p3_out_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p3_out_depthwise_conv.get_output(0), 
                    num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = lname+'.conv3_up.pointwise_conv.conv')
    p3_out_bn = addBatchNorm2d(network = network,
                    weights = weights, 
                    inputTensor = p3_out_pointwise_conv.get_output(0),
                    lname = lname+'.conv3_up.bn') 
    # 已验证
    # # ----------------------- P4_0, P4_1 and p3_2 to P4_2-----------------------------
    p4_w2_relu=p4_w2*(p4_w2>0)
    p4_w2_sum=np.sum(p4_w2_relu.numpy())+epsilon
    weight4_2=(p4_w2_relu/p4_w2_sum).numpy()
    weight0_p4_2_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p4_2_in_scale = np.full((160,),weight4_2[0])
    weight0_p4_2_in_scale.astype(np.float32)
    #weight_p4_2_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))

    p4_2_in_w0=network.add_scale(p4_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p4_2_in_shift,scale=weight0_p4_2_in_scale)
    weight0_p4_2_up_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p4_2_up_scale = np.full((160,),weight4_2[1])
    weight0_p4_2_up_scale.astype(np.float32)
    p4_2_up_w0=network.add_scale(p4_up_bn.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p4_2_up_shift,scale=weight0_p4_2_up_scale)
    
    p3_out_downsample = maxPoolingBlock(network, weights, p3_out_bn.get_output(0),3, 2,  pre_padding = (0, 0), post_padding = (1,1))

    
    weight1_p3_out_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p3_out_scale = np.full((160,),weight4_2[2])
    weight1_p3_out_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p3_out_down_w=network.add_scale(p3_out_downsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p3_out_shift,scale=weight1_p3_out_scale)
    
    p4_0_p4_2_sum = network.add_elementwise(p4_2_in_w0.get_output(0), p4_2_up_w0.get_output(0), trt.ElementWiseOperation.SUM)
    
    p4_0_p4_2_p3_2_sum = network.add_elementwise(p4_0_p4_2_sum.get_output(0), p3_out_down_w.get_output(0), trt.ElementWiseOperation.SUM)

    p4_0_p4_2_p3_2_swish = addSwish(network = network,inputTensor = p4_0_p4_2_p3_2_sum.get_output(0))

    
    p4_out_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p4_0_p4_2_p3_2_swish.get_output(0), 
                    num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv4_down.depthwise_conv.conv', 
                    pre_padding = (1, 1), post_padding = (1, 1))
    p4_out_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p4_out_depthwise_conv.get_output(0), 
                    num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                    lname = lname+'.conv4_down.pointwise_conv.conv')
    p4_out_bn = addBatchNorm2d(network = network,
                    weights = weights, 
                    inputTensor = p4_out_pointwise_conv.get_output(0),
                    lname = lname+'.conv4_down.bn') 
    # 已验证
    # -----------------------  Weights for P5_0, P5_1 and P4_2 to P5_2-----------------------------
    p5_w2_relu=p5_w2*(p5_w2>0)
    p5_w2_sum=np.sum(p5_w2_relu.numpy())+epsilon

    weight5_2=(p5_w2_relu/p5_w2_sum).numpy()
    weight0_p5_2_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p5_2_in_scale = np.full((160,),weight5_2[0])
    weight0_p5_2_in_scale.astype(np.float32)
    #weight_p4_2_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p5_in_w0=network.add_scale(p5_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p5_2_in_shift,scale=weight0_p5_2_in_scale)
    
    
    weight0_p5_2_up_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p5_2_up_scale = np.full((160,),weight5_2[1])
    weight0_p5_2_up_scale.astype(np.float32)
    p5_up_w1=network.add_scale(p5_up_bn.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p5_2_up_shift,scale=weight0_p5_2_up_scale)
    

    p4_out_downsample = maxPoolingBlock(network, weights, p4_out_bn.get_output(0),3, 2,  pre_padding = (0, 0), post_padding = (1,1))
    weight1_p4_out_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p4_out_scale = np.full((160,),weight5_2[2])
    weight1_p4_out_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p4_out_down_w=network.add_scale(p4_out_downsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p4_out_shift,scale=weight1_p4_out_scale)
    

    p5_0_p5_2_sum = network.add_elementwise(p5_in_w0.get_output(0), p5_up_w1.get_output(0), trt.ElementWiseOperation.SUM)
    
    p5_0_p5_2_p4_2_sum = network.add_elementwise(p5_0_p5_2_sum.get_output(0), p4_out_down_w.get_output(0), trt.ElementWiseOperation.SUM)

    p5_0_p5_2_p4_2_swish = addSwish(network = network,inputTensor = p5_0_p5_2_p4_2_sum.get_output(0))

    
    p5_out_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p5_0_p5_2_p4_2_swish.get_output(0), 
                    num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv5_down.depthwise_conv.conv', 
                    pre_padding = (1, 1), post_padding = (1, 1))
    p5_out_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p5_out_depthwise_conv.get_output(0), 
                    num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                    lname = lname+'.conv5_down.pointwise_conv.conv')
    p5_out_bn = addBatchNorm2d(network = network,
                    weights = weights, 
                    inputTensor = p5_out_pointwise_conv.get_output(0),
                    lname = lname+'.conv5_down.bn') 
    # -----------------------  #  Weights for P6_0, P6_1 and P5_2 to P6_2-----------------------------
    p6_w2_relu=p6_w2*(p6_w2>0)
    p6_w2_sum=np.sum(p6_w2_relu.numpy())+epsilon

    weight6_2=(p6_w2_relu/p6_w2_sum).numpy()
    weight0_p6_2_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p6_2_in_scale = np.full((160,),weight6_2[0])
    weight0_p6_2_in_scale.astype(np.float32)
    #weight_p4_2_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_in_w0=network.add_scale(p6_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p6_2_in_shift,scale=weight0_p6_2_in_scale)
    
    
    weight0_p6_2_up_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p6_2_up_scale = np.full((160,),weight6_2[1])
    weight0_p6_2_up_scale.astype(np.float32)
    p6_up_w1=network.add_scale(p6_up_bn.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p6_2_up_shift,scale=weight0_p6_2_up_scale)
    

    p5_out_downsample = maxPoolingBlock(network, weights, p5_out_bn.get_output(0),3, 2,  pre_padding = (0, 0), post_padding = (1,1))
    weight1_p5_out_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p5_out_scale = np.full((160,),weight6_2[2])
    weight1_p5_out_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p5_out_down_w=network.add_scale(p5_out_downsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p5_out_shift,scale=weight1_p5_out_scale)
    

    p6_0_p6_2_sum = network.add_elementwise(p6_in_w0.get_output(0), p6_up_w1.get_output(0), trt.ElementWiseOperation.SUM)
    
    p6_0_p6_2_p5_2_sum = network.add_elementwise(p6_0_p6_2_sum.get_output(0), p5_out_down_w.get_output(0), trt.ElementWiseOperation.SUM)

    p6_0_p6_2_p5_2_swish = addSwish(network = network,inputTensor = p6_0_p6_2_p5_2_sum.get_output(0))

    
    p6_out_depthwise_conv = convBlock(network = network, weights = weights, 
                            inputTensor = p6_0_p6_2_p5_2_swish.get_output(0), 
                            num_output_maps = 160, ksize = 3, stride = 1, group = 160,
                            lname = lname+'.conv6_down.depthwise_conv.conv',
                            pre_padding = (1, 1), post_padding = (1, 1))
    p6_out_pointwise_conv = convBlock(network = network, weights = weights, 
                            inputTensor = p6_out_depthwise_conv.get_output(0),
                            num_output_maps = 160, ksize = 1, stride = 1, group = 1,
                            lname = lname+'.conv6_down.pointwise_conv.conv')
    p6_out_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p6_out_pointwise_conv.get_output(0),
                         lname = lname+'.conv6_down.bn') 
    # -----------------------  # # Weights for P7_0 and P6_2 to P7_2-----------------------------
    p7_w2_relu=p7_w2*(p7_w2>0)
    p7_w2_sum=np.sum(p7_w2_relu.numpy())+epsilon

    weight7_2=(p7_w2_relu/p7_w2_sum).numpy()
    weight0_p7_2_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p7_2_in_scale = np.full((160,),weight7_2[0])
    weight0_p7_2_in_scale.astype(np.float32)
    #weight_p4_2_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p7_in_w0=network.add_scale(p7_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p7_2_in_shift,scale=weight0_p7_2_in_scale)
   
    p6_out_downsample = maxPoolingBlock(network, weights, p6_out_bn.get_output(0),3, 2,  pre_padding = (0, 0), post_padding = (1,1))
    weight1_p6_out_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p6_out_scale = np.full((160,),weight7_2[1])
    weight1_p6_out_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_out_down_w=network.add_scale(p6_out_downsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p6_out_shift,scale=weight1_p6_out_scale)
        
    p7_0_p6_1_sum = network.add_elementwise(p7_in_w0.get_output(0), p6_out_down_w.get_output(0), trt.ElementWiseOperation.SUM)

    p7_0_p6_1_swish = addSwish(network = network,inputTensor = p7_0_p6_1_sum.get_output(0))

    p7_out_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p7_0_p6_1_swish.get_output(0), 
                    num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv7_down.depthwise_conv.conv', 
                    pre_padding = (1, 1), post_padding = (1, 1))
    p7_out_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p7_out_depthwise_conv.get_output(0), 
                    num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                    lname = lname+'.conv7_down.pointwise_conv.conv')
    p7_out_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p7_out_pointwise_conv.get_output(0),
                         lname = lname+'.conv7_down.bn') 


# --------------------------------------------bifpn3--------------------------------
    lname="bifpn.3"
    p3_in=p3_out_bn
    p4_in=p4_out_bn
    p5_in=p5_out_bn
    p6_in=p6_out_bn
    p7_in=p7_out_bn

    p6_w1=weights[lname+".p6_w1"]
    p5_w1=weights[lname+".p5_w1"]
    p4_w1=weights[lname+".p4_w1"]
    p3_w1=weights[lname+".p3_w1"]

    p4_w2=weights[lname+".p4_w2"]
    p5_w2=weights[lname+".p5_w2"]
    p6_w2=weights[lname+".p6_w2"]
    p7_w2=weights[lname+".p7_w2"]
    
    epsilon=1e-4
    # ----------------------- P6_0 and P7_0 to P6_1-----------------------------
    p6_w1_relu=p6_w1*(p6_w1>0)
    p6_w1_sum=np.sum(p6_w1_relu.numpy())+epsilon
    weight6=(p6_w1_relu/p6_w1_sum).numpy()
    weight0_p6_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p6_in_scale = np.full((160,),weight6[0])
    weight0_p6_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_in_w0=network.add_scale(p6_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p6_in_shift,scale=weight0_p6_in_scale)
    #p7_in_upsample = addUpsample(network, p7_in.get_output(0))

    p6_upsample = addUpsample(network, p7_in.get_output(0), [1, 2, 2])

    weight1_p6p7_in_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p6p7_in_scale = np.full((160,),weight6[1])
    weight1_p6p7_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_in_w1=network.add_scale(p6_upsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p6p7_in_shift,scale=weight1_p6p7_in_scale)

    p6_in_w = network.add_elementwise(p6_in_w0.get_output(0), p6_in_w1.get_output(0), trt.ElementWiseOperation.SUM)
    p6_up_swish = addSwish(network = network,inputTensor = p6_in_w.get_output(0))

    p6_up_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p6_up_swish.get_output(0), 
                        num_output_maps = 160, ksize = 3, stride = 1, group = 160, 
                        lname = lname+'.conv6_up.depthwise_conv.conv', 
                        pre_padding = (1, 1), post_padding = (1, 1))
    p6_up_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p6_up_depthwise_conv.get_output(0), 
                        num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                        lname = lname+'.conv6_up.pointwise_conv.conv')
    p6_up_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p6_up_pointwise_conv.get_output(0),
                         lname = lname+'.conv6_up.bn')
    # 已验证
    # ----------------------- P5_0 and P6_1 to P5_1-----------------------------
    p5_w1_relu=p5_w1*(p5_w1>0)
    p5_w1_sum=np.sum(p5_w1_relu.numpy())+epsilon
    weight5=(p5_w1_relu/p5_w1_sum).numpy()
    weight0_p5_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p5_in_scale = np.full((160,),weight5[0])
    weight0_p5_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p5_in_w0=network.add_scale(p5_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p5_in_shift,scale=weight0_p5_in_scale)
    #p7_in_upsample = addUpsample(network, p7_in.get_output(0))

    p6_up_upsample = addUpsample(network, p6_up_bn.get_output(0), [1, 2, 2])

    weight1_p5p6up_in_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p5p6up_in_scale = np.full((160,),weight5[1])
    weight1_p5p6up_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_up_w=network.add_scale(p6_up_upsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p5p6up_in_shift,scale=weight1_p5p6up_in_scale)

    p5p6up_in_w = network.add_elementwise(p5_in_w0.get_output(0), p6_up_w.get_output(0), trt.ElementWiseOperation.SUM)
    p5p6up_swish = addSwish(network = network,inputTensor = p5p6up_in_w.get_output(0))

    p5_up_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p5p6up_swish.get_output(0), 
                num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv5_up.depthwise_conv.conv', 
                pre_padding = (1, 1), post_padding = (1, 1))
    p5_up_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p5_up_depthwise_conv.get_output(0), 
                num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                lname = lname+'.conv5_up.pointwise_conv.conv')
    p5_up_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p5_up_pointwise_conv.get_output(0),
                         lname = lname+'.conv5_up.bn')    
    # 已验证
    # ----------------------- P4_0 and P5_1 to P4_1-----------------------------
    p4_w1_relu=p4_w1*(p4_w1>0)
    p4_w1_sum=np.sum(p4_w1_relu.numpy())+epsilon
    weight4=(p4_w1_relu/p4_w1_sum).numpy()
    weight0_p4_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p4_in_scale = np.full((160,),weight4[0])
    weight0_p4_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p4_in_w0=network.add_scale(p4_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p4_in_shift,scale=weight0_p4_in_scale)
    #p7_in_upsample = addUpsample(network, p7_in.get_output(0))

    p5_up_upsample = addUpsample(network, p5_up_bn.get_output(0), [1, 2, 2])

    weight1_p4p5up_in_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p4p5up_in_scale = np.full((160,),weight4[1])
    weight1_p4p5up_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p5_up_w=network.add_scale(p5_up_upsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p4p5up_in_shift,scale=weight1_p4p5up_in_scale)

    p4p5up_in_w = network.add_elementwise(p4_in_w0.get_output(0), p5_up_w.get_output(0), trt.ElementWiseOperation.SUM)
    p4p5up_swish = addSwish(network = network,inputTensor = p4p5up_in_w.get_output(0))

    p4_up_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p4p5up_swish.get_output(0), 
                        num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv4_up.depthwise_conv.conv', 
                        pre_padding = (1, 1), post_padding = (1, 1))
    p4_up_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p4_up_depthwise_conv.get_output(0), 
                        num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                        lname = lname+'.conv4_up.pointwise_conv.conv')
    p4_up_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p4_up_pointwise_conv.get_output(0),
                         lname = lname+'.conv4_up.bn') 
    # 已验证
    # ----------------------- P3_0 and P4_1 to P3_2-----------------------------
    p3_w1_relu=p3_w1*(p3_w1>0)
    p3_w1_sum=np.sum(p3_w1_relu.numpy())+epsilon
    weight3=(p3_w1_relu/p3_w1_sum).numpy()
    weight0_p3_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p3_in_scale = np.full((160,),weight3[0])
    weight0_p3_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p3_in_w0=network.add_scale(p3_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p3_in_shift,scale=weight0_p3_in_scale)
    #p7_in_upsample = addUpsample(network, p7_in.get_output(0))

    p4_up_upsample = addUpsample(network, p4_up_bn.get_output(0), [1, 2, 2])

    weight1_p3p4up_in_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p3p4up_in_scale = np.full((160,),weight3[1])
    weight1_p3p4up_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p4_up_w=network.add_scale(p4_up_upsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p3p4up_in_shift,scale=weight1_p3p4up_in_scale)

    p3p4up_in_w = network.add_elementwise(p3_in_w0.get_output(0), p4_up_w.get_output(0), trt.ElementWiseOperation.SUM)
    p3p4up_swish = addSwish(network = network,inputTensor = p3p4up_in_w.get_output(0))

    p3_out_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p3p4up_swish.get_output(0),
                    num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv3_up.depthwise_conv.conv',
                    pre_padding = (1, 1), post_padding = (1, 1))
    p3_out_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p3_out_depthwise_conv.get_output(0), 
                    num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = lname+'.conv3_up.pointwise_conv.conv')
    p3_out_bn = addBatchNorm2d(network = network,
                    weights = weights, 
                    inputTensor = p3_out_pointwise_conv.get_output(0),
                    lname = lname+'.conv3_up.bn') 
    # 已验证
    # # ----------------------- P4_0, P4_1 and p3_2 to P4_2-----------------------------
    p4_w2_relu=p4_w2*(p4_w2>0)
    p4_w2_sum=np.sum(p4_w2_relu.numpy())+epsilon
    weight4_2=(p4_w2_relu/p4_w2_sum).numpy()
    weight0_p4_2_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p4_2_in_scale = np.full((160,),weight4_2[0])
    weight0_p4_2_in_scale.astype(np.float32)
    #weight_p4_2_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))

    p4_2_in_w0=network.add_scale(p4_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p4_2_in_shift,scale=weight0_p4_2_in_scale)
    weight0_p4_2_up_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p4_2_up_scale = np.full((160,),weight4_2[1])
    weight0_p4_2_up_scale.astype(np.float32)
    p4_2_up_w0=network.add_scale(p4_up_bn.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p4_2_up_shift,scale=weight0_p4_2_up_scale)
    
    p3_out_downsample = maxPoolingBlock(network, weights, p3_out_bn.get_output(0),3, 2,  pre_padding = (0, 0), post_padding = (1,1))

    
    weight1_p3_out_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p3_out_scale = np.full((160,),weight4_2[2])
    weight1_p3_out_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p3_out_down_w=network.add_scale(p3_out_downsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p3_out_shift,scale=weight1_p3_out_scale)
    
    p4_0_p4_2_sum = network.add_elementwise(p4_2_in_w0.get_output(0), p4_2_up_w0.get_output(0), trt.ElementWiseOperation.SUM)
    
    p4_0_p4_2_p3_2_sum = network.add_elementwise(p4_0_p4_2_sum.get_output(0), p3_out_down_w.get_output(0), trt.ElementWiseOperation.SUM)

    p4_0_p4_2_p3_2_swish = addSwish(network = network,inputTensor = p4_0_p4_2_p3_2_sum.get_output(0))

    
    p4_out_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p4_0_p4_2_p3_2_swish.get_output(0), 
                    num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv4_down.depthwise_conv.conv', 
                    pre_padding = (1, 1), post_padding = (1, 1))
    p4_out_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p4_out_depthwise_conv.get_output(0), 
                    num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                    lname = lname+'.conv4_down.pointwise_conv.conv')
    p4_out_bn = addBatchNorm2d(network = network,
                    weights = weights, 
                    inputTensor = p4_out_pointwise_conv.get_output(0),
                    lname = lname+'.conv4_down.bn') 
    # 已验证
    # -----------------------  Weights for P5_0, P5_1 and P4_2 to P5_2-----------------------------
    p5_w2_relu=p5_w2*(p5_w2>0)
    p5_w2_sum=np.sum(p5_w2_relu.numpy())+epsilon

    weight5_2=(p5_w2_relu/p5_w2_sum).numpy()
    weight0_p5_2_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p5_2_in_scale = np.full((160,),weight5_2[0])
    weight0_p5_2_in_scale.astype(np.float32)
    #weight_p4_2_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p5_in_w0=network.add_scale(p5_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p5_2_in_shift,scale=weight0_p5_2_in_scale)
    
    
    weight0_p5_2_up_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p5_2_up_scale = np.full((160,),weight5_2[1])
    weight0_p5_2_up_scale.astype(np.float32)
    p5_up_w1=network.add_scale(p5_up_bn.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p5_2_up_shift,scale=weight0_p5_2_up_scale)
    

    p4_out_downsample = maxPoolingBlock(network, weights, p4_out_bn.get_output(0),3, 2,  pre_padding = (0, 0), post_padding = (1,1))
    weight1_p4_out_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p4_out_scale = np.full((160,),weight5_2[2])
    weight1_p4_out_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p4_out_down_w=network.add_scale(p4_out_downsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p4_out_shift,scale=weight1_p4_out_scale)
    

    p5_0_p5_2_sum = network.add_elementwise(p5_in_w0.get_output(0), p5_up_w1.get_output(0), trt.ElementWiseOperation.SUM)
    
    p5_0_p5_2_p4_2_sum = network.add_elementwise(p5_0_p5_2_sum.get_output(0), p4_out_down_w.get_output(0), trt.ElementWiseOperation.SUM)

    p5_0_p5_2_p4_2_swish = addSwish(network = network,inputTensor = p5_0_p5_2_p4_2_sum.get_output(0))

    
    p5_out_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p5_0_p5_2_p4_2_swish.get_output(0), 
                    num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv5_down.depthwise_conv.conv', 
                    pre_padding = (1, 1), post_padding = (1, 1))
    p5_out_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p5_out_depthwise_conv.get_output(0), 
                    num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                    lname = lname+'.conv5_down.pointwise_conv.conv')
    p5_out_bn = addBatchNorm2d(network = network,
                    weights = weights, 
                    inputTensor = p5_out_pointwise_conv.get_output(0),
                    lname = lname+'.conv5_down.bn') 
    # -----------------------  #  Weights for P6_0, P6_1 and P5_2 to P6_2-----------------------------
    p6_w2_relu=p6_w2*(p6_w2>0)
    p6_w2_sum=np.sum(p6_w2_relu.numpy())+epsilon

    weight6_2=(p6_w2_relu/p6_w2_sum).numpy()
    weight0_p6_2_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p6_2_in_scale = np.full((160,),weight6_2[0])
    weight0_p6_2_in_scale.astype(np.float32)
    #weight_p4_2_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_in_w0=network.add_scale(p6_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p6_2_in_shift,scale=weight0_p6_2_in_scale)
    
    
    weight0_p6_2_up_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p6_2_up_scale = np.full((160,),weight6_2[1])
    weight0_p6_2_up_scale.astype(np.float32)
    p6_up_w1=network.add_scale(p6_up_bn.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p6_2_up_shift,scale=weight0_p6_2_up_scale)
    

    p5_out_downsample = maxPoolingBlock(network, weights, p5_out_bn.get_output(0),3, 2,  pre_padding = (0, 0), post_padding = (1,1))
    weight1_p5_out_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p5_out_scale = np.full((160,),weight6_2[2])
    weight1_p5_out_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p5_out_down_w=network.add_scale(p5_out_downsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p5_out_shift,scale=weight1_p5_out_scale)
    

    p6_0_p6_2_sum = network.add_elementwise(p6_in_w0.get_output(0), p6_up_w1.get_output(0), trt.ElementWiseOperation.SUM)
    
    p6_0_p6_2_p5_2_sum = network.add_elementwise(p6_0_p6_2_sum.get_output(0), p5_out_down_w.get_output(0), trt.ElementWiseOperation.SUM)

    p6_0_p6_2_p5_2_swish = addSwish(network = network,inputTensor = p6_0_p6_2_p5_2_sum.get_output(0))

    
    p6_out_depthwise_conv = convBlock(network = network, weights = weights, 
                            inputTensor = p6_0_p6_2_p5_2_swish.get_output(0), 
                            num_output_maps = 160, ksize = 3, stride = 1, group = 160,
                            lname = lname+'.conv6_down.depthwise_conv.conv',
                            pre_padding = (1, 1), post_padding = (1, 1))
    p6_out_pointwise_conv = convBlock(network = network, weights = weights, 
                            inputTensor = p6_out_depthwise_conv.get_output(0),
                            num_output_maps = 160, ksize = 1, stride = 1, group = 1,
                            lname = lname+'.conv6_down.pointwise_conv.conv')
    p6_out_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p6_out_pointwise_conv.get_output(0),
                         lname = lname+'.conv6_down.bn') 
    # -----------------------  # # Weights for P7_0 and P6_2 to P7_2-----------------------------
    p7_w2_relu=p7_w2*(p7_w2>0)
    p7_w2_sum=np.sum(p7_w2_relu.numpy())+epsilon

    weight7_2=(p7_w2_relu/p7_w2_sum).numpy()
    weight0_p7_2_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p7_2_in_scale = np.full((160,),weight7_2[0])
    weight0_p7_2_in_scale.astype(np.float32)
    #weight_p4_2_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p7_in_w0=network.add_scale(p7_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p7_2_in_shift,scale=weight0_p7_2_in_scale)
   
    p6_out_downsample = maxPoolingBlock(network, weights, p6_out_bn.get_output(0),3, 2,  pre_padding = (0, 0), post_padding = (1,1))
    weight1_p6_out_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p6_out_scale = np.full((160,),weight7_2[1])
    weight1_p6_out_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_out_down_w=network.add_scale(p6_out_downsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p6_out_shift,scale=weight1_p6_out_scale)
        
    p7_0_p6_1_sum = network.add_elementwise(p7_in_w0.get_output(0), p6_out_down_w.get_output(0), trt.ElementWiseOperation.SUM)

    p7_0_p6_1_swish = addSwish(network = network,inputTensor = p7_0_p6_1_sum.get_output(0))

    p7_out_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p7_0_p6_1_swish.get_output(0), 
                    num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv7_down.depthwise_conv.conv', 
                    pre_padding = (1, 1), post_padding = (1, 1))
    p7_out_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p7_out_depthwise_conv.get_output(0), 
                    num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                    lname = lname+'.conv7_down.pointwise_conv.conv')
    p7_out_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p7_out_pointwise_conv.get_output(0),
                         lname = lname+'.conv7_down.bn') 
# --------------------------------------------bifpn4--------------------------------
    lname="bifpn.4"
    p3_in=p3_out_bn
    p4_in=p4_out_bn
    p5_in=p5_out_bn
    p6_in=p6_out_bn
    p7_in=p7_out_bn

    p6_w1=weights[lname+".p6_w1"]
    p5_w1=weights[lname+".p5_w1"]
    p4_w1=weights[lname+".p4_w1"]
    p3_w1=weights[lname+".p3_w1"]

    p4_w2=weights[lname+".p4_w2"]
    p5_w2=weights[lname+".p5_w2"]
    p6_w2=weights[lname+".p6_w2"]
    p7_w2=weights[lname+".p7_w2"]
    
    epsilon=1e-4
    # ----------------------- P6_0 and P7_0 to P6_1-----------------------------
    p6_w1_relu=p6_w1*(p6_w1>0)
    p6_w1_sum=np.sum(p6_w1_relu.numpy())+epsilon
    weight6=(p6_w1_relu/p6_w1_sum).numpy()
    weight0_p6_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p6_in_scale = np.full((160,),weight6[0])
    weight0_p6_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_in_w0=network.add_scale(p6_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p6_in_shift,scale=weight0_p6_in_scale)
    #p7_in_upsample = addUpsample(network, p7_in.get_output(0))

    p6_upsample = addUpsample(network, p7_in.get_output(0), [1, 2, 2])

    weight1_p6p7_in_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p6p7_in_scale = np.full((160,),weight6[1])
    weight1_p6p7_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_in_w1=network.add_scale(p6_upsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p6p7_in_shift,scale=weight1_p6p7_in_scale)

    p6_in_w = network.add_elementwise(p6_in_w0.get_output(0), p6_in_w1.get_output(0), trt.ElementWiseOperation.SUM)
    p6_up_swish = addSwish(network = network,inputTensor = p6_in_w.get_output(0))

    p6_up_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p6_up_swish.get_output(0), 
                        num_output_maps = 160, ksize = 3, stride = 1, group = 160, 
                        lname = lname+'.conv6_up.depthwise_conv.conv', 
                        pre_padding = (1, 1), post_padding = (1, 1))
    p6_up_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p6_up_depthwise_conv.get_output(0), 
                        num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                        lname = lname+'.conv6_up.pointwise_conv.conv')
    p6_up_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p6_up_pointwise_conv.get_output(0),
                         lname = lname+'.conv6_up.bn')
    # 已验证
    # ----------------------- P5_0 and P6_1 to P5_1-----------------------------
    p5_w1_relu=p5_w1*(p5_w1>0)
    p5_w1_sum=np.sum(p5_w1_relu.numpy())+epsilon
    weight5=(p5_w1_relu/p5_w1_sum).numpy()
    weight0_p5_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p5_in_scale = np.full((160,),weight5[0])
    weight0_p5_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p5_in_w0=network.add_scale(p5_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p5_in_shift,scale=weight0_p5_in_scale)
    #p7_in_upsample = addUpsample(network, p7_in.get_output(0))

    p6_up_upsample = addUpsample(network, p6_up_bn.get_output(0), [1, 2, 2])

    weight1_p5p6up_in_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p5p6up_in_scale = np.full((160,),weight5[1])
    weight1_p5p6up_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_up_w=network.add_scale(p6_up_upsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p5p6up_in_shift,scale=weight1_p5p6up_in_scale)

    p5p6up_in_w = network.add_elementwise(p5_in_w0.get_output(0), p6_up_w.get_output(0), trt.ElementWiseOperation.SUM)
    p5p6up_swish = addSwish(network = network,inputTensor = p5p6up_in_w.get_output(0))

    p5_up_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p5p6up_swish.get_output(0), 
                num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv5_up.depthwise_conv.conv', 
                pre_padding = (1, 1), post_padding = (1, 1))
    p5_up_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p5_up_depthwise_conv.get_output(0), 
                num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                lname = lname+'.conv5_up.pointwise_conv.conv')
    p5_up_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p5_up_pointwise_conv.get_output(0),
                         lname = lname+'.conv5_up.bn')    
    # 已验证
    # ----------------------- P4_0 and P5_1 to P4_1-----------------------------
    p4_w1_relu=p4_w1*(p4_w1>0)
    p4_w1_sum=np.sum(p4_w1_relu.numpy())+epsilon
    weight4=(p4_w1_relu/p4_w1_sum).numpy()
    weight0_p4_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p4_in_scale = np.full((160,),weight4[0])
    weight0_p4_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p4_in_w0=network.add_scale(p4_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p4_in_shift,scale=weight0_p4_in_scale)
    #p7_in_upsample = addUpsample(network, p7_in.get_output(0))

    p5_up_upsample = addUpsample(network, p5_up_bn.get_output(0), [1, 2, 2])

    weight1_p4p5up_in_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p4p5up_in_scale = np.full((160,),weight4[1])
    weight1_p4p5up_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p5_up_w=network.add_scale(p5_up_upsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p4p5up_in_shift,scale=weight1_p4p5up_in_scale)

    p4p5up_in_w = network.add_elementwise(p4_in_w0.get_output(0), p5_up_w.get_output(0), trt.ElementWiseOperation.SUM)
    p4p5up_swish = addSwish(network = network,inputTensor = p4p5up_in_w.get_output(0))

    p4_up_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p4p5up_swish.get_output(0), 
                        num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv4_up.depthwise_conv.conv', 
                        pre_padding = (1, 1), post_padding = (1, 1))
    p4_up_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p4_up_depthwise_conv.get_output(0), 
                        num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                        lname = lname+'.conv4_up.pointwise_conv.conv')
    p4_up_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p4_up_pointwise_conv.get_output(0),
                         lname = lname+'.conv4_up.bn') 
    # 已验证
    # ----------------------- P3_0 and P4_1 to P3_2-----------------------------
    p3_w1_relu=p3_w1*(p3_w1>0)
    p3_w1_sum=np.sum(p3_w1_relu.numpy())+epsilon
    weight3=(p3_w1_relu/p3_w1_sum).numpy()
    weight0_p3_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p3_in_scale = np.full((160,),weight3[0])
    weight0_p3_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p3_in_w0=network.add_scale(p3_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p3_in_shift,scale=weight0_p3_in_scale)
    #p7_in_upsample = addUpsample(network, p7_in.get_output(0))

    p4_up_upsample = addUpsample(network, p4_up_bn.get_output(0), [1, 2, 2])

    weight1_p3p4up_in_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p3p4up_in_scale = np.full((160,),weight3[1])
    weight1_p3p4up_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p4_up_w=network.add_scale(p4_up_upsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p3p4up_in_shift,scale=weight1_p3p4up_in_scale)

    p3p4up_in_w = network.add_elementwise(p3_in_w0.get_output(0), p4_up_w.get_output(0), trt.ElementWiseOperation.SUM)
    p3p4up_swish = addSwish(network = network,inputTensor = p3p4up_in_w.get_output(0))

    p3_out_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p3p4up_swish.get_output(0),
                    num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv3_up.depthwise_conv.conv',
                    pre_padding = (1, 1), post_padding = (1, 1))
    p3_out_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p3_out_depthwise_conv.get_output(0), 
                    num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = lname+'.conv3_up.pointwise_conv.conv')
    p3_out_bn = addBatchNorm2d(network = network,
                    weights = weights, 
                    inputTensor = p3_out_pointwise_conv.get_output(0),
                    lname = lname+'.conv3_up.bn') 
    # 已验证
    # # ----------------------- P4_0, P4_1 and p3_2 to P4_2-----------------------------
    p4_w2_relu=p4_w2*(p4_w2>0)
    p4_w2_sum=np.sum(p4_w2_relu.numpy())+epsilon
    weight4_2=(p4_w2_relu/p4_w2_sum).numpy()
    weight0_p4_2_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p4_2_in_scale = np.full((160,),weight4_2[0])
    weight0_p4_2_in_scale.astype(np.float32)
    #weight_p4_2_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))

    p4_2_in_w0=network.add_scale(p4_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p4_2_in_shift,scale=weight0_p4_2_in_scale)
    weight0_p4_2_up_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p4_2_up_scale = np.full((160,),weight4_2[1])
    weight0_p4_2_up_scale.astype(np.float32)
    p4_2_up_w0=network.add_scale(p4_up_bn.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p4_2_up_shift,scale=weight0_p4_2_up_scale)
    
    p3_out_downsample = maxPoolingBlock(network, weights, p3_out_bn.get_output(0),3, 2,  pre_padding = (0, 0), post_padding = (1,1))

    
    weight1_p3_out_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p3_out_scale = np.full((160,),weight4_2[2])
    weight1_p3_out_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p3_out_down_w=network.add_scale(p3_out_downsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p3_out_shift,scale=weight1_p3_out_scale)
    
    p4_0_p4_2_sum = network.add_elementwise(p4_2_in_w0.get_output(0), p4_2_up_w0.get_output(0), trt.ElementWiseOperation.SUM)
    
    p4_0_p4_2_p3_2_sum = network.add_elementwise(p4_0_p4_2_sum.get_output(0), p3_out_down_w.get_output(0), trt.ElementWiseOperation.SUM)

    p4_0_p4_2_p3_2_swish = addSwish(network = network,inputTensor = p4_0_p4_2_p3_2_sum.get_output(0))

    
    p4_out_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p4_0_p4_2_p3_2_swish.get_output(0), 
                    num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv4_down.depthwise_conv.conv', 
                    pre_padding = (1, 1), post_padding = (1, 1))
    p4_out_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p4_out_depthwise_conv.get_output(0), 
                    num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                    lname = lname+'.conv4_down.pointwise_conv.conv')
    p4_out_bn = addBatchNorm2d(network = network,
                    weights = weights, 
                    inputTensor = p4_out_pointwise_conv.get_output(0),
                    lname = lname+'.conv4_down.bn') 
    # 已验证
    # -----------------------  Weights for P5_0, P5_1 and P4_2 to P5_2-----------------------------
    p5_w2_relu=p5_w2*(p5_w2>0)
    p5_w2_sum=np.sum(p5_w2_relu.numpy())+epsilon

    weight5_2=(p5_w2_relu/p5_w2_sum).numpy()
    weight0_p5_2_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p5_2_in_scale = np.full((160,),weight5_2[0])
    weight0_p5_2_in_scale.astype(np.float32)
    #weight_p4_2_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p5_in_w0=network.add_scale(p5_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p5_2_in_shift,scale=weight0_p5_2_in_scale)
    
    
    weight0_p5_2_up_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p5_2_up_scale = np.full((160,),weight5_2[1])
    weight0_p5_2_up_scale.astype(np.float32)
    p5_up_w1=network.add_scale(p5_up_bn.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p5_2_up_shift,scale=weight0_p5_2_up_scale)
    

    p4_out_downsample = maxPoolingBlock(network, weights, p4_out_bn.get_output(0),3, 2,  pre_padding = (0, 0), post_padding = (1,1))
    weight1_p4_out_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p4_out_scale = np.full((160,),weight5_2[2])
    weight1_p4_out_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p4_out_down_w=network.add_scale(p4_out_downsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p4_out_shift,scale=weight1_p4_out_scale)
    

    p5_0_p5_2_sum = network.add_elementwise(p5_in_w0.get_output(0), p5_up_w1.get_output(0), trt.ElementWiseOperation.SUM)
    
    p5_0_p5_2_p4_2_sum = network.add_elementwise(p5_0_p5_2_sum.get_output(0), p4_out_down_w.get_output(0), trt.ElementWiseOperation.SUM)

    p5_0_p5_2_p4_2_swish = addSwish(network = network,inputTensor = p5_0_p5_2_p4_2_sum.get_output(0))

    
    p5_out_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p5_0_p5_2_p4_2_swish.get_output(0), 
                    num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv5_down.depthwise_conv.conv', 
                    pre_padding = (1, 1), post_padding = (1, 1))
    p5_out_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p5_out_depthwise_conv.get_output(0), 
                    num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                    lname = lname+'.conv5_down.pointwise_conv.conv')
    p5_out_bn = addBatchNorm2d(network = network,
                    weights = weights, 
                    inputTensor = p5_out_pointwise_conv.get_output(0),
                    lname = lname+'.conv5_down.bn') 
    # -----------------------  #  Weights for P6_0, P6_1 and P5_2 to P6_2-----------------------------
    p6_w2_relu=p6_w2*(p6_w2>0)
    p6_w2_sum=np.sum(p6_w2_relu.numpy())+epsilon

    weight6_2=(p6_w2_relu/p6_w2_sum).numpy()
    weight0_p6_2_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p6_2_in_scale = np.full((160,),weight6_2[0])
    weight0_p6_2_in_scale.astype(np.float32)
    #weight_p4_2_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_in_w0=network.add_scale(p6_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p6_2_in_shift,scale=weight0_p6_2_in_scale)
    
    
    weight0_p6_2_up_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p6_2_up_scale = np.full((160,),weight6_2[1])
    weight0_p6_2_up_scale.astype(np.float32)
    p6_up_w1=network.add_scale(p6_up_bn.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p6_2_up_shift,scale=weight0_p6_2_up_scale)
    

    p5_out_downsample = maxPoolingBlock(network, weights, p5_out_bn.get_output(0),3, 2,  pre_padding = (0, 0), post_padding = (1,1))
    weight1_p5_out_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p5_out_scale = np.full((160,),weight6_2[2])
    weight1_p5_out_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p5_out_down_w=network.add_scale(p5_out_downsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p5_out_shift,scale=weight1_p5_out_scale)
    

    p6_0_p6_2_sum = network.add_elementwise(p6_in_w0.get_output(0), p6_up_w1.get_output(0), trt.ElementWiseOperation.SUM)
    
    p6_0_p6_2_p5_2_sum = network.add_elementwise(p6_0_p6_2_sum.get_output(0), p5_out_down_w.get_output(0), trt.ElementWiseOperation.SUM)

    p6_0_p6_2_p5_2_swish = addSwish(network = network,inputTensor = p6_0_p6_2_p5_2_sum.get_output(0))

    
    p6_out_depthwise_conv = convBlock(network = network, weights = weights, 
                            inputTensor = p6_0_p6_2_p5_2_swish.get_output(0), 
                            num_output_maps = 160, ksize = 3, stride = 1, group = 160,
                            lname = lname+'.conv6_down.depthwise_conv.conv',
                            pre_padding = (1, 1), post_padding = (1, 1))
    p6_out_pointwise_conv = convBlock(network = network, weights = weights, 
                            inputTensor = p6_out_depthwise_conv.get_output(0),
                            num_output_maps = 160, ksize = 1, stride = 1, group = 1,
                            lname = lname+'.conv6_down.pointwise_conv.conv')
    p6_out_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p6_out_pointwise_conv.get_output(0),
                         lname = lname+'.conv6_down.bn') 
    # -----------------------  # # Weights for P7_0 and P6_2 to P7_2-----------------------------
    p7_w2_relu=p7_w2*(p7_w2>0)
    p7_w2_sum=np.sum(p7_w2_relu.numpy())+epsilon

    weight7_2=(p7_w2_relu/p7_w2_sum).numpy()
    weight0_p7_2_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p7_2_in_scale = np.full((160,),weight7_2[0])
    weight0_p7_2_in_scale.astype(np.float32)
    #weight_p4_2_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p7_in_w0=network.add_scale(p7_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p7_2_in_shift,scale=weight0_p7_2_in_scale)
   
    p6_out_downsample = maxPoolingBlock(network, weights, p6_out_bn.get_output(0),3, 2,  pre_padding = (0, 0), post_padding = (1,1))
    weight1_p6_out_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p6_out_scale = np.full((160,),weight7_2[1])
    weight1_p6_out_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_out_down_w=network.add_scale(p6_out_downsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p6_out_shift,scale=weight1_p6_out_scale)
        
    p7_0_p6_1_sum = network.add_elementwise(p7_in_w0.get_output(0), p6_out_down_w.get_output(0), trt.ElementWiseOperation.SUM)

    p7_0_p6_1_swish = addSwish(network = network,inputTensor = p7_0_p6_1_sum.get_output(0))

    p7_out_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p7_0_p6_1_swish.get_output(0), 
                    num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv7_down.depthwise_conv.conv', 
                    pre_padding = (1, 1), post_padding = (1, 1))
    p7_out_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p7_out_depthwise_conv.get_output(0), 
                    num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                    lname = lname+'.conv7_down.pointwise_conv.conv')
    p7_out_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p7_out_pointwise_conv.get_output(0),
                         lname = lname+'.conv7_down.bn')
    # --------------------------------------------bifpn5--------------------------------
    lname="bifpn.5"
    p3_in=p3_out_bn
    p4_in=p4_out_bn
    p5_in=p5_out_bn
    p6_in=p6_out_bn
    p7_in=p7_out_bn

    p6_w1=weights[lname+".p6_w1"]
    p5_w1=weights[lname+".p5_w1"]
    p4_w1=weights[lname+".p4_w1"]
    p3_w1=weights[lname+".p3_w1"]

    p4_w2=weights[lname+".p4_w2"]
    p5_w2=weights[lname+".p5_w2"]
    p6_w2=weights[lname+".p6_w2"]
    p7_w2=weights[lname+".p7_w2"]
    
    epsilon=1e-4
    # ----------------------- P6_0 and P7_0 to P6_1-----------------------------
    p6_w1_relu=p6_w1*(p6_w1>0)
    p6_w1_sum=np.sum(p6_w1_relu.numpy())+epsilon
    weight6=(p6_w1_relu/p6_w1_sum).numpy()
    weight0_p6_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p6_in_scale = np.full((160,),weight6[0])
    weight0_p6_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_in_w0=network.add_scale(p6_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p6_in_shift,scale=weight0_p6_in_scale)
    #p7_in_upsample = addUpsample(network, p7_in.get_output(0))

    p6_upsample = addUpsample(network, p7_in.get_output(0), [1, 2, 2])

    weight1_p6p7_in_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p6p7_in_scale = np.full((160,),weight6[1])
    weight1_p6p7_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_in_w1=network.add_scale(p6_upsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p6p7_in_shift,scale=weight1_p6p7_in_scale)

    p6_in_w = network.add_elementwise(p6_in_w0.get_output(0), p6_in_w1.get_output(0), trt.ElementWiseOperation.SUM)
    p6_up_swish = addSwish(network = network,inputTensor = p6_in_w.get_output(0))

    p6_up_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p6_up_swish.get_output(0), 
                        num_output_maps = 160, ksize = 3, stride = 1, group = 160, 
                        lname = lname+'.conv6_up.depthwise_conv.conv', 
                        pre_padding = (1, 1), post_padding = (1, 1))
    p6_up_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p6_up_depthwise_conv.get_output(0), 
                        num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                        lname = lname+'.conv6_up.pointwise_conv.conv')
    p6_up_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p6_up_pointwise_conv.get_output(0),
                         lname = lname+'.conv6_up.bn')
    # 已验证
    # ----------------------- P5_0 and P6_1 to P5_1-----------------------------
    p5_w1_relu=p5_w1*(p5_w1>0)
    p5_w1_sum=np.sum(p5_w1_relu.numpy())+epsilon
    weight5=(p5_w1_relu/p5_w1_sum).numpy()
    weight0_p5_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p5_in_scale = np.full((160,),weight5[0])
    weight0_p5_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p5_in_w0=network.add_scale(p5_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p5_in_shift,scale=weight0_p5_in_scale)
    #p7_in_upsample = addUpsample(network, p7_in.get_output(0))

    p6_up_upsample = addUpsample(network, p6_up_bn.get_output(0), [1, 2, 2])

    weight1_p5p6up_in_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p5p6up_in_scale = np.full((160,),weight5[1])
    weight1_p5p6up_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_up_w=network.add_scale(p6_up_upsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p5p6up_in_shift,scale=weight1_p5p6up_in_scale)

    p5p6up_in_w = network.add_elementwise(p5_in_w0.get_output(0), p6_up_w.get_output(0), trt.ElementWiseOperation.SUM)
    p5p6up_swish = addSwish(network = network,inputTensor = p5p6up_in_w.get_output(0))

    p5_up_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p5p6up_swish.get_output(0), 
                num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv5_up.depthwise_conv.conv', 
                pre_padding = (1, 1), post_padding = (1, 1))
    p5_up_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p5_up_depthwise_conv.get_output(0), 
                num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                lname = lname+'.conv5_up.pointwise_conv.conv')
    p5_up_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p5_up_pointwise_conv.get_output(0),
                         lname = lname+'.conv5_up.bn')    
    # 已验证
    # ----------------------- P4_0 and P5_1 to P4_1-----------------------------
    p4_w1_relu=p4_w1*(p4_w1>0)
    p4_w1_sum=np.sum(p4_w1_relu.numpy())+epsilon
    weight4=(p4_w1_relu/p4_w1_sum).numpy()
    weight0_p4_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p4_in_scale = np.full((160,),weight4[0])
    weight0_p4_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p4_in_w0=network.add_scale(p4_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p4_in_shift,scale=weight0_p4_in_scale)
    #p7_in_upsample = addUpsample(network, p7_in.get_output(0))

    p5_up_upsample = addUpsample(network, p5_up_bn.get_output(0), [1, 2, 2])

    weight1_p4p5up_in_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p4p5up_in_scale = np.full((160,),weight4[1])
    weight1_p4p5up_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p5_up_w=network.add_scale(p5_up_upsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p4p5up_in_shift,scale=weight1_p4p5up_in_scale)

    p4p5up_in_w = network.add_elementwise(p4_in_w0.get_output(0), p5_up_w.get_output(0), trt.ElementWiseOperation.SUM)
    p4p5up_swish = addSwish(network = network,inputTensor = p4p5up_in_w.get_output(0))

    p4_up_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p4p5up_swish.get_output(0), 
                        num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv4_up.depthwise_conv.conv', 
                        pre_padding = (1, 1), post_padding = (1, 1))
    p4_up_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p4_up_depthwise_conv.get_output(0), 
                        num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                        lname = lname+'.conv4_up.pointwise_conv.conv')
    p4_up_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p4_up_pointwise_conv.get_output(0),
                         lname = lname+'.conv4_up.bn') 
    # 已验证
    # ----------------------- P3_0 and P4_1 to P3_2-----------------------------
    p3_w1_relu=p3_w1*(p3_w1>0)
    p3_w1_sum=np.sum(p3_w1_relu.numpy())+epsilon
    weight3=(p3_w1_relu/p3_w1_sum).numpy()
    weight0_p3_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p3_in_scale = np.full((160,),weight3[0])
    weight0_p3_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p3_in_w0=network.add_scale(p3_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p3_in_shift,scale=weight0_p3_in_scale)
    #p7_in_upsample = addUpsample(network, p7_in.get_output(0))

    p4_up_upsample = addUpsample(network, p4_up_bn.get_output(0), [1, 2, 2])

    weight1_p3p4up_in_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p3p4up_in_scale = np.full((160,),weight3[1])
    weight1_p3p4up_in_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p4_up_w=network.add_scale(p4_up_upsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p3p4up_in_shift,scale=weight1_p3p4up_in_scale)

    p3p4up_in_w = network.add_elementwise(p3_in_w0.get_output(0), p4_up_w.get_output(0), trt.ElementWiseOperation.SUM)
    p3p4up_swish = addSwish(network = network,inputTensor = p3p4up_in_w.get_output(0))

    p3_out_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p3p4up_swish.get_output(0),
                    num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv3_up.depthwise_conv.conv',
                    pre_padding = (1, 1), post_padding = (1, 1))
    p3_out_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p3_out_depthwise_conv.get_output(0), 
                    num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = lname+'.conv3_up.pointwise_conv.conv')
    p3_out_bn = addBatchNorm2d(network = network,
                    weights = weights, 
                    inputTensor = p3_out_pointwise_conv.get_output(0),
                    lname = lname+'.conv3_up.bn') 
    # 已验证
    # # ----------------------- P4_0, P4_1 and p3_2 to P4_2-----------------------------
    p4_w2_relu=p4_w2*(p4_w2>0)
    p4_w2_sum=np.sum(p4_w2_relu.numpy())+epsilon
    weight4_2=(p4_w2_relu/p4_w2_sum).numpy()
    weight0_p4_2_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p4_2_in_scale = np.full((160,),weight4_2[0])
    weight0_p4_2_in_scale.astype(np.float32)
    #weight_p4_2_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))

    p4_2_in_w0=network.add_scale(p4_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p4_2_in_shift,scale=weight0_p4_2_in_scale)
    weight0_p4_2_up_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p4_2_up_scale = np.full((160,),weight4_2[1])
    weight0_p4_2_up_scale.astype(np.float32)
    p4_2_up_w0=network.add_scale(p4_up_bn.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p4_2_up_shift,scale=weight0_p4_2_up_scale)
    
    p3_out_downsample = maxPoolingBlock(network, weights, p3_out_bn.get_output(0),3, 2,  pre_padding = (0, 0), post_padding = (1,1))

    
    weight1_p3_out_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p3_out_scale = np.full((160,),weight4_2[2])
    weight1_p3_out_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p3_out_down_w=network.add_scale(p3_out_downsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p3_out_shift,scale=weight1_p3_out_scale)
    
    p4_0_p4_2_sum = network.add_elementwise(p4_2_in_w0.get_output(0), p4_2_up_w0.get_output(0), trt.ElementWiseOperation.SUM)
    
    p4_0_p4_2_p3_2_sum = network.add_elementwise(p4_0_p4_2_sum.get_output(0), p3_out_down_w.get_output(0), trt.ElementWiseOperation.SUM)

    p4_0_p4_2_p3_2_swish = addSwish(network = network,inputTensor = p4_0_p4_2_p3_2_sum.get_output(0))

    
    p4_out_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p4_0_p4_2_p3_2_swish.get_output(0), 
                    num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv4_down.depthwise_conv.conv', 
                    pre_padding = (1, 1), post_padding = (1, 1))
    p4_out_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p4_out_depthwise_conv.get_output(0), 
                    num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                    lname = lname+'.conv4_down.pointwise_conv.conv')
    p4_out_bn = addBatchNorm2d(network = network,
                    weights = weights, 
                    inputTensor = p4_out_pointwise_conv.get_output(0),
                    lname = lname+'.conv4_down.bn') 
    # 已验证
    # -----------------------  Weights for P5_0, P5_1 and P4_2 to P5_2-----------------------------
    p5_w2_relu=p5_w2*(p5_w2>0)
    p5_w2_sum=np.sum(p5_w2_relu.numpy())+epsilon

    weight5_2=(p5_w2_relu/p5_w2_sum).numpy()
    weight0_p5_2_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p5_2_in_scale = np.full((160,),weight5_2[0])
    weight0_p5_2_in_scale.astype(np.float32)
    #weight_p4_2_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p5_in_w0=network.add_scale(p5_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p5_2_in_shift,scale=weight0_p5_2_in_scale)
    
    
    weight0_p5_2_up_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p5_2_up_scale = np.full((160,),weight5_2[1])
    weight0_p5_2_up_scale.astype(np.float32)
    p5_up_w1=network.add_scale(p5_up_bn.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p5_2_up_shift,scale=weight0_p5_2_up_scale)
    

    p4_out_downsample = maxPoolingBlock(network, weights, p4_out_bn.get_output(0),3, 2,  pre_padding = (0, 0), post_padding = (1,1))
    weight1_p4_out_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p4_out_scale = np.full((160,),weight5_2[2])
    weight1_p4_out_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p4_out_down_w=network.add_scale(p4_out_downsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p4_out_shift,scale=weight1_p4_out_scale)
    

    p5_0_p5_2_sum = network.add_elementwise(p5_in_w0.get_output(0), p5_up_w1.get_output(0), trt.ElementWiseOperation.SUM)
    
    p5_0_p5_2_p4_2_sum = network.add_elementwise(p5_0_p5_2_sum.get_output(0), p4_out_down_w.get_output(0), trt.ElementWiseOperation.SUM)

    p5_0_p5_2_p4_2_swish = addSwish(network = network,inputTensor = p5_0_p5_2_p4_2_sum.get_output(0))

    
    p5_out_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p5_0_p5_2_p4_2_swish.get_output(0), 
                    num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv5_down.depthwise_conv.conv', 
                    pre_padding = (1, 1), post_padding = (1, 1))
    p5_out_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p5_out_depthwise_conv.get_output(0), 
                    num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                    lname = lname+'.conv5_down.pointwise_conv.conv')
    p5_out_bn = addBatchNorm2d(network = network,
                    weights = weights, 
                    inputTensor = p5_out_pointwise_conv.get_output(0),
                    lname = lname+'.conv5_down.bn') 
    # -----------------------  #  Weights for P6_0, P6_1 and P5_2 to P6_2-----------------------------
    p6_w2_relu=p6_w2*(p6_w2>0)
    p6_w2_sum=np.sum(p6_w2_relu.numpy())+epsilon

    weight6_2=(p6_w2_relu/p6_w2_sum).numpy()
    weight0_p6_2_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p6_2_in_scale = np.full((160,),weight6_2[0])
    weight0_p6_2_in_scale.astype(np.float32)
    #weight_p4_2_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_in_w0=network.add_scale(p6_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p6_2_in_shift,scale=weight0_p6_2_in_scale)
    
    
    weight0_p6_2_up_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p6_2_up_scale = np.full((160,),weight6_2[1])
    weight0_p6_2_up_scale.astype(np.float32)
    p6_up_w1=network.add_scale(p6_up_bn.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p6_2_up_shift,scale=weight0_p6_2_up_scale)
    

    p5_out_downsample = maxPoolingBlock(network, weights, p5_out_bn.get_output(0),3, 2,  pre_padding = (0, 0), post_padding = (1,1))
    weight1_p5_out_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p5_out_scale = np.full((160,),weight6_2[2])
    weight1_p5_out_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p5_out_down_w=network.add_scale(p5_out_downsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p5_out_shift,scale=weight1_p5_out_scale)
    

    p6_0_p6_2_sum = network.add_elementwise(p6_in_w0.get_output(0), p6_up_w1.get_output(0), trt.ElementWiseOperation.SUM)
    
    p6_0_p6_2_p5_2_sum = network.add_elementwise(p6_0_p6_2_sum.get_output(0), p5_out_down_w.get_output(0), trt.ElementWiseOperation.SUM)

    p6_0_p6_2_p5_2_swish = addSwish(network = network,inputTensor = p6_0_p6_2_p5_2_sum.get_output(0))

    
    p6_out_depthwise_conv = convBlock(network = network, weights = weights, 
                            inputTensor = p6_0_p6_2_p5_2_swish.get_output(0), 
                            num_output_maps = 160, ksize = 3, stride = 1, group = 160,
                            lname = lname+'.conv6_down.depthwise_conv.conv',
                            pre_padding = (1, 1), post_padding = (1, 1))
    p6_out_pointwise_conv = convBlock(network = network, weights = weights, 
                            inputTensor = p6_out_depthwise_conv.get_output(0),
                            num_output_maps = 160, ksize = 1, stride = 1, group = 1,
                            lname = lname+'.conv6_down.pointwise_conv.conv')
    p6_out_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p6_out_pointwise_conv.get_output(0),
                         lname = lname+'.conv6_down.bn') 
    # -----------------------  # # Weights for P7_0 and P6_2 to P7_2-----------------------------
    p7_w2_relu=p7_w2*(p7_w2>0)
    p7_w2_sum=np.sum(p7_w2_relu.numpy())+epsilon

    weight7_2=(p7_w2_relu/p7_w2_sum).numpy()
    weight0_p7_2_in_shift = np.zeros((160, ), dtype=np.float32)
    weight0_p7_2_in_scale = np.full((160,),weight7_2[0])
    weight0_p7_2_in_scale.astype(np.float32)
    #weight_p4_2_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p7_in_w0=network.add_scale(p7_in.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight0_p7_2_in_shift,scale=weight0_p7_2_in_scale)
   
    p6_out_downsample = maxPoolingBlock(network, weights, p6_out_bn.get_output(0),3, 2,  pre_padding = (0, 0), post_padding = (1,1))
    weight1_p6_out_shift = np.zeros((160, ), dtype=np.float32)
    weight1_p6_out_scale = np.full((160,),weight7_2[1])
    weight1_p6_out_scale.astype(np.float32)
    #weight_p6_in_shift = trt.Weights(np.zeros_like(weight_p6_in_scale))
    p6_out_down_w=network.add_scale(p6_out_downsample.get_output(0),mode=trt.ScaleMode.CHANNEL,shift=weight1_p6_out_shift,scale=weight1_p6_out_scale)
        
    p7_0_p6_1_sum = network.add_elementwise(p7_in_w0.get_output(0), p6_out_down_w.get_output(0), trt.ElementWiseOperation.SUM)

    p7_0_p6_1_swish = addSwish(network = network,inputTensor = p7_0_p6_1_sum.get_output(0))

    p7_out_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p7_0_p6_1_swish.get_output(0), 
                    num_output_maps = 160, ksize = 3, stride = 1, group = 160, lname = lname+'.conv7_down.depthwise_conv.conv', 
                    pre_padding = (1, 1), post_padding = (1, 1))
    p7_out_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = p7_out_depthwise_conv.get_output(0), 
                    num_output_maps = 160, ksize = 1, stride = 1, group = 1, 
                    lname = lname+'.conv7_down.pointwise_conv.conv')
    p7_out_bn = addBatchNorm2d(network = network,
                         weights = weights, 
                         inputTensor = p7_out_pointwise_conv.get_output(0),
                         lname = lname+'.conv7_down.bn')

    p3_in=p3_out_bn
    p4_in=p4_out_bn
    p5_in=p5_out_bn
    p6_in=p6_out_bn
    p7_in=p7_out_bn

    return p3_in, p4_in, p5_in, p6_in, p7_in

def Regressor(network, weights, input_tenors):
    p3_in, p4_in, p5_in, p6_in, p7_in = input_tenors
    #-----------------------------------------------------------------------------------------------#   p3_in
    #for0
    regressor_conv_list0_0_depthwise_conv = convBlock(network = network, weights = weights, inputTensor =   p3_in.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1),     post_padding = (1, 1), group = 160, lname = 'regressor.conv_list.0.depthwise_conv.conv')
    regressor_conv_list0_0_pointwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_conv_list0_0_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1,    group = 1, lname = 'regressor.conv_list.0.pointwise_conv.conv')
    regressor_bn_list_0_0  = addBatchNorm2d(network = network,  weights = weights, inputTensor =    regressor_conv_list0_0_pointwise_conv.get_output(0), lname = 'regressor.bn_list.0.0')
    regressor_swish_0_0 = addSwish(network, inputTensor = regressor_bn_list_0_0.get_output(0))
    #for1
    regressor_conv_list0_1_depthwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_swish_0_0.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1)    , post_padding = (1, 1), group = 160, lname = 'regressor.conv_list.1.depthwise_conv.conv')
    regressor_conv_list0_1_pointwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_conv_list0_1_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1,    group = 1, lname = 'regressor.conv_list.1.pointwise_conv.conv')
    regressor_bn_list_0_1  = addBatchNorm2d(network = network,  weights = weights, inputTensor =    regressor_conv_list0_1_pointwise_conv.get_output(0), lname = 'regressor.bn_list.0.1')
    regressor_swish_0_1 = addSwish(network, inputTensor = regressor_bn_list_0_1.get_output(0))
    #for2
    regressor_conv_list0_2_depthwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_swish_0_1.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1)    , post_padding = (1, 1), group = 160, lname = 'regressor.conv_list.2.depthwise_conv.conv')
    regressor_conv_list0_2_pointwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_conv_list0_2_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1,    group = 1, lname = 'regressor.conv_list.2.pointwise_conv.conv')
    regressor_bn_list_0_2  = addBatchNorm2d(network = network,  weights = weights, inputTensor =    regressor_conv_list0_2_pointwise_conv.get_output(0), lname = 'regressor.bn_list.0.2')
    regressor_swish_0_2 = addSwish(network, inputTensor = regressor_bn_list_0_2.get_output(0))
    #for3
    regressor_conv_list0_3_depthwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_swish_0_2.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1)    , post_padding = (1, 1), group = 160, lname = 'regressor.conv_list.3.depthwise_conv.conv')
    regressor_conv_list0_3_pointwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_conv_list0_3_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1,    group = 1, lname = 'regressor.conv_list.3.pointwise_conv.conv')
    regressor_bn_list_0_3  = addBatchNorm2d(network = network,  weights = weights, inputTensor =    regressor_conv_list0_3_pointwise_conv.get_output(0), lname = 'regressor.bn_list.0.3')
    regressor_swish_0_3 = addSwish(network, inputTensor = regressor_bn_list_0_3.get_output(0))
    #header
    regressor_header_depthwise_conv0 = convBlock(network = network, weights = weights, inputTensor =    regressor_swish_0_3.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1) , post_padding = (1, 1), group = 160, lname = 'regressor.header.depthwise_conv.conv')
    regressor_header_pointwise_conv0 = convBlock(network = network, weights = weights, inputTensor =    regressor_header_depthwise_conv0.get_output(0), num_output_maps = 36, ksize = 1, stride = 1, group =   1, lname = 'regressor.header.pointwise_conv.conv')
    regressor_header_shuffle_0_0 = network.add_shuffle(regressor_header_pointwise_conv0.get_output(0))
    regressor_header_shuffle_0_0.first_transpose = (1, 2, 0)
    #feat[0]
    regressor_header_shuffle_0_1 = network.add_shuffle(regressor_header_shuffle_0_0.get_output(0))
    regressor_header_shuffle_0_1.reshape_dims  = (1, -1, 4)
    feat0 = regressor_header_shuffle_0_1
    #-----------------------------------------------------------------------------------------------#   p4_in
    #for0
    regressor_conv_list1_0_depthwise_conv = convBlock(network = network, weights = weights, inputTensor =   p4_in.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1),     post_padding = (1, 1), group = 160, lname = 'regressor.conv_list.0.depthwise_conv.conv')
    regressor_conv_list1_0_pointwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_conv_list1_0_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1,    group = 1, lname = 'regressor.conv_list.0.pointwise_conv.conv')
    regressor_bn_list_1_0  = addBatchNorm2d(network = network,  weights = weights, inputTensor =    regressor_conv_list1_0_pointwise_conv.get_output(0), lname = 'regressor.bn_list.1.0')
    regressor_swish_1_0 = addSwish(network, inputTensor = regressor_bn_list_1_0.get_output(0))
    #for1
    regressor_conv_list1_1_depthwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_swish_1_0.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1)    , post_padding = (1, 1), group = 160, lname = 'regressor.conv_list.1.depthwise_conv.conv')
    regressor_conv_list1_1_pointwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_conv_list1_1_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1,    group = 1, lname = 'regressor.conv_list.1.pointwise_conv.conv')
    regressor_bn_list_1_1  = addBatchNorm2d(network = network,  weights = weights, inputTensor =    regressor_conv_list1_1_pointwise_conv.get_output(0), lname = 'regressor.bn_list.1.1')
    regressor_swish_1_1 = addSwish(network, inputTensor = regressor_bn_list_1_1.get_output(0))
    #for2
    regressor_conv_list1_2_depthwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_swish_1_1.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1)    , post_padding = (1, 1), group = 160, lname = 'regressor.conv_list.2.depthwise_conv.conv')
    regressor_conv_list1_2_pointwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_conv_list1_2_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1,    group = 1, lname = 'regressor.conv_list.2.pointwise_conv.conv')
    regressor_bn_list_1_2  = addBatchNorm2d(network = network,  weights = weights, inputTensor =    regressor_conv_list1_2_pointwise_conv.get_output(0), lname = 'regressor.bn_list.1.2')
    regressor_swish_1_2 = addSwish(network, inputTensor = regressor_bn_list_1_2.get_output(0))
    #for3
    regressor_conv_list1_3_depthwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_swish_1_2.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1)    , post_padding = (1, 1), group = 160, lname = 'regressor.conv_list.3.depthwise_conv.conv')
    regressor_conv_list1_3_pointwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_conv_list1_3_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1,    group = 1, lname = 'regressor.conv_list.3.pointwise_conv.conv')
    regressor_bn_list_1_3  = addBatchNorm2d(network = network,  weights = weights, inputTensor =    regressor_conv_list1_3_pointwise_conv.get_output(0), lname = 'regressor.bn_list.1.3')
    regressor_swish_1_3 = addSwish(network, inputTensor = regressor_bn_list_1_3.get_output(0))
    #header
    regressor_header_depthwise_conv1 = convBlock(network = network, weights = weights, inputTensor =    regressor_swish_1_3.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1) , post_padding = (1, 1), group = 160, lname = 'regressor.header.depthwise_conv.conv')
    regressor_header_pointwise_conv1 = convBlock(network = network, weights = weights, inputTensor =    regressor_header_depthwise_conv1.get_output(0), num_output_maps = 36, ksize = 1, stride = 1, group =   1, lname = 'regressor.header.pointwise_conv.conv')
    regressor_header_shuffle_1_0 = network.add_shuffle(regressor_header_pointwise_conv1.get_output(0))
    regressor_header_shuffle_1_0.first_transpose = (1, 2, 0)
    #feat[1]
    regressor_header_shuffle_1_1 = network.add_shuffle(regressor_header_shuffle_1_0.get_output(0))
    regressor_header_shuffle_1_1.reshape_dims  = (1, -1, 4)
    feat1 = regressor_header_shuffle_1_1
    #-----------------------------------------------------------------------------------------------#   p5_in
    #for0
    regressor_conv_list2_0_depthwise_conv = convBlock(network = network, weights = weights, inputTensor =   p5_in.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1),     post_padding = (1, 1), group = 160, lname = 'regressor.conv_list.0.depthwise_conv.conv')
    regressor_conv_list2_0_pointwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_conv_list2_0_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1,    group = 1, lname = 'regressor.conv_list.0.pointwise_conv.conv')
    regressor_bn_list_2_0  = addBatchNorm2d(network = network,  weights = weights, inputTensor =    regressor_conv_list2_0_pointwise_conv.get_output(0), lname = 'regressor.bn_list.2.0')
    regressor_swish_2_0 = addSwish(network, inputTensor = regressor_bn_list_2_0.get_output(0))
    #for1
    regressor_conv_list2_1_depthwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_swish_2_0.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1)    , post_padding = (1, 1), group = 160, lname = 'regressor.conv_list.1.depthwise_conv.conv')
    regressor_conv_list2_1_pointwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_conv_list2_1_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1,    group = 1, lname = 'regressor.conv_list.1.pointwise_conv.conv')
    regressor_bn_list_2_1  = addBatchNorm2d(network = network,  weights = weights, inputTensor =    regressor_conv_list2_1_pointwise_conv.get_output(0), lname = 'regressor.bn_list.2.1')
    regressor_swish_2_1 = addSwish(network, inputTensor = regressor_bn_list_2_1.get_output(0))
    #for2
    regressor_conv_list2_2_depthwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_swish_2_1.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1)    , post_padding = (1, 1), group = 160, lname = 'regressor.conv_list.2.depthwise_conv.conv')
    regressor_conv_list2_2_pointwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_conv_list2_2_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1,    group = 1, lname = 'regressor.conv_list.2.pointwise_conv.conv')
    regressor_bn_list_2_2  = addBatchNorm2d(network = network,  weights = weights, inputTensor =    regressor_conv_list2_2_pointwise_conv.get_output(0), lname = 'regressor.bn_list.2.2')
    regressor_swish_2_2 = addSwish(network, inputTensor = regressor_bn_list_2_2.get_output(0))
    #for3
    regressor_conv_list2_3_depthwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_swish_2_2.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1)    , post_padding = (1, 1), group = 160, lname = 'regressor.conv_list.3.depthwise_conv.conv')
    regressor_conv_list2_3_pointwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_conv_list2_3_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1,    group = 1, lname = 'regressor.conv_list.3.pointwise_conv.conv')
    regressor_bn_list_2_3  = addBatchNorm2d(network = network,  weights = weights, inputTensor =    regressor_conv_list2_3_pointwise_conv.get_output(0), lname = 'regressor.bn_list.2.3')
    regressor_swish_2_3 = addSwish(network, inputTensor = regressor_bn_list_2_3.get_output(0))
    #header
    regressor_header_depthwise_conv2 = convBlock(network = network, weights = weights, inputTensor =    regressor_swish_2_3.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1) , post_padding = (1, 1), group = 160, lname = 'regressor.header.depthwise_conv.conv')
    regressor_header_pointwise_conv2 = convBlock(network = network, weights = weights, inputTensor =    regressor_header_depthwise_conv2.get_output(0), num_output_maps = 36, ksize = 1, stride = 1, group =   1, lname = 'regressor.header.pointwise_conv.conv')
    regressor_header_shuffle_2_0 = network.add_shuffle(regressor_header_pointwise_conv2.get_output(0))
    regressor_header_shuffle_2_0.first_transpose = (1, 2, 0)
    #feat[1]
    regressor_header_shuffle_2_1 = network.add_shuffle(regressor_header_shuffle_2_0.get_output(0))
    regressor_header_shuffle_2_1.reshape_dims  = (1, -1, 4)
    feat2 = regressor_header_shuffle_2_1
    #-----------------------------------------------------------------------------------------------#   p6_in
    #for0
    regressor_conv_list3_0_depthwise_conv = convBlock(network = network, weights = weights, inputTensor =   p6_in.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1),     post_padding = (1, 1), group = 160, lname = 'regressor.conv_list.0.depthwise_conv.conv')
    regressor_conv_list3_0_pointwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_conv_list3_0_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1,    group = 1, lname = 'regressor.conv_list.0.pointwise_conv.conv')
    regressor_bn_list_3_0  = addBatchNorm2d(network = network,  weights = weights, inputTensor =    regressor_conv_list3_0_pointwise_conv.get_output(0), lname = 'regressor.bn_list.3.0')
    regressor_swish_3_0 = addSwish(network, inputTensor = regressor_bn_list_3_0.get_output(0))
    #for1
    regressor_conv_list3_1_depthwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_swish_3_0.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1)    , post_padding = (1, 1), group = 160, lname = 'regressor.conv_list.1.depthwise_conv.conv')
    regressor_conv_list3_1_pointwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_conv_list3_1_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1,    group = 1, lname = 'regressor.conv_list.1.pointwise_conv.conv')
    regressor_bn_list_3_1  = addBatchNorm2d(network = network,  weights = weights, inputTensor =    regressor_conv_list3_1_pointwise_conv.get_output(0), lname = 'regressor.bn_list.3.1')
    regressor_swish_3_1 = addSwish(network, inputTensor = regressor_bn_list_3_1.get_output(0))
    #for2
    regressor_conv_list3_2_depthwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_swish_3_1.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1)    , post_padding = (1, 1), group = 160, lname = 'regressor.conv_list.2.depthwise_conv.conv')
    regressor_conv_list3_2_pointwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_conv_list3_2_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1,    group = 1, lname = 'regressor.conv_list.2.pointwise_conv.conv')
    regressor_bn_list_3_2  = addBatchNorm2d(network = network,  weights = weights, inputTensor =    regressor_conv_list3_2_pointwise_conv.get_output(0), lname = 'regressor.bn_list.3.2')
    regressor_swish_3_2 = addSwish(network, inputTensor = regressor_bn_list_3_2.get_output(0))
    #for3
    regressor_conv_list3_3_depthwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_swish_3_2.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1)    , post_padding = (1, 1), group = 160, lname = 'regressor.conv_list.3.depthwise_conv.conv')
    regressor_conv_list3_3_pointwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_conv_list3_3_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1,    group = 1, lname = 'regressor.conv_list.3.pointwise_conv.conv')
    regressor_bn_list_3_3  = addBatchNorm2d(network = network,  weights = weights, inputTensor =    regressor_conv_list3_3_pointwise_conv.get_output(0), lname = 'regressor.bn_list.3.3')
    regressor_swish_3_3 = addSwish(network, inputTensor = regressor_bn_list_3_3.get_output(0))
    #header
    regressor_header_depthwise_conv3 = convBlock(network = network, weights = weights, inputTensor =    regressor_swish_3_3.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1) , post_padding = (1, 1), group = 160, lname = 'regressor.header.depthwise_conv.conv')
    regressor_header_pointwise_conv3 = convBlock(network = network, weights = weights, inputTensor =    regressor_header_depthwise_conv3.get_output(0), num_output_maps = 36, ksize = 1, stride = 1, group =   1, lname = 'regressor.header.pointwise_conv.conv')
    regressor_header_shuffle_3_0 = network.add_shuffle(regressor_header_pointwise_conv3.get_output(0))
    regressor_header_shuffle_3_0.first_transpose = (1, 2, 0)
    #feat[1]
    regressor_header_shuffle_3_1 = network.add_shuffle(regressor_header_shuffle_3_0.get_output(0))
    regressor_header_shuffle_3_1.reshape_dims  = (1, -1, 4)
    feat3 = regressor_header_shuffle_3_1
    #-----------------------------------------------------------------------------------------------#   p7_in
    #for0
    regressor_conv_list4_0_depthwise_conv = convBlock(network = network, weights = weights, inputTensor =   p7_in.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1),     post_padding = (1, 1), group = 160, lname = 'regressor.conv_list.0.depthwise_conv.conv')
    regressor_conv_list4_0_pointwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_conv_list4_0_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1,    group = 1, lname = 'regressor.conv_list.0.pointwise_conv.conv')
    regressor_bn_list_4_0  = addBatchNorm2d(network = network,  weights = weights, inputTensor =    regressor_conv_list4_0_pointwise_conv.get_output(0), lname = 'regressor.bn_list.4.0')
    regressor_swish_4_0 = addSwish(network, inputTensor = regressor_bn_list_4_0.get_output(0))
    #for1
    regressor_conv_list4_1_depthwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_swish_4_0.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1)    , post_padding = (1, 1), group = 160, lname = 'regressor.conv_list.1.depthwise_conv.conv')
    regressor_conv_list4_1_pointwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_conv_list4_1_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1,    group = 1, lname = 'regressor.conv_list.1.pointwise_conv.conv')
    regressor_bn_list_4_1  = addBatchNorm2d(network = network,  weights = weights, inputTensor =    regressor_conv_list4_1_pointwise_conv.get_output(0), lname = 'regressor.bn_list.4.1')
    regressor_swish_4_1 = addSwish(network, inputTensor = regressor_bn_list_4_1.get_output(0))
    #for2
    regressor_conv_list4_2_depthwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_swish_4_1.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1)    , post_padding = (1, 1), group = 160, lname = 'regressor.conv_list.2.depthwise_conv.conv')
    regressor_conv_list4_2_pointwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_conv_list4_2_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1,    group = 1, lname = 'regressor.conv_list.2.pointwise_conv.conv')
    regressor_bn_list_4_2  = addBatchNorm2d(network = network,  weights = weights, inputTensor =    regressor_conv_list4_2_pointwise_conv.get_output(0), lname = 'regressor.bn_list.4.2')
    regressor_swish_4_2 = addSwish(network, inputTensor = regressor_bn_list_4_2.get_output(0))
    #for3
    regressor_conv_list4_3_depthwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_swish_4_2.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1)    , post_padding = (1, 1), group = 160, lname = 'regressor.conv_list.3.depthwise_conv.conv')
    regressor_conv_list4_3_pointwise_conv = convBlock(network = network, weights = weights, inputTensor =   regressor_conv_list4_3_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1,    group = 1, lname = 'regressor.conv_list.3.pointwise_conv.conv')
    regressor_bn_list_4_3  = addBatchNorm2d(network = network,  weights = weights, inputTensor =    regressor_conv_list4_3_pointwise_conv.get_output(0), lname = 'regressor.bn_list.4.3')
    regressor_swish_4_3 = addSwish(network, inputTensor = regressor_bn_list_4_3.get_output(0))
    #header
    regressor_header_depthwise_conv4 = convBlock(network = network, weights = weights, inputTensor =    regressor_swish_4_3.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1) , post_padding = (1, 1), group = 160, lname = 'regressor.header.depthwise_conv.conv')
    regressor_header_pointwise_conv4 = convBlock(network = network, weights = weights, inputTensor =    regressor_header_depthwise_conv4.get_output(0), num_output_maps = 36, ksize = 1, stride = 1, group =   1, lname = 'regressor.header.pointwise_conv.conv')
    regressor_header_shuffle_4_0 = network.add_shuffle(regressor_header_pointwise_conv4.get_output(0))
    regressor_header_shuffle_4_0.first_transpose = (1, 2, 0)
    #feat[1]
    regressor_header_shuffle_4_1 = network.add_shuffle(regressor_header_shuffle_4_0.get_output(0))
    regressor_header_shuffle_4_1.reshape_dims  = (1, -1, 4)
    feat4 = regressor_header_shuffle_4_1
    feats = [feat0.get_output(0), feat1.get_output(0), feat2.get_output(0), feat3.get_output(0), feat4. get_output(0)]
    cat_feats = network.add_concatenation(feats)
    cat_feats.axis = 1
    return cat_feats

def Classifier(network, weights, input_tenors):
    p3_in, p4_in, p5_in, p6_in, p7_in = input_tenors
    #-----------------------------------------------------------------------------------------------# p4_in
    #for0
    classifier_conv_list0_0_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p3_in.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1), post_padding = (1, 1), group = 160, lname = 'classifier.conv_list.0.depthwise_conv.conv')

    classifier_conv_list0_0_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_conv_list0_0_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'classifier.conv_list.0.pointwise_conv.conv')

    classifier_bn_list_0_0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = classifier_conv_list0_0_pointwise_conv.get_output(0), lname = 'classifier.bn_list.0.0')
    classifier_swish_0_0 = addSwish(network, inputTensor = classifier_bn_list_0_0.get_output(0))
    #for1
    classifier_conv_list0_1_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_swish_0_0.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1), post_padding = (1, 1), group = 160, lname = 'classifier.conv_list.1.depthwise_conv.conv')
    
    classifier_conv_list0_1_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_conv_list0_1_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'classifier.conv_list.1.pointwise_conv.conv')

    classifier_bn_list_0_1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = classifier_conv_list0_1_pointwise_conv.get_output(0), lname = 'classifier.bn_list.0.1')

    classifier_swish_0_1 = addSwish(network, inputTensor = classifier_bn_list_0_1.get_output(0))

    #for2
    classifier_conv_list0_2_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_swish_0_1.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1), post_padding = (1, 1), group = 160, lname = 'classifier.conv_list.2.depthwise_conv.conv')

    classifier_conv_list0_2_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_conv_list0_2_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'classifier.conv_list.2.pointwise_conv.conv')

    classifier_bn_list_0_2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = classifier_conv_list0_2_pointwise_conv.get_output(0), lname = 'classifier.bn_list.0.2')

    classifier_swish_0_2 = addSwish(network, inputTensor = classifier_bn_list_0_2.get_output(0))

    #for3
    classifier_conv_list0_3_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_swish_0_2.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1), post_padding = (1, 1), group = 160, lname = 'classifier.conv_list.3.depthwise_conv.conv')

    classifier_conv_list0_3_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_conv_list0_3_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'classifier.conv_list.3.pointwise_conv.conv')

    classifier_bn_list_0_3  = addBatchNorm2d(network = network,  weights = weights, inputTensor = classifier_conv_list0_3_pointwise_conv.get_output(0), lname = 'classifier.bn_list.0.3')

    classifier_swish_0_3 = addSwish(network, inputTensor = classifier_bn_list_0_3.get_output(0))


    #header
    classifier_header_depthwise_conv0 = convBlock(network = network, weights = weights, inputTensor = classifier_swish_0_3.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1), post_padding = (1, 1), group = 160, lname = 'classifier.header.depthwise_conv.conv')

    classifier_header_pointwise_conv0 = convBlock(network = network, weights = weights, inputTensor = classifier_header_depthwise_conv0.get_output(0), num_output_maps = 810, ksize = 1, stride = 1, group = 1, lname = 'classifier.header.pointwise_conv.conv')

    classifier_header_shuffle_0_0 = network.add_shuffle(classifier_header_pointwise_conv0.get_output(0))
    classifier_header_shuffle_0_0.first_transpose = (1, 2, 0)

    #feat[0]
    classifier_header_shuffle_0_1 = network.add_shuffle(classifier_header_shuffle_0_0.get_output(0))
    classifier_header_shuffle_0_1.reshape_dims  = (1, 112, 112, 9, 90)
    classifier_header_shuffle_0_2 = network.add_shuffle(classifier_header_shuffle_0_1.get_output(0))
    classifier_header_shuffle_0_2.reshape_dims  = (1, -1, 90)
    classifier_feat0 = classifier_header_shuffle_0_2

    #-----------------------------------------------------------------------------------------------# p4_in
    #for0
    classifier_conv_list1_0_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p4_in.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1), post_padding = (1, 1), group = 160, lname = 'classifier.conv_list.0.depthwise_conv.conv')

    classifier_conv_list1_0_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_conv_list1_0_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'classifier.conv_list.0.pointwise_conv.conv')

    classifier_bn_list_1_0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = classifier_conv_list1_0_pointwise_conv.get_output(0), lname = 'classifier.bn_list.1.0')
    classifier_swish_1_0 = addSwish(network, inputTensor = classifier_bn_list_1_0.get_output(0))

    #for1
    classifier_conv_list1_1_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_swish_1_0.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1), post_padding = (1, 1), group = 160, lname = 'classifier.conv_list.1.depthwise_conv.conv')
    
    classifier_conv_list1_1_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_conv_list1_1_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'classifier.conv_list.1.pointwise_conv.conv')

    classifier_bn_list_1_1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = classifier_conv_list1_1_pointwise_conv.get_output(0), lname = 'classifier.bn_list.1.1')

    classifier_swish_1_1 = addSwish(network, inputTensor = classifier_bn_list_1_1.get_output(0))

    #for2
    classifier_conv_list1_2_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_swish_1_1.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1), post_padding = (1, 1), group = 160, lname = 'classifier.conv_list.2.depthwise_conv.conv')

    classifier_conv_list1_2_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_conv_list1_2_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'classifier.conv_list.2.pointwise_conv.conv')

    classifier_bn_list_1_2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = classifier_conv_list1_2_pointwise_conv.get_output(0), lname = 'classifier.bn_list.1.2')

    classifier_swish_1_2 = addSwish(network, inputTensor = classifier_bn_list_1_2.get_output(0))

    #for3
    classifier_conv_list1_3_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_swish_1_2.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1), post_padding = (1, 1), group = 160, lname = 'classifier.conv_list.3.depthwise_conv.conv')

    classifier_conv_list1_3_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_conv_list1_3_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'classifier.conv_list.3.pointwise_conv.conv')

    classifier_bn_list_1_3  = addBatchNorm2d(network = network,  weights = weights, inputTensor = classifier_conv_list1_3_pointwise_conv.get_output(0), lname = 'classifier.bn_list.1.3')

    classifier_swish_1_3 = addSwish(network, inputTensor = classifier_bn_list_1_3.get_output(0))

    
    #header
    classifier_header_depthwise_conv1 = convBlock(network = network, weights = weights, inputTensor = classifier_swish_1_3.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1), post_padding = (1, 1), group = 160, lname = 'classifier.header.depthwise_conv.conv')

    classifier_header_pointwise_conv1 = convBlock(network = network, weights = weights, inputTensor = classifier_header_depthwise_conv1.get_output(0), num_output_maps = 810, ksize = 1, stride = 1, group = 1, lname = 'classifier.header.pointwise_conv.conv')

    classifier_header_shuffle_1_0 = network.add_shuffle(classifier_header_pointwise_conv1.get_output(0))
    classifier_header_shuffle_1_0.first_transpose = (1, 2, 0)

    #feat[0]
    classifier_header_shuffle_1_1 = network.add_shuffle(classifier_header_shuffle_1_0.get_output(0))
    classifier_header_shuffle_1_1.reshape_dims  = (1, 56, 56, 9, 90)
    classifier_header_shuffle_1_2 = network.add_shuffle(classifier_header_shuffle_1_1.get_output(0))
    classifier_header_shuffle_1_2.reshape_dims  = (1, -1, 90)
    classifier_feat1 = classifier_header_shuffle_1_2

#-----------------------------------------------------------------------------------------------# p5_in
    #for0
    classifier_conv_list2_0_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p5_in.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1), post_padding = (1, 1), group = 160, lname = 'classifier.conv_list.0.depthwise_conv.conv')

    classifier_conv_list2_0_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_conv_list2_0_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'classifier.conv_list.0.pointwise_conv.conv')

    classifier_bn_list_2_0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = classifier_conv_list2_0_pointwise_conv.get_output(0), lname = 'classifier.bn_list.2.0')
    classifier_swish_2_0 = addSwish(network, inputTensor = classifier_bn_list_2_0.get_output(0))

    #for1
    classifier_conv_list2_1_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_swish_2_0.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1), post_padding = (1, 1), group = 160, lname = 'classifier.conv_list.1.depthwise_conv.conv')
    
    classifier_conv_list2_1_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_conv_list2_1_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'classifier.conv_list.1.pointwise_conv.conv')

    classifier_bn_list_2_1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = classifier_conv_list2_1_pointwise_conv.get_output(0), lname = 'classifier.bn_list.2.1')

    classifier_swish_2_1 = addSwish(network, inputTensor = classifier_bn_list_2_1.get_output(0))

    #for2
    classifier_conv_list2_2_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_swish_2_1.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1), post_padding = (1, 1), group = 160, lname = 'classifier.conv_list.2.depthwise_conv.conv')

    classifier_conv_list2_2_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_conv_list2_2_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'classifier.conv_list.2.pointwise_conv.conv')

    classifier_bn_list_2_2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = classifier_conv_list2_2_pointwise_conv.get_output(0), lname = 'classifier.bn_list.2.2')

    classifier_swish_2_2 = addSwish(network, inputTensor = classifier_bn_list_2_2.get_output(0))

    #for3
    classifier_conv_list2_3_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_swish_2_2.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1), post_padding = (1, 1), group = 160, lname = 'classifier.conv_list.3.depthwise_conv.conv')

    classifier_conv_list2_3_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_conv_list2_3_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'classifier.conv_list.3.pointwise_conv.conv')

    classifier_bn_list_2_3  = addBatchNorm2d(network = network,  weights = weights, inputTensor = classifier_conv_list2_3_pointwise_conv.get_output(0), lname = 'classifier.bn_list.2.3')

    classifier_swish_2_3 = addSwish(network, inputTensor = classifier_bn_list_2_3.get_output(0))

    #header
    classifier_header_depthwise_conv2 = convBlock(network = network, weights = weights, inputTensor = classifier_swish_2_3.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1), post_padding = (1, 1), group = 160, lname = 'classifier.header.depthwise_conv.conv')

    classifier_header_pointwise_conv2 = convBlock(network = network, weights = weights, inputTensor = classifier_header_depthwise_conv2.get_output(0), num_output_maps = 810, ksize = 1, stride = 1, group = 1, lname = 'classifier.header.pointwise_conv.conv')

    classifier_header_shuffle_2_0 = network.add_shuffle(classifier_header_pointwise_conv2.get_output(0))
    classifier_header_shuffle_2_0.first_transpose = (1, 2, 0)

    #feat[0]
    classifier_header_shuffle_2_1 = network.add_shuffle(classifier_header_shuffle_2_0.get_output(0))
    classifier_header_shuffle_2_1.reshape_dims  = (1, 28, 28, 9, 90)
    classifier_header_shuffle_2_2 = network.add_shuffle(classifier_header_shuffle_2_1.get_output(0))
    classifier_header_shuffle_2_2.reshape_dims  = (1, -1, 90)
    classifier_feat2 = classifier_header_shuffle_2_2

    #-----------------------------------------------------------------------------------------------# p6_in
    #for0
    classifier_conv_list3_0_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p6_in.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1), post_padding = (1, 1), group = 160, lname = 'classifier.conv_list.0.depthwise_conv.conv')

    classifier_conv_list3_0_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_conv_list3_0_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'classifier.conv_list.0.pointwise_conv.conv')

    classifier_bn_list_3_0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = classifier_conv_list3_0_pointwise_conv.get_output(0), lname = 'classifier.bn_list.3.0')
    classifier_swish_3_0 = addSwish(network, inputTensor = classifier_bn_list_3_0.get_output(0))

    #for1
    classifier_conv_list3_1_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_swish_3_0.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1), post_padding = (1, 1), group = 160, lname = 'classifier.conv_list.1.depthwise_conv.conv')
    
    classifier_conv_list3_1_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_conv_list3_1_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'classifier.conv_list.1.pointwise_conv.conv')

    classifier_bn_list_3_1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = classifier_conv_list3_1_pointwise_conv.get_output(0), lname = 'classifier.bn_list.3.1')

    classifier_swish_3_1 = addSwish(network, inputTensor = classifier_bn_list_3_1.get_output(0))

    #for2
    classifier_conv_list3_2_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_swish_3_1.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1), post_padding = (1, 1), group = 160, lname = 'classifier.conv_list.2.depthwise_conv.conv')

    classifier_conv_list3_2_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_conv_list3_2_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'classifier.conv_list.2.pointwise_conv.conv')

    classifier_bn_list_3_2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = classifier_conv_list3_2_pointwise_conv.get_output(0), lname = 'classifier.bn_list.3.2')

    classifier_swish_3_2 = addSwish(network, inputTensor = classifier_bn_list_3_2.get_output(0))

    #for3
    classifier_conv_list3_3_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_swish_3_2.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1), post_padding = (1, 1), group = 160, lname = 'classifier.conv_list.3.depthwise_conv.conv')

    classifier_conv_list3_3_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_conv_list3_3_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'classifier.conv_list.3.pointwise_conv.conv')

    classifier_bn_list_3_3  = addBatchNorm2d(network = network,  weights = weights, inputTensor = classifier_conv_list3_3_pointwise_conv.get_output(0), lname = 'classifier.bn_list.3.3')

    classifier_swish_3_3 = addSwish(network, inputTensor = classifier_bn_list_3_3.get_output(0))

    #header
    classifier_header_depthwise_conv3 = convBlock(network = network, weights = weights, inputTensor = classifier_swish_3_3.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1), post_padding = (1, 1), group = 160, lname = 'classifier.header.depthwise_conv.conv')

    classifier_header_pointwise_conv3 = convBlock(network = network, weights = weights, inputTensor = classifier_header_depthwise_conv3.get_output(0), num_output_maps = 810, ksize = 1, stride = 1, group = 1, lname = 'classifier.header.pointwise_conv.conv')

    classifier_header_shuffle_3_0 = network.add_shuffle(classifier_header_pointwise_conv3.get_output(0))
    classifier_header_shuffle_3_0.first_transpose = (1, 2, 0)

    #feat[0]
    classifier_header_shuffle_3_1 = network.add_shuffle(classifier_header_shuffle_3_0.get_output(0))
    classifier_header_shuffle_3_1.reshape_dims  = (1, 14, 14, 9, 90)
    classifier_header_shuffle_3_2 = network.add_shuffle(classifier_header_shuffle_3_1.get_output(0))
    classifier_header_shuffle_3_2.reshape_dims  = (1, -1, 90)
    classifier_feat3 = classifier_header_shuffle_3_2

    #-----------------------------------------------------------------------------------------------# p7_in
    #for0
    classifier_conv_list4_0_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = p7_in.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1), post_padding = (1, 1), group = 160, lname = 'classifier.conv_list.0.depthwise_conv.conv')

    classifier_conv_list4_0_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_conv_list4_0_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'classifier.conv_list.0.pointwise_conv.conv')

    classifier_bn_list_4_0  = addBatchNorm2d(network = network,  weights = weights, inputTensor = classifier_conv_list4_0_pointwise_conv.get_output(0), lname = 'classifier.bn_list.4.0')
    classifier_swish_4_0 = addSwish(network, inputTensor = classifier_bn_list_4_0.get_output(0))

    #for1
    classifier_conv_list4_1_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_swish_4_0.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1), post_padding = (1, 1), group = 160, lname = 'classifier.conv_list.1.depthwise_conv.conv')
    
    classifier_conv_list4_1_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_conv_list4_1_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'classifier.conv_list.1.pointwise_conv.conv')

    classifier_bn_list_4_1  = addBatchNorm2d(network = network,  weights = weights, inputTensor = classifier_conv_list4_1_pointwise_conv.get_output(0), lname = 'classifier.bn_list.4.1')

    classifier_swish_4_1 = addSwish(network, inputTensor = classifier_bn_list_4_1.get_output(0))

    #for2
    classifier_conv_list4_2_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_swish_4_1.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1), post_padding = (1, 1), group = 160, lname = 'classifier.conv_list.2.depthwise_conv.conv')

    classifier_conv_list4_2_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_conv_list4_2_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'classifier.conv_list.2.pointwise_conv.conv')

    classifier_bn_list_4_2  = addBatchNorm2d(network = network,  weights = weights, inputTensor = classifier_conv_list4_2_pointwise_conv.get_output(0), lname = 'classifier.bn_list.4.2')

    classifier_swish_4_2 = addSwish(network, inputTensor = classifier_bn_list_4_2.get_output(0))

    #for3
    classifier_conv_list4_3_depthwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_swish_4_2.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1), post_padding = (1, 1), group = 160, lname = 'classifier.conv_list.3.depthwise_conv.conv')

    classifier_conv_list4_3_pointwise_conv = convBlock(network = network, weights = weights, inputTensor = classifier_conv_list4_3_depthwise_conv.get_output(0), num_output_maps = 160, ksize = 1, stride = 1, group = 1, lname = 'classifier.conv_list.3.pointwise_conv.conv')

    classifier_bn_list_4_3  = addBatchNorm2d(network = network,  weights = weights, inputTensor = classifier_conv_list4_3_pointwise_conv.get_output(0), lname = 'classifier.bn_list.4.3')

    classifier_swish_4_3 = addSwish(network, inputTensor = classifier_bn_list_4_3.get_output(0))

    #header
    classifier_header_depthwise_conv4 = convBlock(network = network, weights = weights, inputTensor = classifier_swish_4_3.get_output(0), num_output_maps = 160, ksize = 3, stride = 1,  pre_padding = (1, 1), post_padding = (1, 1), group = 160, lname = 'classifier.header.depthwise_conv.conv')

    classifier_header_pointwise_conv4 = convBlock(network = network, weights = weights, inputTensor = classifier_header_depthwise_conv4.get_output(0), num_output_maps = 810, ksize = 1, stride = 1, group = 1, lname = 'classifier.header.pointwise_conv.conv')

    classifier_header_shuffle_4_0 = network.add_shuffle(classifier_header_pointwise_conv4.get_output(0))
    classifier_header_shuffle_4_0.first_transpose = (1, 2, 0)

    #feat[0]
    classifier_header_shuffle_4_1 = network.add_shuffle(classifier_header_shuffle_4_0.get_output(0))
    classifier_header_shuffle_4_1.reshape_dims  = (1, 7, 7, 9, 90)
    classifier_header_shuffle_4_2 = network.add_shuffle(classifier_header_shuffle_4_1.get_output(0))
    classifier_header_shuffle_4_2.reshape_dims  = (1, -1, 90)
    classifier_feat4 = classifier_header_shuffle_4_2

    classifier_feats = [classifier_feat0.get_output(0), classifier_feat1.get_output(0), classifier_feat2.get_output(0), classifier_feat3.get_output(0), classifier_feat4.get_output(0)]
    cat_classifier_feats = network.add_concatenation(classifier_feats)
    cat_classifier_feats.axis = 1

    sigmoid_classifier_feats = network.add_activation(cat_classifier_feats.get_output(0), trt.ActivationType.SIGMOID)

    return sigmoid_classifier_feats