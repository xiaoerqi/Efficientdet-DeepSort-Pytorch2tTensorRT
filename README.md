## Background Introduction 

The purpose of this project is to provide a faster target tracking system. We choose EfficientDet and DeepSort as the project algorithms. We use TensorRT API to convert the Pytorch models of EfficientDet and DeepSort into TensorRT for acceleration. The pytorch source code of this project comes from [Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) and [Yolov5_DeepSort_Pytorch](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch).**Our motivation for completing this project is the TensorRT Hackthon competition held by NVIDIA and Alibaba. Thanks very much for the training and hardware environment provided by NVIDIA and Alibaba.**

## Now Our Work
- [x] EfficientDet-D3 use TenorRT API conversion model
- [x] EfficientDet-D3 int8 quantization
- [x] deepsort use onnx conversion model
- [ ] deepsort use int8 quantization
- [ ] cpp demo

Later we will complete the conversion of all EfficientDet-D0～D7 models 
## Test Results 
<details>
  <summary>Figure Notes (click to expand)</summary>
  
  * GPU Speed measures model process time per image averaged 1000 images using a 1080ti GPU,  not includes image preprocessing, postprocessing
  * TensorRT version 7.2.3.4
  * **Reproduce** by `python effientdet_trt_test.py --img_path test/images/img.png --engine_file_path tensorrt_engine/efficientdet.engine --batch_size=1 `
</details>

## EfficientDet-D3 Performance
Model |Batchsize<br><sup>(1) |Latency<sup><br>(ms)|Throughput<sup><br>1000/latency*batchsize) |Latency Speedup<sup><br>(TRT latency / original latency) |Throughput speedup<br><sup>(TRT throughput / original thoughput) 
---   |---  |---        |---         |---             |---                       
PyTorch   |1  |-     |-     |-     |- 
PyTorch   |4  |-     |-     |-     |-     
PyTorch   |8  |-     |-     |-     |-     
PyTorch   |16  |- |- |- |-     
| | | | | | || 
TensorRT  |1 |-     |-     |-     |- 
TensorRT  |4 |-     |-     |-     |-     
TensorRT  |8 |-     |-     |-     |-   
TensorRT  |16 |- |- |- |-    
| | | | | | 

Model |Latency<br><sup>(fp32, ms) |Latency<sup>val<br>(fp16, ms)|Latency<sup>val<br>(int8, ms)
---   |---  |--- |---       
PyTorch   |-  |-   |  -
TensorRT  | 38 |35   |  27

## Environments

This project may be run in any of the following up-to-date verified environments (with all dependencies including [CUDA](https://developer.nvidia.com/cuda)/[CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/) and [PyTorch](https://pytorch.org/) preinstalled):

- **Operating System** Our test code runs on Ubuntu 20.04.1 LTS,We think it also can run normally on Ubuntu 18.04 
- **CUDA** Our NVIDIA driver version is 455.23.05 and the CUDA version is 455.23.05

## Requirements

* Python 3.8 or later with all [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) dependencies installed, including `torch>=1.7`. To run the following :
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->
We recommend using conda to create a virtual environment:
```bash
$ conda create -n pytorch2trt python=3.8
$ conda activate pytorch2trt
```
Update your pip and setuptools:
```bash
$ pip install --upgrade setuptools pip
```
Then install the requirements:
```bash
$ pip install -r requirements.txt
```


## How to run ？
First you must download the Yet-Another-EfficientDet-Pytorch project to your own directory:
```bash
$ git clone https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch.git & cd Yet-Another-EfficientDet-Pytorch 
```
Then download our project:
```bash
$ git clone & cd EfficientDet-DeepSort-Pytorch-TensorRT
```
## Conversion EfficientDet model 

`conver2trt.py` You can convert the pytorch model of EfficientDet to a tensorrt model, and you can get the engine file through it , downloading EfficientDet-D3 models automatically from the [EfficientDet-D3](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d3.pth) and put it in `weights`.
```bash
$ python conver2trt.py --weight_path weights   # pytorch models path       
                       --engine_file_path tensorrt_engine # save path
                       --precision fp32 #precision
                       --batch_size 1 #batch_size
                       
```
To int8 quantify, you must download the [BaiduYun, l5dp](https://pan.baidu.com/s/1yZYYXgKd0r5Au6wMO0zJyg data, put it in the data/tensorrtx-int8calib-data/coco_calib folder, Then:
```bash
$ python conver2trt.py --weight_path weights   # pytorch models path       
                       --engine_file_path tensorrt_engine # save path
                       --precision int8 #precision
                       --batch_size 1 #batch_size
                       
```

## EfficientDet Test
To run EfficientDet Test on example images in `test/images`:
```bash
$ python effientdet_trt_test.py --img_path test/images/img.png 
                                --engine_file_path tensorrt_engine/efficientdet.engine
                                --batch_size 1
```
<img width="500" src="/test/img_inferred_d3_this_repo_0.jpg">  

## Conversion DeepSort model 
`conver2onnx.py` You can convert the pytorch model of DeepSort to a tensorrt model, and you can get the engine file through it .
```bash
$ conver2onnx.py 
$ trtexec --explicitBatch --onnx=deepsort.onnx --saveEngine=deepsort.trt 
```
## DeepSort Test


## Contact

**Issues should be raised directly in the repository.** For business inquiries or professional support requests please email ZhangQi at xiaoer_qi@live.com. 
