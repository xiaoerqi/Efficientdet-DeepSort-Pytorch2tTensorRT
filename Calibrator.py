import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import sys
sys.path.append("../")
from utils.utils import  preprocessCalib

class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibCount, inputShape, calibDataPath, cacheFile):
        trt.IInt8EntropyCalibrator2.__init__(self)                                              # 基类默认构造函数
        self.calibCount     = calibCount
        self.shape          = inputShape
        self.calibDataSet   = self.laodData(calibDataPath)                                      # 需要自己实现一个读数据的函数
        self.cacheFile      = cacheFile
        self.calibData      = np.zeros(self.shape, dtype=np.float32)
        self.dIn            = cuda.mem_alloc(trt.volume(self.shape) * trt.float32.itemsize)     # 准备好校正用的设备内存
        self.oneBatch       = self.batchGenerator()
        self.batch_size=self.shape[0]

    def batchGenerator(self):                                                                   # calibrator 的核心，一个提供数据的生成器
        for i in range(self.calibCount):
            print("> calibration ", i)
            self.calibDataPath = np.random.choice(self.calibDataSet, self.shape[0], replace=False)  # 随机选取数据
            data=preprocessCalib(self.calibDataPath)
            yield np.ascontiguousarray(data, dtype=np.float32)                        # 调整数据格式后抛出

    def get_batch_size(self):                                                                   # TensorRT 会调用，不能改函数名
        return self.shape[0]
    def laodData(self,DataPath):
        files=os.listdir(DataPath)
        all_path=[os.path.join(DataPath,file) for file in files]

        return all_path
    def get_batch(self, names):                                                                 # TensorRT 会调用，不能改函数名，老版本 TensorRT 的输入参数个数可能不一样
        try:
            data = next(self.oneBatch)                                                          # 生成下一组校正数据，拷贝到设备并返回设备地址，否则退出
            cuda.memcpy_htod(self.dIn, data)
            return [int(self.dIn)]
        except StopIteration:
            return None

    def read_calibration_cache(self):                                                           # TensorRT 会调用，不能改函数名
        if os.path.exists(self.cacheFile):
            print( "cahce file: %s" %(self.cacheFile) )
            f = open(self.cacheFile, "rb")
            cache = f.read()
            f.close()
            return cache

    def write_calibration_cache(self, cache):                                                   # TensorRT 会调用，不能改函数名
        print( "cahce file: %s" %(self.cacheFile) )
        f = open(self.cacheFile, "wb")
        f.write(cache)
        f.close()
# calib=MyCalibrator(100,(1,896,896,3),"data/bingqiu","cache.txt")
# batch=calib.get_batch("")
# print(batch)
