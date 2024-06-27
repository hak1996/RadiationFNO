
import matplotlib.pyplot as plt
import SampleSourcePDF
import numpy
import os

class Converter():
    def __init__(self):
        HU2D = numpy.loadtxt("HU2DENSITY.txt")
        self.HU_bin = HU2D[:,0] + 24
        self.Den_bin = HU2D[:,1]
        self.HU2Mat = numpy.loadtxt('HU2Mat.txt')

    def Convert2Density(self, HU):
        return numpy.interp(HU, self.HU_bin, self.Den_bin).astype(dtype=numpy.float32)

    def Convert2Material(self, HU):
        result = numpy.zeros_like(HU, dtype=numpy.uint8)
        for i in range(len(self.HU2Mat)):
            if i == 0:
                cond = (HU<=self.HU2Mat[i])
            else:
                cond = (HU<=self.HU2Mat[i]) & (HU>self.HU2Mat[i-1])
            result[cond] = i+2

        right = self.HU2Mat[-2]

        result[HU>=right] = 24

        return result

def GenHU(sigma, height, cx, cy, cz):
    half_width = 64
    x = numpy.arange(-half_width, half_width)+0.5
    x_mesh, y_mesh, z_mesh = numpy.meshgrid(x, x, x, indexing='xy')
    r = numpy.sqrt((x_mesh-cx)**2  + (y_mesh-cy)**2  + (z_mesh-cz)**2)
    kernel = numpy.exp(-0.5 * r * r / sigma / sigma) * height

    HU = kernel*1000.0 - 1000.0
    return HU

def run():
    height = 2.8 + numpy.random.randn(1)*0.2
    sigma = numpy.random.rand(1)*29.0 + 1.0
    cx = numpy.random.rand(1)*60.0-30.0
    cy = numpy.random.rand(1) * 60.0 - 30.0
    cz = numpy.random.rand(1) * 60.0 - 30.0
    HU = GenHU(sigma, height,cx,cy,cz)

    print(numpy.max(HU))

    max_sigma = 15.0
    min_sigma = 1.0
    max_heihgt = 1.0
    min_height = 0.1

    activity = SampleSourcePDF.gen_source(max_sigma, min_sigma, max_heihgt, min_height, 10, HU)

    convert = Converter()

    den = convert.Convert2Density(HU)
    mat = convert.Convert2Material(HU)
    return den, mat, activity

if __name__ == "__main__":
    numpy.random.seed(123456897)

    for i in range(0,500):

        if i>-1:
            den, mat, act = run()

            output_dir = f"../RBF_inputs/data_{i}"

            den.tofile(output_dir + "/density.raw")
            mat.tofile(output_dir + "/mat.raw")
            act.tofile(output_dir + "/act.raw")




