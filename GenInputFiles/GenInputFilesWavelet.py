import numpy
import SimpleITK as sitk
import ReadCT
import SampleSourcePDF
import CTFusion
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

        result[HU > right] = 24

        return result

def run(ct1, ct2):
    HU = CTFusion.FuseCT(ct1,ct2)
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

    num_dataset = 120
    numpy.random.seed(123456)

    CT_dir = '../CT_cropped'
    CT_files = os.listdir(CT_dir)

    ct_array = []
    for ct_f in CT_files:
        CT_fname = CT_dir + '/' + ct_f
        ct_img = CTFusion.getCT(CT_fname)
        ct_array.append(ct_img)


    for i in range(num_dataset):
        choice = numpy.random.choice(range(5),2,replace=False)
        print(i)
        ct1 = ct_array[choice[0]]
        ct2 = ct_array[choice[1]]
        den, mat, act = run(ct1,ct2)

        output_dir = f"../inputfiles/data_{i}"
        #os.mkdir(output_dir)

        den.tofile(output_dir + "/density.raw")
        mat.tofile(output_dir + "/mat.raw")
        act.tofile(output_dir + "/act.raw")