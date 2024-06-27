
import numpy
import SimpleITK as sitk
import ReadCT
#import SampleSourcePDF
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

        return result

def run(image_dir):

    CT_image = sitk.ReadImage(image_dir)

    print(CT_image.GetSize())
    print(CT_image.GetSpacing())
    size = CT_image.GetSize()

    # 将图像转换为NumPy数组
    image_array = sitk.GetArrayFromImage(CT_image)
    cropped = ReadCT.get_cropped_array(image_array)
    cropped_img = ReadCT.array_to_simpleitk_image(cropped, CT_image.GetSpacing())
    cropped_img = ReadCT.resize_image_itk(cropped_img, [128, 128, 128])

    return cropped_img



if __name__ == "__main__":

    CT_dir = '../CT_ori'
    CT_files = os.listdir(CT_dir)

    for i in range(5):
        CT_img = CT_dir + '/' + CT_files[i]
        HU = run(CT_img)

        sitk.WriteImage(HU,CT_files[i],useCompression=True)








