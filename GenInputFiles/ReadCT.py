
import SimpleITK as sitk
import numpy
import matplotlib.pyplot as plt

def array_to_simpleitk_image(array_image, spacing):
    data = array_image
    direction = [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0]  #to keep the same with CT image
    origin = [0.0, 0.0, 0.0]  #origin of our dataset
    itk_image = sitk.GetImageFromArray(data, isVector=False)
    itk_image.SetSpacing(spacing)
    itk_image.SetDirection(direction)
    itk_image.SetOrigin(origin)
    return itk_image


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkLinear):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    originSpacing = itkimage.GetSpacing()
    newSize = numpy.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(numpy.int32)
    resampler.SetReferenceImage(itkimage)
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled

def get_cropped_array(image_array):
    middle_slices = image_array[80: 80 + 256, :, :]
    indices = numpy.argwhere(middle_slices > -300)
    W_indices = indices[:, 1]
    H_indices = indices[:, 2]
    max_W = numpy.max(W_indices)
    min_W = numpy.min(W_indices)
    max_H = numpy.max(H_indices)
    min_H = numpy.min(H_indices)
    center_W = (max_W + min_W) // 2
    center_H = (max_H + min_H) // 2

    start_W = max(center_W - 128, 0)
    start_H = max(center_H - 128, 0)

    end_W = start_W + 256
    end_H = start_H + 256

    if end_W > 390:
        end_W = 390
        start_W = 390 - 256

    if end_H > 390:
        end_H = 390
        start_H = 390 - 256

    cropped = middle_slices[:, start_W:end_W, start_H:end_H]
    return cropped


if __name__ == "__main__":
    image_dir = 'CT_output/s0040.nii.gz'
    CT_image = sitk.ReadImage(image_dir)

    print(CT_image.GetSize())
    print(CT_image.GetSpacing())
    size = CT_image.GetSize()

    # 将图像转换为NumPy数组
    image_array = sitk.GetArrayFromImage(CT_image)
    cropped = get_cropped_array(image_array)
    cropped_img = array_to_simpleitk_image(cropped, CT_image.GetSpacing())
    cropped_img = resize_image_itk(cropped_img, [128, 128, 128])
    sitk.WriteImage(cropped_img, 'test.nii.gz', useCompression=True)





