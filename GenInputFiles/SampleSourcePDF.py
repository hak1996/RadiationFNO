
from ReadCT import array_to_simpleitk_image, resize_image_itk, get_cropped_array

import numpy
import SimpleITK as sitk

numpy.random.seed(123)

max_size_img = 128

def get_gaussian_kernel(sigma, height): #3 sigma
    half_width = 3*numpy.ceil(sigma).astype(numpy.int32)[0]
    x = numpy.arange(-half_width,half_width+1)
    x_mesh,y_mesh,z_mesh = numpy.meshgrid(x,x,x,indexing='xy')
    r = numpy.sqrt(x_mesh*x_mesh + y_mesh*y_mesh + z_mesh*z_mesh)
    kernel = numpy.exp(-0.5*r*r/sigma/sigma)*height
    return kernel, half_width

def gen_source(max_sigma, min_sigma, max_height, min_height, n_points, image_array):
    indices = numpy.argwhere(image_array > -800)
    indices_chosen = numpy.random.choice(range(len(indices)), size=n_points)
    activity = numpy.zeros_like(image_array, dtype=numpy.double)
    for index in indices_chosen:
        sigma = min_sigma + (max_sigma - min_sigma) * numpy.random.rand(1)
        height = min_height + (max_height - min_height) * numpy.random.rand(1)
        kernel, half_width = get_gaussian_kernel(sigma, height)
        indice = indices[index]

        voxel_in_body = numpy.zeros_like(image_array, dtype=numpy.double)
        voxel_in_body[image_array > -800] = 1.0

        activity_tmp = numpy.zeros_like(image_array, dtype=numpy.double)
        min_W = indice[1] - half_width
        min_H = indice[2] - half_width
        min_D = indice[0] - half_width

        max_W = indice[1] + half_width + 1
        max_H = indice[2] + half_width + 1
        max_D = indice[0] + half_width + 1

        start_W = min_W
        if min_W < 0:
            start_W = 0
            kernel = kernel[:, :, -min_W:]
        end_W = max_W
        if max_W > max_size_img:
            end_W = max_size_img
            kernel = kernel[:, :, :end_W - max_W]

        start_H = min_H
        if min_H < 0:
            start_H = 0
            kernel = kernel[:, -min_H:, :]
        end_H = max_H
        if max_H > max_size_img:
            end_H = max_size_img
            kernel = kernel[:, :end_H - max_H, :]

        start_D = min_D
        if min_D < 0:
            start_D = 0
            kernel = kernel[-min_D:, :, :]
        end_D = max_D
        if max_D > max_size_img:
            end_D = max_size_img
            kernel = kernel[:end_D - max_D, :, :]

        activity_tmp[start_D:end_D, start_H:end_H, start_W:end_W] = kernel
        activity += activity_tmp
    activity *= voxel_in_body
    return activity

if __name__ == "__main__":
    image_dir = 'test.nii.gz'
    CT_image = sitk.ReadImage(image_dir)
    image_array = sitk.GetArrayFromImage(CT_image)
    max_sigma = 15.0
    min_sigma = 1.0
    max_heihgt = 1.0
    min_height = 0.1

    activity = gen_source(max_sigma, min_sigma, max_heihgt, min_height, 7, image_array)

    a_img = array_to_simpleitk_image(activity, CT_image.GetSpacing())

    sitk.WriteImage(a_img, 'act.nii.gz', useCompression=True)





