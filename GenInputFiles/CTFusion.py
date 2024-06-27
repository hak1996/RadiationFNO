import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

import pywt

def getCT(name):
    itk_img = sitk.ReadImage(name)
    original = sitk.GetArrayFromImage(itk_img).astype(np.float32)
    original = (original + 1024.0) / 1000.0
    return original

def FuseCT(ct1, ct2):
    coeffs1 = pywt.wavedecn(ct1, 'db4', level=4)
    coeffs2 = pywt.wavedecn(ct2, 'db4', level=4)

    dict_keys = coeffs1[1].keys()

    coeffs_new = []

    cfs1 = coeffs1[0]
    cfs2 = coeffs2[0]
    std1 = np.sqrt(np.abs(cfs2))
    std2 = np.sqrt(np.abs(cfs2))
    shape = np.shape(cfs1)
    noise1 = np.random.randn(shape[0], shape[1], shape[2])
    noise2 = np.random.randn(shape[0], shape[1], shape[2])
    noise1 *= std1*2.0
    noise2 *= std2*2.0
    cfs1 += noise1
    cfs2 += noise2

    w1 = np.random.rand(1)
    w2 = 1 - w1
    cfs = w1 * cfs1 + w2 * cfs2
    coeffs_new.append(cfs)
    for i in range(1, 5):
        cfs_dict = {}
        for key in dict_keys:
            cfs1 = coeffs1[i][key]
            cfs2 = coeffs2[i][key]
            std1 = np.sqrt(np.abs(cfs2))
            std2 = np.sqrt(np.abs(cfs2))
            shape = np.shape(cfs1)
            noise1 = np.random.randn(shape[0], shape[1], shape[2])
            noise2 = np.random.randn(shape[0], shape[1], shape[2])
            if i < 3:
                noise1 *= std1
                noise2 *= std2
                cfs1 += noise1
                cfs2 += noise2

            w1 = np.random.rand(1)
            w2 = 1 - w1
            cfs = w1 * cfs1 + w2 * cfs2
            # cfs[:,:,:]=0.0
            # cfs[abs(cfs)<thres] = 0.0

            cfs_dict[key] = cfs

        coeffs_new.append(cfs_dict)

    new = pywt.waverecn(coeffs_new, 'db4')
    new = new*1000.0-1024.0

    #plt.imshow(new[:, :, 40])
    #plt.show()

    return new





if __name__ == "__main__":
    np.random.seed(123456)
    print(pywt.wavelist(kind='discrete'))
    ct1 = getCT('s0011.nii.gz')
    ct2 = getCT('s0476.nii.gz')
    new = FuseCT(ct1,ct2)
    plt.imshow(new[:,:,40])
    plt.show()