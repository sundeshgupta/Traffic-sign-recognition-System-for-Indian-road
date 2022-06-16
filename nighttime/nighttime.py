import os
import sys
import cv2
import numpy as np
# from guidedfilter import *
# from gf import *
import shutil
import sys
from numpy.lib.stride_tricks import as_strided

from filter import GuidedFilter

# utility function to show an image
def show(img, title = "Image"):
    cv2.imshow(title, img)
    cv2.waitKey(0)

def get_illumination_channel_new(A, w):

    stride = 1
    kernel_size = w
    # Padding
    A = np.pad(A, ((int(w/2), int(w/2)), (int(w/2), int(w/2)), (0, 0)), 'edge')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size) // stride + 1,
      (A.shape[1] - kernel_size) // stride + 1)

    shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size) 
    strides_w = (stride * A.strides[0], stride * A.strides[1], A.strides[0], A.strides[1])

    A_w = as_strided(A, shape_w, strides_w)
    
    return A_w.min(axis = (2, 3)).astype('float64'), A_w.max(axis = (2, 3)).astype('float64')

def get_illumination_channel(I, w):
    M, N, _ = I.shape
    # padding for channels
    padded = np.pad(I, ((int(w/2), int(w/2)), (int(w/2), int(w/2)), (0, 0)), 'edge')
    darkch = np.zeros((M, N))
    brightch = np.zeros((M, N))

    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :]) # dark channel
        brightch[i, j] = np.max(padded[i:i + w, j:j + w, :]) # bright channel

    return darkch, brightch

def get_atmosphere(I, brightch, p=0.1):
    M, N = brightch.shape
    flatI = I.reshape(M*N, 3) # reshaping image array
    flatbright = brightch.ravel() #flattening image array

    searchidx = (-flatbright).argsort()[:int(M*N*p)] # sorting and slicing
    A = np.mean(flatI.take(searchidx, axis=0), dtype=np.float64, axis=0)
    return A

def get_initial_transmission(A, brightch):
    A_c = np.max(A)
    init_t = (brightch-A_c)/(1.-A_c) # finding initial transmission map
    return (init_t - np.min(init_t))/(np.max(init_t) - np.min(init_t)) # normalized initial transmission map

def get_corrected_transmission(I, A, darkch, brightch, init_t, alpha, omega, w):
    im = np.empty(I.shape, I.dtype);
    for ind in range(0, 3):
        im[:, :, ind] = I[:, :, ind] / A[ind] #divide pixel values by atmospheric light
    dark_c, _ = get_illumination_channel_new(im, w) # dark channel transmission map
    dark_t = 1 - omega*dark_c # corrected dark transmission map
    corrected_t = init_t # initializing corrected transmission map with initial transmission map
    diffch = brightch - darkch # difference between transmission maps

    # for i in range(diffch.shape[0]):
    #     for j in range(diffch.shape[1]):
    #         if(diffch[i, j] < alpha):
    #             corrected_t[i, j] = dark_t[i, j] * init_t[i, j]

    corrected_t[diffch < alpha] = dark_t[diffch < alpha] * init_t[diffch < alpha]

    return np.abs(corrected_t)

def get_final_image(I, A, refined_t, tmin):
    refined_t_broadcasted = np.broadcast_to(refined_t[:, :, None], (refined_t.shape[0], refined_t.shape[1], 3)) # duplicating the channel of 2D refined map to 3 channels
    J = (I-A) / (np.where(refined_t_broadcasted < tmin, tmin, refined_t_broadcasted)) + A # finding result 

    return (J - np.min(J))/(np.max(J) - np.min(J)) # normalized image

def reduce_init_t(init_t):
    init_t = (init_t*255).astype(np.uint8) 
    xp = [0, 32, 255]
    fp = [0, 32, 48]
    x = np.arange(256) # creating array [0,...,255]
    table = np.interp(x, xp, fp).astype('uint8') # interpreting fp according to xp in range of x
    init_t = cv2.LUT(init_t, table) # lookup table
    init_t = init_t.astype(np.float64)/255 # normalizing the transmission map
    return init_t

# def guideFilter(I, p, winSize, eps):
    
#     print(winSize)
#     #I's mean smoothing
#     mean_I = cv2.blur(I, winSize)
    
#     #p's mean smoothing
#     mean_p = cv2.blur(p, winSize)
    
#     #I*I and I*p mean smoothing
#     mean_II = cv2.blur(I*I, winSize)
    
#     mean_Ip = cv2.blur(I*p, winSize)
    
#     #variance
#     var_I = mean_II-mean_I * mean_I #variance formula
    
#     #Covariance
#     cov_Ip = mean_Ip-mean_I * mean_p
   
#     a = cov_Ip/(var_I + eps)
#     b = mean_p-a*mean_I
    
#     #Smooth the mean of a and b
#     mean_a = cv2.blur(a, winSize)
#     mean_b = cv2.blur(b, winSize)
    
#     q = mean_a*I + mean_b
    
#     return q
    
def dehaze(I, tmin=0.1, w=15, alpha=0.4, omega=0.75, p=0.1, eps=1e-3, reduce=False):
    I = np.asarray(I, dtype=np.float64) # Convert the input to a float array.
    I = I[:, :, :3] / 255
    m, n, _ = I.shape
    Idark, Ibright = get_illumination_channel_new(I, w)
    A = get_atmosphere(I, Ibright, p)

    init_t = get_initial_transmission(A, Ibright) 
    if reduce:
        init_t = reduce_init_t(init_t)
    corrected_t = get_corrected_transmission(I, A, Idark, Ibright, init_t, alpha, omega, w)

    normI = (I - I.min()) / (I.max() - I.min())
    # refined_t = guided_filter(normI, corrected_t, w, eps) # applying guided filter
    # refined_t = guideFilter(normI, corrected_t, w, eps) # applying guided filter

    GF = GuidedFilter(normI, w, eps)
    refined_t = GF.filter(corrected_t)

    J_refined = get_final_image(I, A, refined_t, tmin)
    
    enhanced = (J_refined*255).astype(np.uint8)
    f_enhanced = cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)
    f_enhanced = cv2.edgePreservingFilter(f_enhanced, flags=1, sigma_s=64, sigma_r=0.2)
    return f_enhanced

import time
if __name__ == '__main__':



    DATA_DIR = "../ITSD/"
    # for filename in os.listdir(DATA_DIR):
    #     break
    #     if (os.path.splitext(filename)[-1]).lower() == '.jpg':
    #         img_path = os.path.join(DATA_DIR, filename)
    #         img = cv2.imread(img_path)
    #         # img = cv2.resize(img, (256, 256))
    #         # show(img, "original")
    #         enhanced = dehaze(img, reduce = True)
    #         # show(enhanced, "enhanced")
    #         cv2.imwrite('../nighttime_itsd/' + filename, enhanced)
    #         # cv2.destroyAllWindows()
    # img_path = './Datacluster Traffic Sign (128).jpg'
    # img_path = '00017.ppm'
    filename = sys.argv[1]
    # img_path = os.path.join(DATA_DIR, filename)
    img_path = filename
    img = cv2.imread(img_path)
    # img = cv2.resize(img, (img.shape[1], img.shape[0]))
    # print(img.shape)
    # # show(img, "original")
    enhanced = dehaze(img, reduce = True)
    # # show(enhanced, "enhanced")
    cv2.imwrite('../nighttime_itsd/' + filename, enhanced)

    # cv2.imwrite('./enhanced_new2.jpg', enhanced)
    # cv2.destroyAllWindows()

    # test_annotation = open('./test_annotation.txt', 'r')
    # SAVE_DIR = '/home/anuj/Desktop/hdd1/swastik/darknet/MTSD_data/MTSD/test_images'
    # cnt = 0
    # for filename in test_annotation.readlines():
    #     filename = filename[:-1]
    #     img_path = os.path.normpath(filename)
    #     img_name = os.path.splitext(os.path.split(img_path)[1])[0] + '.jpg'
    #     txt_name = os.path.splitext(os.path.split(img_path)[1])[0] + '.txt'
    #     txt_path = ''.join(os.path.splitext(img_path)[:-1]) + '.txt'
    #     shutil.copyfile(txt_path, os.path.join(SAVE_DIR, txt_name))
    #     img = cv2.imread(img_path)
    #     enhanced = dehaze(img, reduce = True)
    #     cv2.imwrite(os.path.join(SAVE_DIR, img_name), enhanced)
    #     cnt += 1

    # test_annotation.close()


