import numpy as np
import cv2
import os

from fourier_transform import *


def img_FFT(img):
    if len(img.shape) == 2 or len(img.shape) == 3:
        return FFT_2D(img)
    else:
        raise ValueError("Please input a gray or RGB image!")


def img_FFT_inverse(img):
    if len(img.shape) == 2 or len(img.shape) == 3:
        return inverseFFT_2D(img)
    else:
        raise ValueError("Please input a gray or RGB image!")


def findpower2(num):
    """find the nearest number that is the power of 2"""
    if num & (num-1) == 0:
        return num

    bin_num = bin(num)
    origin_bin_num = str(bin_num)[2:]
    near_power2 = pow(10, len(origin_bin_num))
    near_power2 = "0b" + str(near_power2)
    near_power2 = int(near_power2, base=2)

    return near_power2


def image_padding(img):
    """ padding the image size to power of 2, for fft computation requirement"""
    if len(img.shape) == 2:
        h, w = img.shape[0], img.shape[1]
        h_pad = findpower2(h)-h
        w_pad = findpower2(w)-w
        img = np.pad(img, pad_width=((0, h_pad), (0, w_pad)), mode='constant')
        return img
    elif len(img.shape) == 3:
        h, w = img.shape[0], img.shape[1]
        h_pad = findpower2(h)-h
        w_pad = findpower2(w)-w
        img = np.pad(img, pad_width=((0, h_pad), (0, w_pad), (0, 0)), mode='constant')
        return img


def image_fft(img_path, result_folder_path="result/"):
    """ read, padding, fft, cut to origin size and save """
    data_root, img_name = os.path.split(img_path)

    if img_name[-3:] != "png" and img_name[-3:] != "tif" \
        and img_name[-4:] != "jpeg" and img_name[-3:] != "jpg":
        return 0

    if not os.path.exists(result_folder_path):
        os.mkdir(result_folder_path)

    img_origin = cv2.imread(img_path, 0)
    img = image_padding(img_origin)

    img_fft = img_FFT(img)

    
    if len(img_origin.shape) == 2:
        img_fft = img_fft[:img_origin.shape[0], :img_origin.shape[1]]
    else:
        img_fft = img_fft[:img_origin.shape[0], :img_origin.shape[1], :]

    img_fft_complex = img_fft.copy()

    img_fft = np.fft.fftshift(img_fft_complex)
    
    # save real value for human seeing
    img_fft = np.real(img_fft)
    name, _ = img_name.split(".")
    save_img_name = result_folder_path + name + "_fft.png"
    cv2.imwrite(save_img_name, img_fft)

    

    return img_fft_complex


def image_fft_inverse(img_fft_complex, img_path, result_folder_path="result/"):
    """ inverse the read fft_img, cut to origin size and save """
    if not os.path.exists(result_folder_path):
        os.mkdir(result_folder_path)
    _, img_name = os.path.split(img_path)

    img_fft = image_padding(img_fft_complex)

    img_origin = img_FFT_inverse(img_fft)
    img_ifft = np.real(img_origin)

    name, _ = img_name.split(".")
    save_img_name = result_folder_path + name + "_inverse.png"

    if len(img_origin.shape) == 2:
        img_ifft = img_ifft[:img_fft_complex.shape[0], :img_fft_complex.shape[1]]
    else:
        img_ifft = img_ifft[:img_fft_complex.shape[0], :img_fft_complex.shape[1], :]

    cv2.imwrite(save_img_name, img_ifft)

    return img_origin


if __name__ == '__main__':
    x = np.mgrid[:8, :8][0]
    # x = np.mgrid[:4, :4][0]
    print(x)
    print("-------------------")
    # print(np.allclose(FFT(x), np.fft.fft(x)))
    # print(np.allclose(FFT_2D(x), np.fft.fft2(x)))
    # print(FFT_2D(x))
    print(inverseFFT_2D(x))
    # print(inverseDFT_2D(x))

    print("-------------------")
    print(np.fft.ifft2(x))
    print("-------------------")
    # print(np.fft.fft(x))

    # print(np.allclose(np.fft.fft(x), np.fft.fft2(x)))
