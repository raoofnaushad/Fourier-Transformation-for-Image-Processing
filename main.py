import numpy as np
from utils import *
import filters
import cv2
import os

if __name__ == '__main__':

    img_path = "new_data/zebra.jpeg"
    
    fft_img = image_fft(img_path)

    # fft_image_filtered = fft_img * filters.gaussianLP(50,fft_img.shape)
    # fft_image_filtered = fft_img * filters.gaussianHP(50,fft_img.shape) 
    # fft_image_filtered = fft_img * filters.idealFilterLP(50,fft_img.shape)
    fft_image_filtered = fft_img * filters.idealFilterHP(50,fft_img.shape) 
    # fft_image_filtered = fft_img * filters.butterworthLP(50,fft_img.shape)
    # fft_image_filtered = fft_img * filters.butterworthHP(50,fft_img.shape) 

    fft_img = np.fft.ifftshift(fft_image_filtered)   

    image_fft_inverse(fft_img, img_path)

