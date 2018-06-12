from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from PIL import Image
import numpy as np
import sys


if __name__ == '__main__':
    X = np.array(Image.open(sys.argv[1])).astype(float)
    Y = np.array(Image.open(sys.argv[2])).astype(float)

    print('ssim: ', ssim(X, Y, multichannel=True))
    print('psnr: ', psnr(X, Y, data_range=255))
