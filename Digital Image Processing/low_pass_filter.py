import numpy as np
import cv2

import matplotlib.pyplot as plt


image = cv2.imread("a.tif", 0);
img = cv2.resize(image, (512,512))
height,width = img.shape

mean = 10
stddev = 25

noise = np.random.normal(mean, stddev, img.shape)
noise_img = cv2.add(img,noise.astype(np.uint8))
plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title(f"original image")

plt.subplot(222)
plt.imshow(noise_img, cmap='gray')
plt.title(f"noisy image")

F = np.fft.fftshift(np.fft.fft2(noise_img))
D0 = 25
n=2

def gaussian(F):
    M,N = F.shape
    H = np.zeros((M,N))

    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M//2)**2 + (v-N/2)**2)
            H[u,v] = np.exp(-D**2//(2*D0**2))

    new_img = H*F

    f_img = np.abs(np.fft.ifft2(new_img))

    return f_img

def butterworth(F):
    M,N = F.shape
    H = np.zeros((M,N))
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M//2)**2 + (v-N//2)**2)
            H[u,v] = 1/(1+(D/D0)**(2*n))

    new_img = H*F
    f_img = np.abs(np.fft.ifft2(new_img))
    return f_img





gaussian_filter_img = gaussian(F)
butterworth_filter_img = butterworth(F)

plt.subplot(223)
plt.imshow(gaussian_filter_img, cmap='gray')
plt.title(f"gaussian_LP_filter_img")

plt.subplot(224)
plt.imshow(butterworth_filter_img, cmap='gray')
plt.title(f"butterworth_LP_filter_img")
plt.axis('off')
plt.tight_layout()
plt.show()