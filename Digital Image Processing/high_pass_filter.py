import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('a.tif',0)
img = cv2.resize(image,(512,512))

mean = 10
stddev = 5
noise = np.random.normal(mean, stddev, img.shape)

noise_img = np.add(img,noise.astype(np.uint8))

D0 = 15
F = np.fft.fftshift(np.fft.fft2(noise_img)) #describe
def gaussian_hpf(F):
        M,N = F.shape
        H = np.zeros((M,N))
        for u in range(M):
                for v in range(N):
                        D = np.sqrt((u-M//2)**2 + (v-N//2)**2)
                        H[u,v] = 1-np.exp(-D**2/(2*D0)**2)

        new_img = H*F
        f_img = np.abs(np.fft.ifft2(new_img))
        return f_img

n=2                       
def butter_hf(F):
        M,N = F.shape
        H = np.zeros((M,N))
        for u in range(M):
                for v in range(N):
                        D = np.sqrt((u-M//2)**2 + (v-N//2)**2)
                        H[u,v] = 1 - (1/(1+ (D/D0)**(2*n)))

        new_img = H*F
        f_img = np.abs(np.fft.ifft2(new_img))

        return f_img

g_hf = gaussian_hpf(F)
b_hf = butter_hf(F)
plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('original image')

plt.subplot(222)
plt.imshow(g_hf, cmap='gray')
plt.title(f'gaussian high pass filter image with D0= {D0}')

plt.subplot(223)
plt.imshow(b_hf, cmap='gray')
plt.title(f"butterworth high pass filter image with D0= {D0}")
plt.axis('off')
plt.tight_layout()
plt.show()