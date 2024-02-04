import cv2
import numpy as np 
import matplotlib.pyplot as plt 


image = cv2.imread('a.tif',0)
img = cv2.resize(image,(512,512))
ht,wd = img.shape

def salt_pepper_noise(img):
    n_img = np.copy(img)
    noise = 0.01

    for i in range(ht):
        for j in range(wd):
            add_n = np.random.rand()

            if(add_n<noise/2):
                n_img[i,j] = 1
            elif(add_n<noise):
                n_img[i,j] = 0

    return n_img

def harmonic_geometric_f(img):
    n=3 #mask size
    geo_m = np.zeros_like(img)
    her_m = np.zeros_like(img)
    pad_img = np.pad(img, n//2)
    h,w = img.shape
    for i in range(h):
        for j in range(w):
            tmp_window = pad_img[i:i+n, j:j+n]
            # h_wt = n*n/np.sum(1.0/(tmp_window+1e-3))
            h_wt = n * n / np.sum(1.0 / (tmp_window + 1e-3))

            her_m[i,j] = h_wt

            g_wt=0
            cnt=0

            for x in range(n):
                for y in range(n):
                    if tmp_window[x][y]>0:
                        g_wt+= np.log(tmp_window[x][y])
                        cnt+=1

            if(cnt>0):
                g_wt = np.exp(g_wt/cnt)
            else:
                g_wt = 0

            geo_m[i,j] = g_wt

    return her_m,geo_m

def PSNR(img,n_img):
    original_img = img.astype(np.float64)
    n_img = n_img.astype(np.float64)

    mse = np.mean((original_img-n_img)**2)
    mx_p_val = 255
    psnr = 20*np.log10(mx_p_val/(np.sqrt(mse)))
    return psnr


noise_img = salt_pepper_noise(img)
psnr_ns = PSNR(img, noise_img)
harmoic_f,geometric_f = harmonic_geometric_f(noise_img)
psnr_har = PSNR(img, harmoic_f)
psnr_geo = PSNR(img, geometric_f)

plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('original image')
plt.axis('off')
plt.tight_layout()

plt.subplot(222)
plt.imshow(noise_img,cmap='gray')
plt.title(f"noisy image")
plt.axis('off')
plt.tight_layout()

plt.subplot(223)
plt.imshow(harmoic_f,cmap='gray')
plt.title(f"harmonic_filter with psnr : {psnr_har:.2f} DB")
plt.axis('off')
# plt.tight_layout()

plt.subplot(224)
plt.imshow(geometric_f,cmap='gray')
plt.axis('off')
plt.title(f"geometric filter with psnr : {psnr_geo:.2f} DB")
plt.tight_layout()
plt.show()