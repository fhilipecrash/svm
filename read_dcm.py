import pydicom
import cv2
import cupy as cp

def anisotropic_diffusion_with_median_filter_gpu(img, num_iter=5, kappa=50, gamma=0.1):
    img_gpu = cp.array(img, dtype=cp.float32)
    for i in range(num_iter):
        dx = cp.gradient(img_gpu, axis=1)
        dy = cp.gradient(img_gpu, axis=0)
        grad_magnitude = cp.sqrt(dx ** 2 + dy ** 2)
        c = cp.exp(-(grad_magnitude / kappa) ** 2)
        img_gpu += gamma * (c * dx + c * dy)
    img_cpu = cp.asnumpy(img_gpu)
    median_blur = cv2.medianBlur(img_cpu, 3)
    gaussian_blur = cv2.GaussianBlur(img_cpu,(5,5),0)
    return median_blur, gaussian_blur

dcm_data = pydicom.dcmread('B1_F602D90C.dcm')
# img = cv2.cvtColor(dcm_data.pixel_array*128, cv2.IMREAD_GRAYSCALE)
img = dcm_data.pixel_array.astype(cp.float32)
img = (img - cp.min(img)) / (cp.max(img) - cp.min(img) + 1e-7)

gaussian_img, median_img = anisotropic_diffusion_with_median_filter_gpu(img)
gaussian_img = cv2.resize(gaussian_img, (800, 600))
median_img = cv2.resize(median_img, (800, 600))
cv2.imshow('Gaussian', gaussian_img)
cv2.imshow('Median', median_img)
cv2.waitKey(0)
