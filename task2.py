import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# Configuration
IMAGE_PATH   = 'dataset/task2_source.jpg'   # Input: blurred text image
KERNEL_SIZE  = 21                           # Gaussian blur size
SIGMA        = 5                            # Gaussian sigma
SNR          = 0.005                        # Wiener filter noise term

STATIC_DIR   = 'static'  
BLURRED_OUT  = os.path.join(STATIC_DIR, 'task2_blurred.jpg')
RESTORED_OUT = os.path.join(STATIC_DIR, 'task2_restored.jpg')


def recover_channel(channel, ksize, sigma):
    """
    Deblur a single channel using a frequency-domain
    Wiener-style deconvolution + sharpening.
    """
    # 1) Work in float
    channel = channel.astype(np.float32)

    # 2) Build 2D Gaussian kernel
    k_1d = cv2.getGaussianKernel(ksize, sigma)
    k_2d = np.outer(k_1d, k_1d)

    # 3) Pad kernel to image size so FFTs align
    padded_k = np.zeros_like(channel, dtype=np.float32)
    kh, kw = k_2d.shape
    padded_k[:kh, :kw] = k_2d

    # 4) Forward FFT of image and kernel
    img_fft    = np.fft.fft2(channel)
    kernel_fft = np.fft.fft2(padded_k)

    # 5) Wiener-style deconvolution in the frequency domain
    kernel_conj = np.conj(kernel_fft)
    numerator   = img_fft * kernel_conj
    denominator = (np.abs(kernel_fft) ** 2) + SNR
    restored_fft = numerator / denominator

    # 6) Back to spatial domain
    restored = np.abs(np.fft.ifft2(restored_fft))

    # 7) Smart normalization to [0, 255]
    #    Clip extreme highlights (use 99th percentile)
    max_val = np.percentile(restored, 99)
    min_val = np.min(restored)
    restored = np.clip(restored, min_val, max_val)
    restored = cv2.normalize(restored, None, 0, 255, cv2.NORM_MINMAX)

    # 8) Sharpen edges to make text crisp
    restored = restored.astype(np.uint8)
    sharpen_filter = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
    restored = cv2.filter2D(restored, -1, sharpen_filter)

    return restored


# Main script
# Load source image
img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"Error: {IMAGE_PATH} not found.")
    exit()

os.makedirs(STATIC_DIR, exist_ok=True)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

blurred = cv2.GaussianBlur(img_rgb, (KERNEL_SIZE, KERNEL_SIZE), SIGMA)

restored_channels = []
for c in range(3):
    restored_c = recover_channel(blurred[:, :, c], KERNEL_SIZE, SIGMA)
    restored_channels.append(restored_c)

restored = np.stack(restored_channels, axis=2)

restored = cv2.convertScaleAbs(restored, alpha=1.2, beta=-10)

blurred_bgr  = cv2.cvtColor(blurred,  cv2.COLOR_RGB2BGR)
restored_bgr = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)

cv2.imwrite(BLURRED_OUT, blurred_bgr)
cv2.imwrite(RESTORED_OUT, restored_bgr)

print(f"Saved blurred image to  {BLURRED_OUT}")
print(f"Saved restored image to {RESTORED_OUT}")

plt.figure(figsize=(15, 6))
plt.subplot(1, 3, 1); plt.imshow(img_rgb);  plt.title("Original");              plt.axis('off')
plt.subplot(1, 3, 2); plt.imshow(blurred);  plt.title("Blurred (Input)");       plt.axis('off')
plt.subplot(1, 3, 3); plt.imshow(restored); plt.title("Restored (Ultra Sharp)"); plt.axis('off')
plt.tight_layout()
plt.show()
