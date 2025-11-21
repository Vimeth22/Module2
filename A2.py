import cv2
import numpy as np
from numpy.fft import fft2, ifft2
from skimage.restoration import richardson_lucy
import os

# Gaussian PSF + Hybrid Wiener–Lucy Restoration
def gaussian_psf(size, sigma):
    k = size // 2
    x = np.arange(-k, k + 1)
    xx, yy = np.meshgrid(x, x)
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    psf /= psf.sum()
    return psf

def pad_psf(psf, shape):
    pad = np.zeros(shape, dtype=np.float32)
    h, w = psf.shape
    pad[:h, :w] = psf
    pad = np.roll(pad, -h // 2, axis=0)
    pad = np.roll(pad, -w // 2, axis=1)
    return pad

# Hybrid Restoration: Wiener + Richardson–Lucy
def hybrid_wiener_lucy(img_blur, psf, K=0.002, iterations=10):
    psf_padded = pad_psf(psf, img_blur.shape[:2])
    H = fft2(psf_padded)
    absH2 = np.abs(H) ** 2

    result = np.zeros_like(img_blur)
    for c in range(img_blur.shape[2]):
        #Wiener pre-restoration
        G = fft2(img_blur[:, :, c])
        F_hat = np.conj(H) / (absH2 + K) * G
        f_wiener = np.real(ifft2(F_hat))
        f_wiener = np.clip(f_wiener, 0, 1)

        # Richardson–Lucy refinement 
        f_refined = richardson_lucy(f_wiener, psf, num_iter=iterations, clip=False)
        result[:, :, c] = np.clip(f_refined, 0, 1)
    return result

def adjust_gamma(img, gamma=0.9):
    return np.power(np.clip(img, 0, 1), gamma)

# Process Each Image
def process_image(img_path, output_dir="Outputs", sigma=7):
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(img_path))[0]

    L = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if L is None:
        print(f"Skipping {img_path} (not found).")
        return
    L = cv2.cvtColor(L, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Apply Gaussian blur
    L_b = cv2.GaussianBlur(L, (0, 0), sigma)

    # Hybrid recovery
    psf_size = int(6 * sigma) | 1
    psf = gaussian_psf(psf_size, sigma)
    L_recovered = hybrid_wiener_lucy(L_b, psf, K=0.002, iterations=8)

    # White sheet background
    h, w = L.shape[:2]
    margin_top, margin_bottom = 100, 100
    margin_sides, space_between = 150, 150
    total_width = 3 * w + 4 * margin_sides + 2 * space_between
    total_height = h + margin_top + margin_bottom
    white_bg = np.ones((total_height, total_width, 3), dtype=np.float32)

    # Gamma adjust
    L_disp, L_b_disp, L_rec_disp = map(adjust_gamma, [L, L_b, L_recovered])
    positions = [
        (margin_top, margin_sides),
        (margin_top, margin_sides + w + space_between),
        (margin_top, margin_sides + 2 * (w + space_between))
    ]
    for img, (y, x) in zip([L_disp, L_b_disp, L_rec_disp], positions):
        white_bg[y:y+h, x:x+w] = img

    # Titles
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness, color = 1.1, 3, (0, 0, 0)
    titles = ["Original Image", "Blurred Image", "Recovered Image"]
    text_positions = [(p[1] + 40, 70) for p in positions]
    for text, (x_pos, y_pos) in zip(titles, text_positions):
        for dx, dy in [(0,0),(1,0),(0,1),(-1,0),(0,-1)]:
            cv2.putText(white_bg, text, (x_pos+dx, y_pos+dy),
                        font, font_scale, color, thickness, cv2.LINE_AA)

    output = (white_bg * 255).astype(np.uint8)
    save_path = os.path.join(output_dir, f"{name}_comparison.png")
    cv2.imwrite(save_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    print(f"Processed and saved: {save_path}")

# 10 Images
if __name__ == "__main__":
    image_paths = [
        os.path.join("static", "dataset", "Rubic.png"),
        os.path.join("static", "dataset", "light.jpg"),
        os.path.join("static", "dataset", "blueberry.webp"),
        os.path.join("static", "dataset", "rose.jpeg"),
        os.path.join("static", "dataset", "avadaco.jpg"),
        os.path.join("static", "dataset", "Apple.jpeg"),
        os.path.join("static", "dataset", "tennis_ball.jpg"),
        os.path.join("static", "dataset", "camera.webp"),
        os.path.join("static", "dataset", "v2.jpeg"),
        os.path.join("static", "dataset", "AA.webp"),
    ]
    for path in image_paths:
        process_image(path)
    print("All images processed successfully in Outputs folder.")