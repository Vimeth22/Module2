import os
import cv2
import glob
import numpy as np
import shutil
import uuid
import time
from flask import Flask, render_template, request

app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'static'
TEMPLATES_DIR = 'dataset'
SCENE_SOURCE = 'dataset/testScene.jpg'
# Keeping the threshold at 0.45, but normalization should improve scores significantly.
MATCH_THRESHOLD = 0.45 

IGNORE_FILES = {
    'testScene.jpg', 
    'task2_source.jpg', 
    'task1_result.jpg', 
    'result.jpg',
    '.DS_Store'
}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def index():
    original_display = None
    processed_display = None

    if os.path.exists(SCENE_SOURCE):
        static_scene_path = os.path.join(UPLOAD_FOLDER, 'original_scene.jpg')
        if not os.path.exists(static_scene_path):
            shutil.copy(SCENE_SOURCE, static_scene_path)
        original_display = static_scene_path

    if request.method == 'POST':
        if os.path.exists(SCENE_SOURCE):
            processed_filename = process_image_and_blur(SCENE_SOURCE)
            processed_display = f"{os.path.join(UPLOAD_FOLDER, processed_filename)}?v={int(time.time())}"

    return render_template('index.html', original=original_display, processed=processed_display)

def process_image_and_blur(image_path):
    main_img = cv2.imread(image_path)
    gray_main = cv2.cvtColor(main_img, cv2.COLOR_BGR2GRAY)
    final_img = main_img.copy()

    all_files = glob.glob(os.path.join(TEMPLATES_DIR, "*"))
    
    print(f"--- Processing Scene: {image_path} ---")
    
    detected_regions = []

    for t_path in all_files:
        t_name = os.path.basename(t_path)
        
        if t_name in IGNORE_FILES or not t_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        template = cv2.imread(t_path)
        if template is None: continue
        
        # --- FIX: Template Normalization ---
        # 1. Convert to Grayscale
        gray_temp = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # 2. Normalize the template to standard range (0-255). 
        # This helps the matcher ignore global brightness differences and focus on shape.
        gray_temp = cv2.normalize(gray_temp, None, alpha=0, beta=255, 
                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # --- END FIX ---
        
        (tH, tW) = gray_temp.shape[:2]

        if tH > gray_main.shape[0] or tW > gray_main.shape[1]:
            continue

        best_match = None
        
        # Multi-Scale Detection range: 0.5 to 1.5, 25 steps
        for scale in np.linspace(0.5, 1.5, 25): 
            resized_w = int(tW * scale)
            resized_h = int(tH * scale)
            
            if resized_w > gray_main.shape[1] or resized_h > gray_main.shape[0]:
                continue

            resized_t = cv2.resize(gray_temp, (resized_w, resized_h))
            
            if resized_t.shape[0] == 0 or resized_t.shape[1] == 0:
                continue

            res = cv2.matchTemplate(gray_main, resized_t, cv2.TM_CCOEFF_NORMED)
            (_, max_val, _, max_loc) = cv2.minMaxLoc(res)

            if best_match is None or max_val > best_match[0]:
                best_match = (max_val, max_loc, resized_w, resized_h)

        if best_match:
            (score, (x, y), w, h) = best_match
            
            if score >= MATCH_THRESHOLD:
                is_overlap = False
                for (prev_x, prev_y, prev_w, prev_h) in detected_regions:
                    # Simple center-point overlap check
                    center_x, center_y = x + w // 2, y + h // 2
                    if (prev_x < center_x < prev_x + prev_w and 
                        prev_y < center_y < prev_y + prev_h):
                        is_overlap = True
                        break
                
                if not is_overlap:
                    print(f"Blurring {t_name} (Score: {score:.2f})")
                    detected_regions.append((x, y, w, h))
                    
                    # Apply Blur
                    roi = final_img[y:y+h, x:x+w]
                    blurred_roi = cv2.GaussianBlur(roi, (71, 71), 40)
                    final_img[y:y+h, x:x+w] = blurred_roi
                    
                    # Draw Red Border
                    cv2.rectangle(final_img, (x, y), (x + w, y + h), (0, 0, 255), 3)

    unique_name = f"result_{uuid.uuid4().hex[:8]}.jpg"
    result_path = os.path.join(UPLOAD_FOLDER, unique_name)
    
    cv2.imwrite(result_path, final_img)
    return unique_name

if __name__ == '__main__':
    app.run(debug=True)