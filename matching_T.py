# matching_T.py

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

# Paths (relative to your Module 2 folder)
TEMPLATE_FOLDER = os.path.join("static", "dataset")        # template images
TEST_IMAGES_FOLDER = os.path.join("static", "test_images") # test/full images

# Output folder lives under static so Flask can serve it
OUTPUT_SUBDIR = "output_TM"                                # relative to /static
OUTPUT_FOLDER = os.path.join("static", OUTPUT_SUBDIR)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Template matching method
METHOD_NAME = 'TM_CCOEFF_NORMED'
method = getattr(cv, METHOD_NAME)


def run_template_matching(show_plots: bool = False):
    #load all templates 
    template_files = [
        f for f in os.listdir(TEMPLATE_FOLDER)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
    ]
    assert template_files, f"No template images found inside '{TEMPLATE_FOLDER}'!"

    result_files = []

    #main loop: over each test image 
    for test_filename in os.listdir(TEST_IMAGES_FOLDER):
        if not test_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            continue

        test_path = os.path.join(TEST_IMAGES_FOLDER, test_filename)
        img_gray = cv.imread(test_path, cv.IMREAD_GRAYSCALE)

        if img_gray is None:
            print(f"Skipping {test_filename} — unable to read the file.")
            continue

        ih, iw = img_gray.shape[:2]
        print(f"\nProcessing: {test_filename} ({iw}x{ih})")

        #compare with every template
        for template_name in template_files:
            template_path = os.path.join(TEMPLATE_FOLDER, template_name)
            template = cv.imread(template_path, cv.IMREAD_GRAYSCALE)

            if template is None:
                print(f"Template '{template_name}' could not be read — skipping.")
                continue

            th, tw = template.shape[:2]   # template height, width

            #size check 
            if th > ih or tw > iw:
                print(
                    f"Skipping pair: template '{template_name}' "
                    f"({tw}x{th}) bigger than image '{test_filename}' ({iw}x{ih})."
                )
                continue

            #template matching 
            res = cv.matchTemplate(img_gray, template, method)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

            if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc

            bottom_right = (top_left[0] + tw, top_left[1] + th)

            # Draw rectangle on copy
            detected_img = img_gray.copy()
            cv.rectangle(detected_img, top_left, bottom_right, 255, 2)

            # Optional visualization when running from terminal
            if show_plots:
                plt.figure(figsize=(10, 4))

                plt.subplot(1, 2, 1)
                plt.imshow(res, cmap='gray')
                plt.title('Matching Result')
                plt.xticks([]), plt.yticks([])

                plt.subplot(1, 2, 2)
                plt.imshow(detected_img, cmap='gray')
                plt.title('Detected Point')
                plt.xticks([]), plt.yticks([])

                plt.suptitle(f"{METHOD_NAME} | {test_filename} vs {template_name}")
                plt.show()

            #save result 
            output_filename = (
                f"{os.path.splitext(test_filename)[0]}_"
                f"{os.path.splitext(template_name)[0]}_"
                f"{METHOD_NAME}.jpg"
            )
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            cv.imwrite(output_path, detected_img)
            print(f"Saved: {output_path}")

            # Relative path for Flask
            result_files.append(f"{OUTPUT_SUBDIR}/{output_filename}")

    print("\nAll detections completed. Results are saved in:", OUTPUT_FOLDER)
    return sorted(result_files)


if __name__ == "__main__":
    # For debugging directly from terminal (with plots)
    run_template_matching(show_plots=True)
