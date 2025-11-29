import cv2
import numpy as np
import glob
import os

SCENE_PATH = "dataset/testScene.jpg"
TEMPLATES_DIR = "dataset"
OUTPUT_FILENAME = "static/task1_result.jpg"

IGNORE_FILES = {
    "testScene.jpg",
    "task2_source.jpg",
    "task1_result.jpg",
    "result.jpg",
    ".DS_Store",
}


def list_templates():
    paths = []
    for p in glob.glob(os.path.join(TEMPLATES_DIR, "*")):
        name = os.path.basename(p)
        ext = os.path.splitext(name)[1].lower()
        if name in IGNORE_FILES:
            continue
        if ext in [".jpg", ".jpeg", ".png"]:
            paths.append(p)
    return paths


def best_match_gray(gray_scene, t_gray):
    H, W = gray_scene.shape[:2]
    th, tw = t_gray.shape[:2]
    best_score, best = -1.0, None

    for scale in np.linspace(0.7, 1.3, 21):
        w = int(tw * scale)
        h = int(th * scale)
        if w < 10 or h < 10 or w > W or h > H:
            continue

        t_resized = cv2.resize(t_gray, (w, h))
        res = cv2.matchTemplate(gray_scene, t_resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > best_score:
            best_score = max_val
            x, y = max_loc
            best = (x, y, w, h, best_score)

    return best


def detect_highlighter_bgr(scene_bgr):
    H, W = scene_bgr.shape[:2]
    B = scene_bgr[:, :, 0].astype(np.uint8)
    G = scene_bgr[:, :, 1].astype(np.uint8)
    R = scene_bgr[:, :, 2].astype(np.uint8)

    mask = (R > 200) & (G < 120) & (B > 150)
    mask = mask.astype(np.uint8) * 255

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 300: 
        return None

    x, y, w, h = cv2.boundingRect(largest)

    pad = 5
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(W - x, w + 2 * pad)
    h = min(H - y, h + 2 * pad)
    return x, y, w, h


def run_detection():
    scene = cv2.imread(SCENE_PATH)
    if scene is None:
        print("Error: could not load scene image.")
        return

    gray_scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    gray_scene = cv2.GaussianBlur(gray_scene, (3, 3), 0)
    H, W = gray_scene.shape[:2]

    result = scene.copy()
    templates = list_templates()
    print("Templates:", [os.path.basename(p) for p in templates])

    for tpath in templates:
        name = os.path.splitext(os.path.basename(tpath))[0]

        if name == "highlighter":
            box = detect_highlighter_bgr(scene)
            if box is None:
                print("[NO MATCH] highlighter (color detection failed)")
                continue
            x, y, w, h = box
            score = 1.0  
            print(f"[MATCH] highlighter (color): at ({x},{y}) size ({w},{h})")
        else:
            t_color = cv2.imread(tpath)
            if t_color is None:
                continue
            th, tw = t_color.shape[:2]
            if th > H or tw > W:
                print(f"Skipping {name} (template larger than scene)")
                continue

            t_gray = cv2.cvtColor(t_color, cv2.COLOR_BGR2GRAY)
            t_gray = cv2.GaussianBlur(t_gray, (3, 3), 0)

            match = best_match_gray(gray_scene, t_gray)
            if match is None:
                print(f"[NO MATCH] {name}")
                continue
            x, y, w, h, score = match
            print(f"[MATCH] {name}: {score:.2f} at ({x},{y}) size ({w},{h})")

        # Draw final box
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(result, name, (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    os.makedirs(os.path.dirname(OUTPUT_FILENAME), exist_ok=True)
    cv2.imwrite(OUTPUT_FILENAME, result)
    print("Saved:", OUTPUT_FILENAME)

    scale = 900.0 / max(H, W)
    small = cv2.resize(result, (int(W * scale), int(H * scale)))
    cv2.imshow("Task 1 Result", small)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detection()
