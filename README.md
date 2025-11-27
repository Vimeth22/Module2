
# Module 2 

This README describes all tasks implemented in Module 2, based on your uploaded files (`app.py`, `task1.py`, `task2.py`, `index.html`).  
The module covers classical template matching, image restoration, and a Flask-based privacy-redaction system.

---

## Task – Template Matching (OpenCV)

### **Overview**
Task 1 performs object detection using classical **template matching**.

### **Main Steps**
1. Load template images.  
2. Convert images to grayscale.  
3. Use `cv2.matchTemplate()` with method `TM_CCOEFF_NORMED`.  
4. Apply thresholding to keep strong matches.  
5. Draw bounding boxes on detected areas.  
6. Save outputs (e.g., `task1_result.jpg`).

### **Output**
- Image with red bounding boxes indicating detected templates.

---

## Task 2 – Image Restoration (Wiener + Sharpening)

### **Overview**
Task 2 applies digital image restoration to improve blurry or noisy images.

### **Processing Pipeline**
- Wiener filter for deblurring  
- Sharpening kernel for enhancing edges  
- Saving restored images (e.g., `task2_restored.jpg`)

### **Output**
- Restored, sharper version of the input image.

---

## Task 3 – Privacy Redaction Web App (Flask)

### **Overview**
A full web-based pipeline to automatically detect and blur privacy-sensitive regions.

### **Features**
- Load and process images from the dataset  
- Perform template matching  
- Apply Gaussian blur on detected regions  
- Display original and processed results on a webpage  

### **Files**
- **app.py** → Backend logic  
- **index.html** → Frontend UI  
- **static/** → Stores generated outputs  

### **How it Works**
1. User clicks **Execute Redaction**  
2. Flask runs template matching  
3. Sensitive areas are blurred  
4. Processed image is displayed  

---

## Folder Structure

```
MODULE2/
│── app.py
│── README.md
│── task1.py
│── task2.py
│
├── dataset/
├── static/
└── templates/

```

---

## How to Run

### **1. Template Matching**
```bash
python task1.py
```

### **2. Image Restoration**
```bash
python task2.py
```

### **3. Run the Web App**
```bash
python app.py
```

Access the app at:
```
http://127.0.0.1:5000
```

---

## Requirements
Install required packages:

```bash
pip install flask opencv-python numpy
```

