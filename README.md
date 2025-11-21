# Module 2 – Hybrid Restoration & Template Matching
### Part 1: Hybrid Wiener–Lucy Restoration  
### Part 2: Template Matching (matching_T.py)

This module implements a full Flask web application for performing image restoration and template matching.  
The interface allows switching between dataset images, uploaded images, and template-matching mode.  
All processing is done in Python using OpenCV and custom restoration functions.

## 1. Project Structure

```
Module2/
│── Outputs
│── requirements.txt
│── static/
    ├── dataset/             # Provided test images
    ├── output_TM/            # Template-matching results
    ├── results/  # Saved restoration outputs
    └── test_images/           # The images for part 4. (you can upload your own)

├── templates/
│   └── index.html           # Main UI
|
|── app.py
│── A2.py                    # Hybrid restoration functions
│── matching_T.py            # Template matching engine
```

## 2. Features

### Part 1 – Hybrid Wiener–Lucy Restoration
- Gaussian PSF 
- Frequency-domain Wiener filtering  
- Richardson–Lucy iterations  
- Adjustable parameters:
  - σ (sigma)
  - K (noise constant)
  - iterations
- Results saved inside `static/results/`

### Part 2 – Template Matching
- Uses TM_CCOEFF_NORMED
- Draws bounding boxes around best matches
- Saves results into `static/output_TM/`
- Generates multiple preview cards

## 3. How to Run

Install dependencies:
```
pip install -r requirements.txt
```

Run Flask server:
```
python app.py
```

Open browser: the link is supposed to be in the terminal once you run the app.py

## 4. Using the Application

### Hybrid Restoration
1. Select Dataset or Upload tab  
2. Choose an image  
3. Adjust σ, K, iterations  
4. Click "Run Hybrid Restoration"  
5. Results appear below as cards

### Template Matching
1. First you have to upload your images to `static/test_images`
2. Select Template Matching tab  
3. Click "Run Template Matching"  
4. Preview cards will appear


## 5. Output Locations

- Restoration results: `static/results/`
- Template matching results: `static/output_TM/`

## 6. Summary

This module provides an interactive web app for experimenting with:
- Hybrid deblurring
- Frequency + spatial domain filtering
- Template matching
- Rich frontend visualization
