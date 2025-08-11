# Parking Attendent

A Python-based tool for real-time license plate detection and recognition using OCR techniques.

## Overview

This project processes video frames using YOLOv5-style detection, combining multiple OCR methods, including EasyOCR and a fast ONNX-based recognizer, with post-processing and fallback strategies to extract license plate texts. It includes improvements such as image preprocessing, deskewing, and result consolidation across frames.

## Features

- **License Plate Detection** via YOLO-based model (`detectx`)
- **Advanced OCR** combining:
  - `EasyOCR`
  - ONNX-based recognizer (`fast_plate_ocr`)
- **Preprocessing**:
  - CLAHE contrast enhancement
  - Image resizing and denoising
  - Adaptive thresholding and deskewing
  - Morphological filtering
- **Intelligent Recognition Workflow**:
  - Merges and compares OCR outputs
  - Applies fallback with Tesseract if needed (`advanced_recognize_plate`)
- **Bounding Box Tracking**: Updates readings across frames using IoU logic and retains highest-confidence detections
- **Grouping and Filtering**: Consolidates similar text results across frames and selects the most reliable ones

## File Structure

- `car_detection.py` – Main detection logic and OCR pipeline  
- `license_plate.py` – OCR utility functions (preprocessing, cleaning, fallback)  
- `multiranger_test.py` – Demo/test script showcasing the system in action  

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/dchoi2145/parking-attendent.git
cd parking-attendent

# Install required dependencies (e.g., torch, opencv-python, easyocr, pytesseract, fast_plate_ocr)
pip install -r requirements.txt  # if the file is provided
```

**Note:** You may need to install Tesseract on your system.  
```bash
sudo apt-get install tesseract-ocr
```

## Usage

```python
from car_detection import detectx, plot_boxes
from license_plate import advanced_recognize_plate

# Load YOLO model and initialize OCR readers
# Read video frames, run detection and recognition, display or save annotated output
```

Alternatively, run the `multiranger_test.py` script to see the pipeline in action.

## Configuration

Parameters such as:
- `OCR_TH` (OCR confidence threshold)
- `IOU_THRESHOLD` (for merging detections)
- `similarity_threshold`, `min_count` for grouping plates  

are configurable within the code.  
Tweaking these thresholds can improve performance based on your dataset and environment.
