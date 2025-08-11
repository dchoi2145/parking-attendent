# Parking Attendant

A Python tool for detecting and reading license plates in real time using computer vision and OCR.

## Overview

This project combines object detection, OCR, and optional drone control.  
It can:
- Detect license plates in video streams and read them using multiple OCR engines.
- Detect vehicles in the environment for potential obstacle avoidance.
- Run drone flight tests to confirm car detection triggers.

The system uses a YOLOv5-based detector for spotting license plates, EasyOCR and a fast ONNX-based recognizer for text recognition, and Tesseract as a fallback. Preprocessing steps like contrast enhancement, deskewing, and filtering help improve recognition accuracy.

## Features

- **License Plate Detection** with a YOLO-based model (`detectx`)
- **Multiple OCR Methods**:
  - `EasyOCR`
  - ONNX-based recognizer (`fast_plate_ocr`)
  - Tesseract as a fallback
- **Image Preprocessing** for better OCR results:
  - CLAHE contrast enhancement
  - Resizing and denoising
  - Adaptive thresholding and deskewing
  - Morphological filtering
- **Smart Recognition Workflow**:
  - Compares OCR outputs and keeps the most confident result
  - Groups and merges similar detections across frames
- **Vehicle Detection for Obstacles**:
  - Detects cars that could interfere with drone movement
- **Drone Flight Testing**:
  - `multiranger_test.py` checks if the Multiranger deck detects a car
  - If a car is detected, the drone will automatically ascend ~8 ft

## File Structure

- `car_detection.py` – Detects cars as potential obstacles
- `license_plate.py` – Helper functions for OCR and preprocessing  
- `multiranger_test.py` – Tests the Multiranger deck’s car detection; flies upward if triggered

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/dchoi2145/parking-attendent.git
cd parking-attendent

# Install dependencies (torch, opencv-python, easyocr, pytesseract, fast_plate_ocr, etc.)
pip install -r requirements.txt
```

**Note:** You’ll also need Tesseract installed on your system:
```bash
sudo apt-get install tesseract-ocr
```

## Usage

```python
from car_detection import detectx, plot_boxes
from license_plate import advanced_recognize_plate

# Load YOLO model and initialize OCR readers
# Process video frames, detect plates, read text, and display/save results
```

Or simply run:
```bash
python multiranger_test.py
```
to see the full pipeline work.

## Configuration

You can tweak settings inside the code to adjust performance, such as:
- `OCR_TH` - OCR confidence threshold
- `IOU_THRESHOLD` - For merging detections between frames
- `similarity_threshold` and `min_count` - For grouping similar plates
  
Adjusting these can improve results depending on your video quality and environment.
