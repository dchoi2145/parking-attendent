# Imports
import torch
import cv2
import re
import numpy as np
import easyocr
import difflib
import random
import pytesseract 
from fast_plate_ocr import ONNXPlateRecognizer

# Initialize OCR models
ocr_model = ONNXPlateRecognizer('global-plates-mobile-vit-v2-model')
EASY_OCR = easyocr.Reader(['en'])

OCR_TH = 0.2 # Threshold for valid OCR text region
IOU_THRESHOLD = 0.7 # IoU threshold to match overlapping boxes

# Stores all detected plate readings across video frames
global_plate_readings = []

# Compute IoU between two bounding boxes
def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

# Add or update a plate detection in the global list
def update_plate_readings(bbox, text, conf, frame_no):
    global global_plate_readings
    found_match = False
    for entry in global_plate_readings:
        if compute_iou(bbox, entry['bbox']) > IOU_THRESHOLD:
            if conf > entry['conf'] or (conf == entry['conf'] and len(text) > len(entry['text'])):
                entry.update({'bbox': bbox, 'text': text, 'conf': conf, 'frame': frame_no})
            found_match = True
            break
    if not found_match:
        global_plate_readings.append({'bbox': bbox, 'text': text, 'conf': conf, 'frame': frame_no})

# Group similar plate texts and return the most confident reading from each group
def group_plate_readings(plate_list, similarity_threshold=0.8, min_count=10):
    groups = []
    for reading in plate_list:
        added = False
        for group in groups:
            if difflib.SequenceMatcher(None, reading['text'], group[0]['text']).ratio() >= similarity_threshold:
                group.append(reading)
                added = True
                break
        if not added:
            groups.append([reading])
    unique_plates = []
    for group in groups:
        if len(group) >= min_count:
            best = max(group, key=lambda r: r['conf'])
            unique_plates.append(best)
    return unique_plates

# Quick check to discard clearly invalid plate texts
def is_valid_plate(text, min_length=5):
    return len(text) >= min_length and " " not in text

# Run YOLOv5 detection on a frame
def detectx(frame, model):
    results = model([frame])
    labels = results.xyxyn[0][:, -1]
    coords = results.xyxyn[0][:, :-1]
    return labels, coords

# Try to deskew a plate image (correct tilt)
def deskew_plate(image):
    coords = np.column_stack(np.where(image > 0))
    if coords.size == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# Preprocess a plate image to enhance OCR accuracy
def preprocess_plate(nplate):
    gray = cv2.cvtColor(nplate, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    resized = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(resized, (3, 3), 0)
    adaptive_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
    _, otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    combined = cv2.bitwise_and(adaptive_thresh, otsu_thresh)
    deskewed = deskew_plate(combined)
    morph = cv2.morphologyEx(deskewed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return morph

# Remove invalid characters and format plate text
def clean_plate_text(text):
    return re.sub(r'[^A-Z0-9-]', '', text.upper())

# Select the best OCR result based on size of the detected region
def filter_text(region, ocr_result, region_threshold):
    rect_area = region.shape[0] * region.shape[1]
    valid = []
    for (points, string_val, conf) in ocr_result:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        area = w * h
        ratio = area / rect_area if rect_area > 0 else 0
        if ratio > region_threshold:
            valid.append((ratio, string_val, conf))
    if valid:
        best = max(valid, key=lambda x: x[0])
        return best[1], best[2]
    return "", 0.0

# Run EasyOCR on the cropped plate
def recognize_plate_easyocr(img, bbox, reader, region_threshold):
    x1, y1, x2, y2 = bbox
    cropped = img[y1:y2, x1:x2]
    proc = preprocess_plate(cropped)
    ocr_result = reader.readtext(proc)
    text_out, conf = filter_text(proc, ocr_result, region_threshold)
    return clean_plate_text(text_out), conf

# Combine EasyOCR and ONNX OCR results, choose the better one
def robust_recognize_plate(img, bbox, easy_reader, onnx_model, region_threshold):
    easy_text, easy_conf = recognize_plate_easyocr(img, bbox, easy_reader, region_threshold)
    cropped = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    onnx_text = onnx_model.run(gray)
    if isinstance(onnx_text, list):
        onnx_text = " ".join(onnx_text)
    onnx_text = clean_plate_text(onnx_text)
    onnx_conf = 0.9
    similarity = difflib.SequenceMatcher(None, easy_text, onnx_text).ratio()
    if easy_text and onnx_text and similarity >= 0.8:
        return (onnx_text, onnx_conf) if onnx_conf >= easy_conf else (easy_text, easy_conf)
    return (onnx_text, onnx_conf) if onnx_text else (easy_text, easy_conf)

# Tesseract fallback for cases where other OCRs fail
def fallback_recognize_plate(img, bbox):
    x1, y1, x2, y2 = bbox
    cropped = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh, config='--psm 7')
    return clean_plate_text(text), 0.5

# Try EasyOCR + ONNX, then fallback to Tesseract if needed
def advanced_recognize_plate(img, bbox, easy_reader, onnx_model, region_threshold):
    text, conf = robust_recognize_plate(img, bbox, easy_reader, onnx_model, region_threshold)
    if not is_valid_plate(text) or conf < 0.5:
        fallback_text, fallback_conf = fallback_recognize_plate(img, bbox)
        if is_valid_plate(fallback_text) and fallback_conf > conf:
            return fallback_text, fallback_conf
    return text, conf

# Draw results and annotate plates on the frame
def plot_boxes(results, frame, classes, frame_no):
    labels, coords = results
    annotated = frame.copy()
    x_shape, y_shape = annotated.shape[1], annotated.shape[0]
    for i in range(len(labels)):
        row = coords[i]
        conf = float(row[4])
        if conf >= 0.55:
            x1 = int(row[0] * x_shape)
            y1 = int(row[1] * y_shape)
            x2 = int(row[2] * x_shape)
            y2 = int(row[3] * y_shape)
            bbox = [x1, y1, x2, y2]
            text, text_conf = advanced_recognize_plate(annotated, bbox, EASY_OCR, ocr_model, OCR_TH)
            if not text:
                text = classes[int(labels[i])]
                text_conf = conf
            update_plate_readings(bbox, text, text_conf, frame_no)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
            font_scale = max(0.5, (x2 - x1) / 150.0)
            cv2.rectangle(annotated, (x1, y1-25), (x2, y1), (0,255,0), -1)
            cv2.putText(annotated, text, (x1+5, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 2)
    return annotated
