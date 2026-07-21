# Computer Vision and Object Detection Playground

This repository contains a collection of Jupyter Notebooks (`.ipynb`) implementing various computer vision tasks, object detection architectures, and model performance evaluations using Python, PyTorch, Ultralytics YOLOv8, and OpenCV.

---

## Notebooks Overview

1. **`ssd_vgg16_detection.ipynb`**: 
   * Demonstrates object detection on a video stream using a pre-trained **PyTorch SSD300 VGG16** model (COCO weights).
   * Implements custom bounding box post-processing, confidence thresholding, and Non-Maximum Suppression (NMS) using `torchvision.ops`.
2. **`yolov8_training_inference.ipynb`**: 
   * Covers end-to-end workflows for **Ultralytics YOLOv8**:
     * Loading and inspecting model architecture (`yolov8n.yaml`).
     * Training custom datasets (`model.train`).
     * Running video inference with dynamic confidence thresholds and custom bounding box overlays.
     * Parsing and visualizing training metrics (`results.csv`) via `pandas` and `matplotlib`.

---

## Requirements

Ensure you have Python 3.7+ installed. You can install the required packages using pip:

```bash
pip install -r requirements.txt
