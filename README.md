# ADaCITra

ADaCITra is a desktop application for post-training evaluation and
real-time performance assessment of YOLO object-detection models.

ADaCITra supports image-mode post-training evaluation through IoU-based ground-truth matching, synchronized visualization, 
confusion-matrix generation, AP/mAP reporting, and traceable instance-level prediction records in CSV format. 
It also supports video and camera inference with real-time FPS visualization, together with cross-platform latency and throughput measurement.

## Case Study

The application is demonstrated using axle-based vehicle classification
for Indonesian toll-road applications with YOLOv8 and YOLO11 models.

## Dataset

The axle-based vehicle-classification case study uses the
**InaTRC: Indonesia Toll Road Vehicle Classification Dataset**,
available from Mendeley Data:

- Dataset version: Version 1
- DOI: https://doi.org/10.17632/kddgphck3b.1
- Total images: 3,250
- Annotation format: YOLO
- Predefined split: 80% training, 10% validation, and 10% testing

The predefined dataset split consists of:

| Split | Images | Use in this study |
|---|---:|---|
| Training | 2,600 | YOLO model training |
| Validation | 325 | Model validation during training |
| Test | 325 | Post-training evaluation using ADaCITra |

The test split was kept separate from model training and was used
exclusively for the experiments reported in the ADaCITra study. It
contains 3,642 annotated vehicle instances.

The dataset contains five axle-based vehicle classes:

1. **NonTruck** — sedans, pick-ups, minibuses, buses, MPVs, and SUVs
2. **2Axle** — trucks with two axles
3. **3Axle** — trucks with three axles
4. **4Axle** — trucks with four axles
5. **5Axle** — trucks with five or more axles

Each annotation follows the normalized YOLO bounding-box format:

`<class_id> <x_center> <y_center> <width> <height>`

The dataset is not redistributed in this repository. Users should
download Version 1 from the official Mendeley Data repository and use
its predefined test split to reproduce the post-training evaluation.

## Requirements

- Python 3.10
- Ultralytics
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Pillow

## Availability

Source code: https://github.com/umic001/adacitra

## Installation
```bash
pip install -r requirements.txt
python adacitraApps.py
```
***
