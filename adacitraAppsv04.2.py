import os
import cv2
import csv
import torch
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageFont, ImageTk
from ultralytics import YOLO
import threading
import time
import platform

# ================= CONFIG =================
from utils.logger import logger

from utils.config import (
    APP_TITLE,
    CANVAS_W,
    CANVAS_H,
    IMG_W,
    IMG_H,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_CAMERA_HEIGHT,
    CAMERA_BUFFER_SIZE,
)

from utils.helpers import (
    get_usb_camera_index,
    get_class_color,
    load_yolo_label,
)

# ================= EVALUATION CONFIG =================
# Keep these values fixed and report them in the paper.
APP_NAME = "ADaCITra"
APP_VERSION = "0.4.2"
INFERENCE_IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50
WARMUP_RUNS = 3

# AP/mAP requires predictions below the operating confidence threshold so that
# a precision-recall curve can be constructed. Confusion-matrix metrics remain
# evaluated at CONFIDENCE_THRESHOLD.
AP_MIN_CONFIDENCE = 0.001
AP_IOU_THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.50, 0.96, 0.05))
AP_RECALL_POINTS = 101

def calculate_iou_xywhn(box_a, box_b):
    """Calculate IoU for two normalized [x_center, y_center, width, height] boxes."""
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0

    try:
        ax, ay, aw, ah = map(float, box_a)
        bx, by, bw, bh = map(float, box_b)
    except (TypeError, ValueError):
        return 0.0

    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0

    a_x1, a_y1 = ax - aw / 2, ay - ah / 2
    a_x2, a_y2 = ax + aw / 2, ay + ah / 2
    b_x1, b_y1 = bx - bw / 2, by - bh / 2
    b_x2, b_y2 = bx + bw / 2, by + bh / 2

    inter_w = max(0.0, min(a_x2, b_x2) - max(a_x1, b_x1))
    inter_h = max(0.0, min(a_y2, b_y2) - max(a_y1, b_y1))
    intersection = inter_w * inter_h
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def match_detections(gt_objects, pred_objects, iou_threshold=IOU_THRESHOLD):
    """
    Perform class-agnostic one-to-one matching.

    All candidate GT-prediction pairs are sorted by descending IoU. This avoids
    dependence on the order of objects in the label file while still retaining
    class-mismatched spatial matches for the extended confusion matrix.
    """
    candidates = []
    for gt_idx, (_, gt_box) in enumerate(gt_objects):
        for pred_idx, (_, _, pred_box) in enumerate(pred_objects):
            iou = calculate_iou_xywhn(gt_box, pred_box)
            if iou >= iou_threshold:
                candidates.append((iou, gt_idx, pred_idx))

    candidates.sort(key=lambda item: item[0], reverse=True)
    used_gt = set()
    used_pred = set()
    matches = []

    for iou, gt_idx, pred_idx in candidates:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        matches.append((gt_idx, pred_idx, iou))

    unmatched_gt = [i for i in range(len(gt_objects)) if i not in used_gt]
    unmatched_pred = [i for i in range(len(pred_objects)) if i not in used_pred]
    return matches, unmatched_gt, unmatched_pred


def interpolated_ap(recalls, precisions, recall_points=AP_RECALL_POINTS):
    """Calculate COCO-style interpolated AP over evenly spaced recall points."""
    if len(recalls) == 0:
        return 0.0

    sampled_precisions = []
    for recall_level in np.linspace(0.0, 1.0, recall_points):
        valid = precisions[recalls >= recall_level]
        sampled_precisions.append(float(np.max(valid)) if valid.size else 0.0)
    return float(np.mean(sampled_precisions))


def calculate_ap_metrics(
    gt_by_image,
    pred_by_image,
    num_classes,
    iou_thresholds=AP_IOU_THRESHOLDS,
):
    """
    Calculate per-class AP and mAP using confidence-ranked predictions.

    Matching is performed independently for each class and IoU threshold.
    Each prediction can match at most one GT object in the same image.
    Classes without GT objects are excluded from the mAP denominator.
    """
    per_class = {}

    for class_id in range(num_classes):
        gt_boxes_by_image = {}
        total_gt = 0
        for image_name, objects in gt_by_image.items():
            boxes = [
                box
                for gt_class, box in objects
                if int(gt_class) == class_id
            ]
            gt_boxes_by_image[image_name] = boxes
            total_gt += len(boxes)

        predictions = []
        for image_name, objects in pred_by_image.items():
            for pred_class, confidence, box in objects:
                if int(pred_class) == class_id:
                    predictions.append(
                        (float(confidence), image_name, box)
                    )
        predictions.sort(key=lambda item: item[0], reverse=True)

        ap_by_iou = {}
        if total_gt == 0:
            for threshold in iou_thresholds:
                ap_by_iou[threshold] = float("nan")
        else:
            for threshold in iou_thresholds:
                matched_gt = {
                    image_name: set()
                    for image_name in gt_boxes_by_image
                }
                true_positive = np.zeros(len(predictions), dtype=float)
                false_positive = np.zeros(len(predictions), dtype=float)

                for pred_index, (_, image_name, pred_box) in enumerate(predictions):
                    gt_boxes = gt_boxes_by_image.get(image_name, [])
                    best_iou = 0.0
                    best_gt_index = -1

                    for gt_index, gt_box in enumerate(gt_boxes):
                        if gt_index in matched_gt[image_name]:
                            continue
                        iou = calculate_iou_xywhn(gt_box, pred_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_index = gt_index

                    if best_gt_index >= 0 and best_iou >= threshold:
                        true_positive[pred_index] = 1.0
                        matched_gt[image_name].add(best_gt_index)
                    else:
                        false_positive[pred_index] = 1.0

                cumulative_tp = np.cumsum(true_positive)
                cumulative_fp = np.cumsum(false_positive)
                recalls = cumulative_tp / total_gt
                precisions = np.divide(
                    cumulative_tp,
                    cumulative_tp + cumulative_fp,
                    out=np.zeros_like(cumulative_tp),
                    where=(cumulative_tp + cumulative_fp) > 0,
                )
                ap_by_iou[threshold] = interpolated_ap(
                    recalls,
                    precisions,
                )
        valid_ap = [
            value
            for value in ap_by_iou.values()
            if not np.isnan(value)
        ]
        per_class[class_id] = {
            "num_gt": total_gt,
            "num_predictions": len(predictions),
            "ap50": ap_by_iou.get(0.50, float("nan")),
            "ap50_95": (
                float(np.mean(valid_ap))
                if valid_ap
                else float("nan")
            ),
            "ap_by_iou": ap_by_iou,
        }
    valid_ap50 = [
        values["ap50"]
        for values in per_class.values()
        if not np.isnan(values["ap50"])
    ]
    valid_ap50_95 = [
        values["ap50_95"]
        for values in per_class.values()
        if not np.isnan(values["ap50_95"])
    ]
    return {
        "per_class": per_class,
        "map50": float(np.mean(valid_ap50)) if valid_ap50 else 0.0,
        "map50_95": (
            float(np.mean(valid_ap50_95))
            if valid_ap50_95
            else 0.0
        ),
    }

def synchronize_cuda():
    """Synchronize CUDA before timing when a CUDA device is active."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
# ================= UTIL =================
def log_system_info():
    logger.info("=" * 60)
    logger.info("%s v%s", APP_NAME, APP_VERSION)
    logger.info("OS      : %s", platform.system())
    logger.info("Python  : %s", platform.python_version())
    logger.info("OpenCV  : %s", cv2.__version__)
    logger.info("PyTorch : %s", torch.__version__)
    logger.info("CUDA    : %s", torch.cuda.is_available())
    logger.info("=" * 60)

def draw_gt(img, boxes, class_names):
    h, w = img.shape[:2]
    for cls, (x, y, bw, bh) in boxes:
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)
        
        color = get_class_color(cls)
        label = class_names.get(cls, f"Class{cls}")
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img

def draw_pred(img, boxes, class_names):
    h, w = img.shape[:2]
    for cls, conf, (x, y, bw, bh) in boxes:
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)
        color = get_class_color(cls)
        label = f"{class_names.get(cls, f'Class{cls}')} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img

# ================= APP =================
class ADaCITraApp:
    def __init__(self, root):
        self.root = root
        root.title(APP_TITLE)
        self.class_names = {}

        self.model = None
        self.image_dir = None
        self.gt_dir = None
        self.output_dir = None
        self.input_source = None

        self.images = []
        self.idx = 0
        self.pred_results = {}
        self.ap_pred_results = {}
        self.gt_all = {}

        self.video_path = None
        self.video_cap = None
        self.video_running = False
        self.video_paused = False
        self.video_job = None

        self.popup_items = []
        
        self.total_inference_time = 0
        self.fps = 0
        self.view_mode = "image"
        self.prev_frame_time = 0
        self.current_fps = 0

        self.build_gui()
        self.init_canvas()
        
    def load_yaml(self, yaml_path):
        import yaml
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        names = data.get("names", [])

        # format list
        if isinstance(names, list):
            self.class_names = {
                i: name
                for i, name in enumerate(names)
            }

        # format dict
        elif isinstance(names, dict):
            self.class_names = {
                int(k): v
                for k, v in names.items()
            }
        else:
            raise ValueError("The YAML 'names' field must be a list or dictionary.")

        expected_ids = set(range(len(self.class_names)))
        if set(self.class_names) != expected_ids:
            raise ValueError(
                "Class IDs in YAML must be contiguous and start at 0. "
                f"Found: {sorted(self.class_names)}"
            )

        logger.info("===== YAML LOADED =====")
        logger.info(self.class_names)

    def select_yaml(self):
        path = filedialog.askopenfilename(
            filetypes=[("YAML Files", "*.yaml *.yml")]
        )
        logger.info("SELECT YAML CLICKED")

        if path:
            logger.info("YAML PATH:%s", path)
            try:
                self.load_yaml(path)
            except Exception as error:
                logger.exception("Failed to load YAML.")
                messagebox.showerror("YAML Error", str(error))
                return

            logger.info("AFTER LOAD YAML")
            logger.info(self.class_names)

            # REFRESH PREVIEW
            if self.images:
                self.show_current_image()

            self.show_popup("YAML Loaded")

    # ---------- GUI ----------
    def build_gui(self):
        f1 = tk.Frame(self.root)
        f1.pack(pady=5)
        tk.Button(f1, text="Load Model", width=18, command=self.load_model).pack(side=tk.LEFT, padx=5)
        tk.Button(f1,text="Load YAML",width=18,command=self.select_yaml).pack(side=tk.LEFT, padx=5)
        tk.Button(f1, text="Load Images", width=18, command=self.load_images).pack(side=tk.LEFT, padx=5)
        tk.Button(f1, text="Load GT Folder", width=18, command=self.load_gt).pack(side=tk.LEFT, padx=5)
        tk.Button(f1, text="Output Path", width=18, command=self.set_output).pack(side=tk.LEFT, padx=5)
        tk.Button(f1, text="Load Video", width=18, command=self.load_video).pack(side=tk.LEFT, padx=5)

        f2 = tk.Frame(self.root)
        f2.pack(pady=5)
        # progress bar
        self.progress = ttk.Progressbar(self.root, orient="horizontal",length=500,mode="determinate")
        # Removed self.progress.pack(pady=5) from here
        self.progress_label = tk.Label(self.root, text="")
        tk.Button(f1,text="USB Camera",width=18,command=self.load_USBcam).pack(side=tk.LEFT, padx=5)
        tk.Button(f2, text="RUN INFERENCE", width=22, bg="green",fg="white", command=self.run_inference).pack(side=tk.LEFT, padx=10)
        tk.Button(f2, text="GENERATE REPORT", width=22, bg="blue",fg="white", command=self.generate_report).pack(side=tk.LEFT, padx=10)
        tk.Button(f2, text="EXPORT VIDEO", width=22,command=self.export_video).pack(side=tk.LEFT, padx=10)
        self.btn_pause = tk.Button(f2, text="PAUSE VIDEO", width=22, bg="orange", fg="white", command=self.toggle_video)
        self.btn_pause.pack(side=tk.LEFT, padx=10)

        self.title_label = tk.Label(self.root, text="", font=("Arial", 14, "bold"))
        self.title_label.pack()

        self.canvas = tk.Canvas(self.root, width=CANVAS_W, height=CANVAS_H, bg="gray")
        self.canvas.pack()

        f3 = tk.Frame(self.root)
        f3.pack(pady=5)
        tk.Button(f3, text="<< Prev", width=15, command=self.prev_img).pack(side=tk.LEFT, padx=10)
        tk.Button(f3, text="Next >>", width=15, command=self.next_img).pack(side=tk.LEFT, padx=10)
          
    def init_canvas(self):
        blank = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        cv2.putText(blank, "ADaCITra Preview",
                    (CANVAS_W // 2 - 200, CANVAS_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
        self.update_canvas(blank)

    def update_canvas(self, img):
        h, w = img.shape[:2]
        if self.view_mode == "image":
        # jika gambar terdiri dari GT dan Prediction
            if w % 2 == 0 and h > 40:
                half = w // 2
                gt_img = img[:, :half]
                pred_img = img[:, half:]

                img = self.build_compare_view(gt_img, pred_img)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

    def show_current_image(self):
        self.view_mode = "image"
        self.input_source = "image"
        if not self.images:
            return
        name = self.images[self.idx]
        img_path = os.path.join(self.image_dir, name)
        if self.is_video(img_path):
            self.process_video(img_path)
            return

        # IMAGE MODE
        img = cv2.imread(img_path)

        # Resize the original image to fit IMG_W and IMG_H while maintaining aspect ratio
        original_h, original_w = img.shape[:2]
        scale = min(IMG_W / original_w, IMG_H / original_h)
        # scale = min(IMG_W / original_w, IMG_H / original_w)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create blank canvases of IMG_H x IMG_W for GT and Prediction views
        # Initialize with a dark gray color for padding
        gt_canvas = np.full((IMG_H, IMG_W, 3), (50, 50, 50), dtype=np.uint8)
        pred_canvas = np.full((IMG_H, IMG_W, 3), (50, 50, 50), dtype=np.uint8)

        # Calculate offsets to center the resized image within the canvas
        x_offset = (IMG_W - new_w) // 2
        y_offset = (IMG_H - new_h) // 2

        # Copy the resized image to temporary images for drawing
        gt_draw_img = resized_img.copy()
        pred_draw_img = resized_img.copy()

        # Draw GT annotations if available
        if name in self.gt_all:
            gt_draw_img = draw_gt(gt_draw_img, self.gt_all[name],self.class_names)

        # Draw prediction annotations if available
        if name in self.pred_results:
            pred_draw_img = draw_pred(pred_draw_img, self.pred_results[name],self.class_names)

        # Place the drawn images onto their respective padded canvases
        gt_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = gt_draw_img
        pred_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = pred_draw_img

        # Combine the padded GT and Prediction canvases horizontally
        combined = np.hstack([gt_canvas, pred_canvas])

        self.update_canvas(combined)
        self.title_label.config(text=f"Image: {name}   [{self.idx+1}/{len(self.images)}]")
    
    def open_video_source(self, source):
        logger.info("Opening video source: %s", source)
        if self.video_cap is not None:
            self.video_cap.release()

        # Windows → DirectShow
        if platform.system() == "Windows" and isinstance(source, int):
            self.video_cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self.video_cap = cv2.VideoCapture(source)
        logger.info("isOpened = %s", self.video_cap.isOpened())

        self.video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_CAMERA_WIDTH)
        self.video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_CAMERA_HEIGHT)
        self.video_cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
        logger.info("Buffer size = %s",self.video_cap.get(cv2.CAP_PROP_BUFFERSIZE))
        try:
            logger.info("Backend = %s", self.video_cap.getBackendName())
        except Exception as e:
           logger.warning("Cannot get backend name: %s", e)

        if not self.video_cap.isOpened():
            return False
        return True
    
    def load_USBcam(self):
        self.view_mode = "video"
        self.input_source = "usb_camera"
        camera_index = get_usb_camera_index()
        logger.info("Opening USB camera (index=%s)", camera_index)
        if not self.open_video_source(camera_index):
            messagebox.showerror("Error", "Cannot open USB camera")
            return

        self.video_running = True
        self.video_paused = False
        self.play_video()

    def load_IPcam(self):
        self.view_mode = "video"
        self.input_source = "ip_camera"
        rtsp = self.build_rtsp_url()
        if not self.open_video_source(rtsp):
            messagebox.showerror("Error", "Cannot open IP Camera")
            return

        self.video_running = True
        self.video_paused = False
        self.play_video()

    def load_video(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv")
            ]
        )
        if path:
            self.video_path = path
            self.show_popup("Video Inference Started")
            self.process_video(path)
          
    def process_video(self, video_path):
        self.view_mode = "video"
        self.input_source = "video"
        if not self.open_video_source(video_path):
            messagebox.showerror(
                "Error",
                f"Cannot open source:\n{video_path}"
            )
            return

        self.video_running = True
        self.video_paused = False
        self.show_popup("Video Started")
        self.play_video()

    def play_video(self):
        if not self.video_running:
            return
        if self.video_paused:
            return
        
        ret, frame = self.video_cap.read()
        if not ret:
            if self.video_cap is not None:
                self.video_cap.release()
                self.video_cap = None
            self.video_running = False
            if self.input_source == "video":
                self.show_popup("Video Finished")
            elif self.input_source == "usb_camera":
                self.show_popup("USB Camera Disconnected")
            elif self.input_source == "ip_camera":
                self.show_popup("IP Camera Disconnected")
            return

        if self.model is None:
            messagebox.showwarning(
                "Warning",
                "Please load YOLO model first."
            )
            self.stop_video()
            return
        # Measure Ultralytics preprocessing + inference + post-processing.
        synchronize_cuda()
        start_time = time.perf_counter()
        results = self.model(
            frame,
            imgsz=INFERENCE_IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )[0]
        synchronize_cuda()
        end_time = time.perf_counter()

        preds = []
        for b in results.boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            box = b.xywhn[0].tolist()
            preds.append((cls, conf, box))
        annotated = draw_pred(frame.copy(), preds, self.class_names)
        inference_time = end_time - start_time
        if inference_time > 0:
            self.current_fps = 1 / inference_time
        # ================= HUD =================
        # background
        cv2.rectangle(annotated,(10, 10),(240, 70),(40, 40, 40),-1)
        # border
        cv2.rectangle(annotated,(10, 10),(240, 70),(255,255,255),1)
        # title
        cv2.putText(annotated,"VIDEO INFERENCE",(20, 32),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)
        # fps
        cv2.putText(annotated,f"FPS : {self.current_fps:.2f}",(20, 58),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)

        # resize to canvas
        annotated = cv2.resize(annotated,(CANVAS_W, CANVAS_H))
        self.update_canvas(annotated)
        self.video_job = self.root.after(10,self.play_video)

    def stop_video(self):
        self.video_running = False
        if self.video_job is not None:
            self.root.after_cancel(
                self.video_job
            )
            self.video_job = None
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        self.show_popup("Video Stopped")

        self.play_video()

    def toggle_video(self):
        logger.info("CLICK | running=%s paused=%s",self.video_running,self.video_paused)
        if not self.video_running:
            return
        self.video_paused = not self.video_paused

        if self.video_paused:
            if self.video_job is not None:
                self.root.after_cancel(
                    self.video_job
                )
                self.video_job = None
            self.btn_pause.config(
                text="PLAY VIDEO",
                bg="green"
            )
            self.show_popup("Video Paused")
        else:
            self.btn_pause.config(
                text="PAUSE VIDEO",
                bg="orange"
            )
        self.show_popup("Video Playing")
        self.play_video()
    
    def build_compare_view(self, gt_img, pred_img):
        h, w = gt_img.shape[:2]

        title_h = 40
        canvas = np.zeros((h + title_h, w * 2, 3), dtype=np.uint8)
        # background
        canvas[:] = (30, 30, 30)
        # judul kiri
        cv2.putText(canvas, "GROUND TRUTH",(w//2 - 90, 28),cv2.FONT_HERSHEY_SIMPLEX,0.8, (255,255,255), 2)
        # judul kanan
        cv2.putText(canvas, "PREDICTION",(w + w//2 - 90, 28),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        # garis pemisah tengah
        cv2.line(canvas, (w, 0), (w, h + title_h), (120,120,120), 2)
        # masukkan gambar
        canvas[title_h:title_h+h, 0:w] = gt_img
        canvas[title_h:title_h+h, w:w*2] = pred_img
        return canvas
    
    # ---------- POPUP ----------
    def show_popup(self, text):
        # 1. Get current canvas width
        canvas_width = self.canvas.winfo_width()
        
        # 2. Define popup dimensions
        popup_width = 360
        popup_height = 50
        padding_top = 20
        
        # 3. Calculate horizontal center
        # x1 is center minus half the popup width
        x1 = (canvas_width / 2) - (popup_width / 2)
        x2 = x1 + popup_width
        y1 = padding_top
        y2 = y1 + popup_height

        self.clear_popup()
        
        # Draw the rectangle centered
        rect = self.canvas.create_rectangle(x1, y1, x2, y2,fill="#222", outline="#e2dff0", width=2)
        
        # Draw the text at the absolute center point (canvas_width / 2)
        txt = self.canvas.create_text(canvas_width / 2, y1 + (popup_height / 2), text=text, fill="#e2dff0", font=("Arial", 14, "bold"))
        
        self.popup_items = [rect, txt]
        self.root.after(1000, self.clear_popup)

    def clear_popup(self):
        for i in self.popup_items:
            self.canvas.delete(i)
        self.popup_items.clear()
        self.popup_timer = None

    # ---------- LOADERS ----------
    def load_model(self):
        p = filedialog.askopenfilename(filetypes=[("YOLO Model", "*.pt")])
        if p:
            self.model = YOLO(p)
            self.show_popup("Model Loaded")

    def load_images(self):
        self.image_dir = filedialog.askdirectory()
        if self.image_dir:
            # Image mode is evaluated against YOLO ground-truth labels.
            # Videos are loaded separately through Load Video.
            supported_ext = (".jpg", ".jpeg", ".png", ".bmp")
            self.images = sorted([
                f for f in os.listdir(self.image_dir)
                if f.lower().endswith(supported_ext)
            ])
            self.idx = 0
            self.show_popup(f"{len(self.images)} Media Files Loaded")
    
    def is_image(self, filename):
        image_ext = (".jpg", ".jpeg", ".png", ".bmp")
        return filename.lower().endswith(image_ext)
    
    def is_video(self, filename):
        video_ext = (".mp4", ".avi", ".mov", ".mkv")
        return filename.lower().endswith(video_ext)

    def load_gt(self):
        self.gt_dir = filedialog.askdirectory()
        if self.gt_dir:
            self.show_popup("GT Folder Loaded")

    def set_output(self):
        self.output_dir = filedialog.askdirectory()
        if self.output_dir:
            for sub in ["images", "video", "csv", "confusion"]:
                os.makedirs(os.path.join(self.output_dir, sub), exist_ok=True)
            self.show_popup("Output Path Set")

    # ---------- CORE ----------
    def run_inference(self):
        if not self.images or not self.image_dir:
            messagebox.showerror("Error", "Please load an image folder first.")
            return
        if not self.model:
            messagebox.showerror("Error", "Model not loaded. Please load a YOLO model first.")
            return
        if not self.class_names:
            messagebox.showerror("Error", "Please load the dataset YAML first.")
            return
        if not self.output_dir:
            messagebox.showerror("Error", "Please set an output directory first.")
            return

        self.progress.pack(pady=5)
        self.progress_label.pack()
        self.progress["maximum"] = len(self.images)
        self.progress["value"] = 0
        threading.Thread(
            target=self.run_inference_worker,
            daemon=True
        ).start()

    def _update_inference_progress(self, value, total):
        self.progress["value"] = value
        self.progress_label.config(text=f"Processing {value}/{total}")

    def _finish_inference(self, error_message=None):
        self.progress["value"] = 0
        self.progress_label.config(text="")
        self.progress.pack_forget()
        self.progress_label.pack_forget()
        if error_message:
            messagebox.showerror("Inference Error", error_message)
            return
        self.idx = 0
        self.show_current_image()
        self.show_popup("Inference Completed")

    def run_inference_worker(self):
        self.pred_results.clear()
        self.ap_pred_results.clear()
        self.gt_all.clear()
        total = len(self.images)
        csv_dir = os.path.join(self.output_dir, "csv")
        image_output_dir = os.path.join(self.output_dir, "images")
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(image_output_dir, exist_ok=True)

        prediction_rows = [["image", "class_id", "confidence", "x", "y", "w", "h"]]
        ap_prediction_rows = [["image", "class_id", "confidence", "x", "y", "w", "h"]]
        measured_times = []

        try:
            # Warm up the loaded model before collecting timing measurements.
            first_image_path = os.path.join(self.image_dir, self.images[0])
            warmup_image = cv2.imread(first_image_path)
            if warmup_image is None:
                raise ValueError(f"Cannot read the first image: {first_image_path}")
            for _ in range(WARMUP_RUNS):
                self.model(
                    warmup_image,
                    imgsz=INFERENCE_IMAGE_SIZE,
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False,
                )
            synchronize_cuda()

            for i, name in enumerate(self.images):
                img_path = os.path.join(self.image_dir, name)
                original_img = cv2.imread(img_path)
                if original_img is None:
                    logger.warning("Skipping unreadable image: %s", img_path)
                    self.root.after(0, self._update_inference_progress, i + 1, total)
                    continue

                synchronize_cuda()
                start_time = time.perf_counter()
                result = self.model(
                    original_img,
                    imgsz=INFERENCE_IMAGE_SIZE,
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False,
                )[0]
                synchronize_cuda()
                measured_times.append(time.perf_counter() - start_time)

                preds = []
                for box_result in result.boxes:
                    cls = int(box_result.cls[0])
                    conf = float(box_result.conf[0])
                    box = box_result.xywhn[0].tolist()
                    if cls not in self.class_names:
                        logger.warning(
                            "Skipping prediction with unknown class ID %s in %s",
                            cls,
                            name,
                        )
                        continue
                    preds.append((cls, conf, box))
                    prediction_rows.append([name, cls, conf, *box])
                self.pred_results[name] = preds

                # A separate low-confidence pass is required for AP/mAP.
                # It is intentionally excluded from the operating-point timing.
                ap_result = self.model(
                    original_img,
                    imgsz=INFERENCE_IMAGE_SIZE,
                    conf=AP_MIN_CONFIDENCE,
                    iou=NMS_IOU_THRESHOLD,
                    max_det=MAX_DETECTIONS,
                    agnostic_nms=False,
                    verbose=False,
                )[0]
                ap_preds = []
                for box_result in ap_result.boxes:
                    cls = int(box_result.cls[0])
                    confidence = float(box_result.conf[0])
                    box = box_result.xywhn[0].tolist()
                    if cls not in self.class_names:
                        continue
                    ap_preds.append((cls, confidence, box))
                    ap_prediction_rows.append(
                        [name, cls, confidence, *box]
                    )
                self.ap_pred_results[name] = ap_preds

                if self.gt_dir:
                    gt_path = os.path.join(
                        self.gt_dir,
                        os.path.splitext(name)[0] + ".txt",
                    )
                    if os.path.exists(gt_path):
                        valid_gt = []
                        for gt_class, gt_box in load_yolo_label(gt_path):
                            gt_class = int(gt_class)
                            if gt_class not in self.class_names:
                                logger.warning(
                                    "Skipping GT with unknown class ID %s in %s",
                                    gt_class,
                                    name,
                                )
                                continue
                            valid_gt.append((gt_class, gt_box))
                        self.gt_all[name] = valid_gt
                    else:
                        logger.warning("Ground-truth file not found: %s", gt_path)
                        self.gt_all[name] = []

                output_img = draw_pred(original_img.copy(), preds, self.class_names)
                cv2.imwrite(os.path.join(image_output_dir, name), output_img)
                self.root.after(0, self._update_inference_progress, i + 1, total)

            self.total_inference_time = sum(measured_times)
            measured_count = len(measured_times)
            self.fps = (
                measured_count / self.total_inference_time
                if self.total_inference_time > 0
                else 0.0
            )
            latency_ms = (
                1000.0 * self.total_inference_time / measured_count
                if measured_count > 0
                else 0.0
            )

            with open(
                os.path.join(csv_dir, "predictions.csv"),
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                csv.writer(file).writerows(prediction_rows)

            with open(
                os.path.join(csv_dir, "ap_predictions.csv"),
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                csv.writer(file).writerows(ap_prediction_rows)

            with open(
                os.path.join(csv_dir, "summary_report.csv"),
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.writer(file)
                writer.writerow(["Metric", "Value"])
                writer.writerow(["Total Images", total])
                writer.writerow(["Measured Images", measured_count])
                writer.writerow(["Image Size", INFERENCE_IMAGE_SIZE])
                writer.writerow(["Confidence Threshold", CONFIDENCE_THRESHOLD])
                writer.writerow(["AP Minimum Confidence", AP_MIN_CONFIDENCE])
                writer.writerow(["Warm-up Runs", WARMUP_RUNS])
                writer.writerow(["Model Pipeline Time (s)", round(self.total_inference_time, 4)])
                writer.writerow(["FPS", round(self.fps, 2)])
                writer.writerow(["Latency (ms/image)", round(latency_ms, 2)])
                writer.writerow([
                    "Timing Scope",
                    "Ultralytics preprocessing + inference + post-processing; "
                    "excludes AP pass, disk read, drawing, GUI update, and file write",
                ])

            self.root.after(0, self._finish_inference)
        except Exception as error:
            logger.exception("Image inference failed.")
            self.root.after(0, self._finish_inference, str(error))

    def process_video_file(self, video_path, name):
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            # if not ret:
            #     break
            if not ret:
                self.video_running = False
                cap.release()
                self.show_popup("Video Finished")
                return
            results = self.model(
                frame,
                imgsz=INFERENCE_IMAGE_SIZE,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
            )
            annotated = results[0].plot()
            cv2.imshow(name, annotated)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    def export_video(self):
        # Start video export in a separate thread
        if not self.image_dir or not self.images:
            messagebox.showerror("Error", "Please load images first.")
            logger.error("Error: Images not loaded or image_dir not set.")
            return
        if not self.output_dir:
            messagebox.showerror("Error", "Please set an output directory first.")
            logger.error("Error: Output directory not set.")
            return

        # --- GUI operation in main thread for file dialog and initial progress bar setup ---
        video_output_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            # initialdir=self.output_dir,
            initialdir=os.path.join(self.output_dir, "video"), # Start in the 'video' subfolder
            title="Save Video As",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.* ")])
        if not video_output_path:
            messagebox.showinfo("ADaCITra", "Video export cancelled.")
            return

        # Show progress bar
        logger.info("Attempting to pack progress bar.")
        self.progress.pack(pady=5)
        self.progress_label.pack()
        self.progress["maximum"] = len(self.images)
        self.progress["value"] = 0
        self.root.update_idletasks() # Force update to ensure progress bar is shown immediately

        # --- Start worker thread for non-GUI heavy lifting ---
        threading.Thread(target=self.export_video_worker, args=(video_output_path,), daemon=True).start()
    
    def export_video_worker(self, video_output_path):
        logger.info("Entering export_video_worker with path: %s", video_output_path)
        total_frames = len(self.images)
        logger.info(f"Total frames to export: {total_frames}")

        had_error = False # Initialize had_error
        vw = None # Initialize VideoWriter to None

        try:
            # Get dimensions for VideoWriter by processing the first image
            name = self.images[0]
            img_path = os.path.join(self.image_dir, name)
            img = cv2.imread(img_path)

            original_h, original_w = img.shape[:2]
            scale = min(IMG_W / original_w, IMG_H / original_h)
            new_w = int(original_w * scale)
            new_h = int(original_h * scale)
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            gt_canvas = np.full((IMG_H, IMG_W, 3), (50, 50, 50), dtype=np.uint8)
            pred_canvas = np.full((IMG_H, IMG_W, 3), (50, 50, 50), dtype=np.uint8)

            x_offset = (IMG_W - new_w) // 2
            y_offset = (IMG_H - new_h) // 2

            gt_draw_img = resized_img.copy()
            pred_draw_img = resized_img.copy()

            if name in self.gt_all:
                gt_draw_img = draw_gt(gt_draw_img, self.gt_all[name],self.class_names)
            if name in self.pred_results:
                pred_draw_img = draw_pred(pred_draw_img, self.pred_results[name],self.class_names)

            gt_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = gt_draw_img
            pred_canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = pred_draw_img

                        # This combined frame will have the dimensions for the video
            first_frame = self.build_compare_view(gt_canvas, pred_canvas)
            frame_height, frame_width = first_frame.shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
            fps = 5 # Frames per second
            vw = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

            if not vw.isOpened():
                raise IOError("Error: Could not open video writer.")

            # Write the first frame
            vw.write(first_frame)
            self.root.after(0, self._update_export_progress, 1, total_frames)

                        # Loop for the rest of the images
            for i in range(1, total_frames):
                name = self.images[i]
                img_path = os.path.join(self.image_dir, name)
                img = cv2.imread(img_path)

                original_h, original_w = img.shape[:2]
                scale = min(IMG_W / original_w, IMG_H / original_h)
                new_w = int(original_w * scale)
                new_h = int(original_h * scale)
                resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                gt_canvas_loop = np.full((IMG_H, IMG_W, 3), (50, 50, 50), dtype=np.uint8)
                pred_canvas_loop = np.full((IMG_H, IMG_W, 3), (50, 50, 50), dtype=np.uint8)

                x_offset_loop = (IMG_W - new_w) // 2
                y_offset_loop = (IMG_H - new_h) // 2

                gt_draw_img_loop = resized_img.copy()
                pred_draw_img_loop = resized_img.copy()

                if name in self.gt_all:
                    gt_draw_img_loop = draw_gt(gt_draw_img_loop, self.gt_all[name],self.class_names)
                if name in self.pred_results:
                    pred_draw_img_loop = draw_pred(pred_draw_img_loop, self.pred_results[name],self.class_names)

                gt_canvas_loop[y_offset_loop:y_offset_loop+new_h, x_offset_loop:x_offset_loop+new_w] = gt_draw_img_loop
                pred_canvas_loop[y_offset_loop:y_offset_loop+new_h, x_offset_loop:x_offset_loop+new_w] = pred_draw_img_loop

                current_frame = self.build_compare_view(gt_canvas_loop, pred_canvas_loop)
                vw.write(current_frame)
                self.root.after(
                    0,
                    self._update_export_progress,
                    i + 1,
                    total_frames,
                )

        except Exception as e:
            error_message = str(e)
            self.root.after(
                0,
                lambda message=error_message: messagebox.showerror(
                    "Error during video export",
                    message,
                ),
            )
            logger.info(f"Exception during video export: {e}")
            had_error = True
        finally:
            if vw is not None: # Ensure vw was successfully initialized before releasing
                vw.release()
            logger.info("Hiding progress bar.")
            self.root.after(0, self._hide_export_progress, video_output_path, had_error)

    def _update_export_progress(self, value, total):
        self.progress["value"] = value
        self.progress_label.config(text=f"Exporting frame {value}/{total}")

    def _hide_export_progress(self, video_output_path, had_error):
        self.progress["value"] = 0
        self.progress_label.config(text="")
        self.progress.pack_forget()
        self.progress_label.pack_forget()
        self.show_popup(f"Video Export Completed" if not had_error else "Video Export Failed")

    def prev_img(self):
        if self.images and self.idx > 0:
            self.idx -= 1
            self.show_current_image()
        else:
            messagebox.showinfo("ADaCITra", "No previous image.")

    def next_img(self):
        if self.images and self.idx < len(self.images) - 1:
            self.idx += 1
            self.show_current_image()
        else:
            messagebox.showinfo("ADaCITra", "No next image.")

    def process_and_save_image(self, name):
        img = cv2.imread(os.path.join(self.image_dir, name))
        if img is None:
            logger.warning("Cannot save comparison for unreadable image: %s", name)
            return
        gt_image = draw_gt(
            img.copy(),
            self.gt_all.get(name, []),
            self.class_names,
        )
        pred_image = draw_pred(
            img.copy(),
            self.pred_results.get(name, []),
            self.class_names,
        )
        gt_image = cv2.resize(gt_image, (IMG_W, IMG_H))
        pred_image = cv2.resize(pred_image, (IMG_W, IMG_H))
        out = self.build_compare_view(gt_image, pred_image)
        cv2.imwrite(os.path.join(self.output_dir, "images", name), out)

    def show_image(self):
        p = os.path.join(self.output_dir, "images", self.images[self.idx])
        if os.path.exists(p):
            self.update_canvas(cv2.imread(p))

    def generate_report(self):
        if not self.output_dir:
            messagebox.showerror("Error", "Please set an output directory first.")
            return      
        if not self.pred_results:
            messagebox.showinfo("Info", "No inference results found. Please run inference first.")
            return
        if not self.gt_all:
            messagebox.showwarning(
                "Warning",
                "No ground truth was loaded. Quantitative evaluation cannot be generated.",
            )
            return
        
        # This will call the save_confusion_matrix method which now includes all metrics
        try:
            self.save_confusion_matrix()
            self.show_popup("Evaluation Report Generated")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {e}")
    
    # ---------- CONFUSION MATRIX AND METRICS ----------
    def save_confusion_matrix(self):
        logger.info("save_confusion_matrix called.")
        logger.info(f"Output directory: {self.output_dir}")
        if not self.class_names:
            raise ValueError("Class names are empty. Load the dataset YAML first.")

        num_classes = len(self.class_names)
        expected_ids = set(range(num_classes))
        if set(self.class_names) != expected_ids:
            raise ValueError(
                "YAML class IDs must be contiguous and start at 0. "
                f"Found: {sorted(self.class_names)}"
            )

        confusion_dir = os.path.join(self.output_dir, "confusion")
        os.makedirs(confusion_dir, exist_ok=True)

        # The final index has different meanings on the two axes:
        # last row = prediction without GT (FP)
        # last column = GT without prediction (FN)
        unmatched_index = num_classes
        cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
        class_ious = {class_id: [] for class_id in range(num_classes)}
        details = [[
            "image",
            "gt_class_id",
            "gt_class_name",
            "pred_class_id",
            "pred_class_name",
            "confidence",
            "iou",
            "match_status",
        ]]

        for name in self.images:
            gt_objects = self.gt_all.get(name, [])
            pred_objects = self.pred_results.get(name, [])
            matches, unmatched_gt, unmatched_pred = match_detections(
                gt_objects,
                pred_objects,
                IOU_THRESHOLD,
            )

            for gt_idx, pred_idx, iou in matches:
                gt_class = int(gt_objects[gt_idx][0])
                pred_class = int(pred_objects[pred_idx][0])
                confidence = float(pred_objects[pred_idx][1])
                cm[gt_class, pred_class] += 1
                class_ious[gt_class].append(iou)
                status = "true_positive" if gt_class == pred_class else "misclassified"
                details.append([
                    name,
                    gt_class,
                    self.class_names[gt_class],
                    pred_class,
                    self.class_names[pred_class],
                    f"{confidence:.6f}",
                    f"{iou:.6f}",
                    status,
                ])

            for gt_idx in unmatched_gt:
                gt_class = int(gt_objects[gt_idx][0])
                cm[gt_class, unmatched_index] += 1
                details.append([
                    name,
                    gt_class,
                    self.class_names[gt_class],
                    "",
                    "No Prediction Match",
                    "",
                    "0.000000",
                    "false_negative",
                ])

            for pred_idx in unmatched_pred:
                pred_class = int(pred_objects[pred_idx][0])
                confidence = float(pred_objects[pred_idx][1])
                cm[unmatched_index, pred_class] += 1
                details.append([
                    name,
                    "",
                    "No GT Match",
                    pred_class,
                    self.class_names[pred_class],
                    f"{confidence:.6f}",
                    "0.000000",
                    "false_positive",
                ])

        if cm.sum() == 0:
            self.show_popup("Confusion matrix skipped (no GTs or predictions found)")
            logger.info("Confusion matrix skipped: no evaluable objects.")
            return

        tp_counts = np.diag(cm[:num_classes, :num_classes]).astype(np.int64)
        fp_counts = cm[:, :num_classes].sum(axis=0) - tp_counts
        fn_counts = cm[:num_classes, :].sum(axis=1) - tp_counts

        total_tp = int(tp_counts.sum())
        total_fp = int(fp_counts.sum())
        total_fn = int(fn_counts.sum())
        precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
        recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        matched_region = cm[:num_classes, :num_classes]
        matched_count = int(matched_region.sum())
        classification_accuracy = total_tp / matched_count if matched_count else 0.0

        all_matched_ious = [
            iou
            for values in class_ious.values()
            for iou in values
        ]
        mean_iou = (
            float(np.mean(all_matched_ious))
            if all_matched_ious
            else 0.0
        )

        if not self.ap_pred_results:
            raise ValueError(
                "AP predictions are unavailable. Run image inference again "
                "using ADaCITra v0.4.2 before generating the report."
            )
        ap_results = calculate_ap_metrics(
            self.gt_all,
            self.ap_pred_results,
            num_classes,
            AP_IOU_THRESHOLDS,
        )

        metrics_data = [[
            "Metric",
            "Overall",
            *[
                f"Class {class_id} ({self.class_names[class_id]})"
                for class_id in range(num_classes)
            ],
        ]]
        metric_rows = {
            "Mean IoU (all spatial matches)": [mean_iou],
            "Precision": [precision],
            "Recall": [recall],
            "F1-score": [f1_score],
            "Classification accuracy (matched detections)": [classification_accuracy],
            "TP count": [total_tp],
            "FP count": [total_fp],
            "FN count": [total_fn],
        }

        per_class_values = {
            key: []
            for key in metric_rows
        }
        for class_id in range(num_classes):
            tp = int(tp_counts[class_id])
            fp = int(fp_counts[class_id])
            fn = int(fn_counts[class_id])
            class_precision = tp / (tp + fp) if tp + fp else 0.0
            class_recall = tp / (tp + fn) if tp + fn else 0.0
            class_f1 = (
                2 * class_precision * class_recall / (class_precision + class_recall)
                if class_precision + class_recall
                else 0.0
            )
            detected_gt = int(matched_region[class_id, :].sum())
            class_accuracy = tp / detected_gt if detected_gt else 0.0
            class_mean_iou = (
                float(np.mean(class_ious[class_id]))
                if class_ious[class_id]
                else 0.0
            )

            per_class_values["Mean IoU (all spatial matches)"].append(class_mean_iou)
            per_class_values["Precision"].append(class_precision)
            per_class_values["Recall"].append(class_recall)
            per_class_values["F1-score"].append(class_f1)
            per_class_values[
                "Classification accuracy (matched detections)"
            ].append(class_accuracy)
            per_class_values["TP count"].append(tp)
            per_class_values["FP count"].append(fp)
            per_class_values["FN count"].append(fn)

            logger.info(
                "Class %s: TP=%d FP=%d FN=%d Precision=%.4f Recall=%.4f "
                "F1=%.4f MatchedAccuracy=%.4f MeanIoU=%.4f",
                self.class_names[class_id],
                tp,
                fp,
                fn,
                class_precision,
                class_recall,
                class_f1,
                class_accuracy,
                class_mean_iou,
            )

        for metric_name, overall_values in metric_rows.items():
            overall = overall_values[0]
            values = per_class_values[metric_name]
            if "count" in metric_name.lower():
                metrics_data.append([metric_name, int(overall), *[int(v) for v in values]])
            else:
                metrics_data.append([
                    metric_name,
                    f"{overall:.4f}",
                    *[f"{value:.4f}" for value in values],
                ])

        metrics_data.append([
            "AP@0.50",
            f"{ap_results['map50']:.4f}",
            *[
                (
                    f"{ap_results['per_class'][class_id]['ap50']:.4f}"
                    if not np.isnan(
                        ap_results["per_class"][class_id]["ap50"]
                    )
                    else "N/A"
                )
                for class_id in range(num_classes)
            ],
        ])
        metrics_data.append([
            "AP@0.50:0.95",
            f"{ap_results['map50_95']:.4f}",
            *[
                (
                    f"{ap_results['per_class'][class_id]['ap50_95']:.4f}"
                    if not np.isnan(
                        ap_results["per_class"][class_id]["ap50_95"]
                    )
                    else "N/A"
                )
                for class_id in range(num_classes)
            ],
        ])

        metrics_data.extend([
            ["Overall P/R/F1 averaging", "micro", *([""] * num_classes)],
            ["IoU threshold", f"{IOU_THRESHOLD:.2f}", *([""] * num_classes)],
            [
                "Confidence threshold",
                f"{CONFIDENCE_THRESHOLD:.2f}",
                *([""] * num_classes),
            ],
            [
                "AP minimum confidence",
                f"{AP_MIN_CONFIDENCE:.3f}",
                *([""] * num_classes),
            ],
            [
                "AP interpolation",
                f"{AP_RECALL_POINTS}-point",
                *([""] * num_classes),
            ],
        ])

        with open(
            os.path.join(confusion_dir, "metrics_report.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            csv.writer(file).writerows(metrics_data)

        with open(
            os.path.join(confusion_dir, "evaluation_details.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            csv.writer(file).writerows(details)

        ap_headers = [
            "Class ID",
            "Class Name",
            "GT Count",
            "Prediction Count",
            "AP@0.50",
            "AP@0.50:0.95",
            *[
                f"AP@{threshold:.2f}"
                for threshold in AP_IOU_THRESHOLDS
            ],
        ]
        ap_rows = [ap_headers]
        for class_id in range(num_classes):
            class_result = ap_results["per_class"][class_id]
            ap_rows.append([
                class_id,
                self.class_names[class_id],
                class_result["num_gt"],
                class_result["num_predictions"],
                (
                    f"{class_result['ap50']:.6f}"
                    if not np.isnan(class_result["ap50"])
                    else "N/A"
                ),
                (
                    f"{class_result['ap50_95']:.6f}"
                    if not np.isnan(class_result["ap50_95"])
                    else "N/A"
                ),
                *[
                    (
                        f"{class_result['ap_by_iou'][threshold]:.6f}"
                        if not np.isnan(
                            class_result["ap_by_iou"][threshold]
                        )
                        else "N/A"
                    )
                    for threshold in AP_IOU_THRESHOLDS
                ],
            ])
        ap_rows.append([
            "Overall",
            "mAP",
            "",
            "",
            f"{ap_results['map50']:.6f}",
            f"{ap_results['map50_95']:.6f}",
            *[
                (
                    "{:.6f}".format(
                        np.mean([
                            ap_results["per_class"][class_id]["ap_by_iou"][threshold]
                            for class_id in range(num_classes)
                            if not np.isnan(
                                ap_results["per_class"][class_id]["ap_by_iou"][threshold]
                            )
                        ])
                    )
                    if any(
                        not np.isnan(
                            ap_results["per_class"][class_id]["ap_by_iou"][threshold]
                        )
                        for class_id in range(num_classes)
                    )
                    else "N/A"
                )
                for threshold in AP_IOU_THRESHOLDS
            ],
        ])
        with open(
            os.path.join(confusion_dir, "ap_metrics_report.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            csv.writer(file).writerows(ap_rows)

        self._save_ap_by_iou_plot(
            ap_results=ap_results,
            class_names=self.class_names,
            iou_thresholds=AP_IOU_THRESHOLDS,
            output_path=os.path.join(
                confusion_dir,
                "ap_by_iou.png",
            ),
        )

        row_labels = [
            self.class_names[class_id]
            for class_id in range(num_classes)
        ] + ["No GT Match (FP)"]
        column_labels = [
            self.class_names[class_id]
            for class_id in range(num_classes)
        ] + ["No Prediction Match (FN)"]

        with open(
            os.path.join(confusion_dir, "confusion_matrix.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["True Label \\ Predicted Label", *column_labels])
            for label, row in zip(row_labels, cm):
                writer.writerow([label, *row.tolist()])

        row_sums = cm.sum(axis=1, keepdims=True)
        normalized_cm = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0,
        )
        with open(
            os.path.join(confusion_dir, "confusion_matrix_normalized.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["True Label \\ Predicted Label", *column_labels])
            for label, row in zip(row_labels, normalized_cm):
                writer.writerow([label, *[f"{value:.6f}" for value in row]])

        self._save_confusion_plot(
            cm,
            row_labels,
            column_labels,
            os.path.join(confusion_dir, "confusion_matrix.png"),
            "Extended Confusion Matrix (Counts)",
            value_format="d",
        )

        self._save_confusion_plot(
            normalized_cm,
            row_labels,
            column_labels,
            os.path.join(confusion_dir, "confusion_matrix_normalized.png"),
            "Extended Confusion Matrix (Row-normalized)",
            value_format=".2f",
        )

        logger.info(
            "Overall metrics: Precision=%.4f Recall=%.4f F1=%.4f "
            "MatchedAccuracy=%.4f MeanIoU=%.4f",
            precision,
            recall,
            f1_score,
            classification_accuracy,
            mean_iou,
        )
        logger.info(
            "AP metrics: mAP@0.50=%.4f mAP@0.50:0.95=%.4f "
            "(%d-point interpolation, AP min confidence=%.3f)",
            ap_results["map50"],
            ap_results["map50_95"],
            AP_RECALL_POINTS,
            AP_MIN_CONFIDENCE,
        )
        logger.info("Evaluation report saved to: %s", confusion_dir)

    @staticmethod
    def _save_ap_by_iou_plot(
        ap_results,
        class_names,
        iou_thresholds,
        output_path,
    ):
        """
        Save class-wise AP and overall mAP across IoU thresholds.

        The graph is generated using Pillow to avoid loading an additional
        OpenMP runtime through Matplotlib on Windows.
        """
        image_width = 1400
        image_height = 900

        left_margin = 130
        right_margin = 310
        top_margin = 110
        bottom_margin = 130

        plot_left = left_margin
        plot_top = top_margin
        plot_right = image_width - right_margin
        plot_bottom = image_height - bottom_margin

        plot_width = plot_right - plot_left
        plot_height = plot_bottom - plot_top

        image = Image.new(
            "RGB",
            (image_width, image_height),
            "white",
        )
        draw = ImageDraw.Draw(image)

        def load_font(size, bold=False):
            font_candidates = [
                "arialbd.ttf" if bold else "arial.ttf",
                "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
            ]

            for font_name in font_candidates:
                try:
                    return ImageFont.truetype(font_name, size)
                except OSError:
                    continue

            return ImageFont.load_default()

        title_font = load_font(34, bold=True)
        axis_font = load_font(25, bold=True)
        tick_font = load_font(21)
        legend_font = load_font(21)
        legend_title_font = load_font(22, bold=True)

        def text_size(text, font):
            bbox = draw.textbbox((0, 0), str(text), font=font)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]

        # ---------------------------------------------------------
        # Graph title
        # ---------------------------------------------------------
        title = "AP across IoU Thresholds"

        title_width, _ = text_size(title, title_font)

        draw.text(
            ((image_width - title_width) / 2, 35),
            title,
            fill="black",
            font=title_font,
        )

        # ---------------------------------------------------------
        # Coordinate conversion
        # ---------------------------------------------------------
        thresholds = [float(value) for value in iou_thresholds]

        min_iou = min(thresholds)
        max_iou = max(thresholds)

        def x_coordinate(iou):
            if max_iou == min_iou:
                return plot_left

            ratio = (iou - min_iou) / (max_iou - min_iou)
            return plot_left + ratio * plot_width

        def y_coordinate(ap_value):
            value = min(max(float(ap_value), 0.0), 1.0)
            return plot_bottom - value * plot_height

        # ---------------------------------------------------------
        # Grid and Y-axis ticks
        # ---------------------------------------------------------
        for tick_index in range(11):
            ap_value = tick_index / 10.0
            y = y_coordinate(ap_value)

            draw.line(
                [(plot_left, y), (plot_right, y)],
                fill=(220, 220, 220),
                width=1,
            )

            label = f"{ap_value:.1f}"
            label_width, label_height = text_size(label, tick_font)

            draw.text(
                (
                    plot_left - label_width - 15,
                    y - label_height / 2,
                ),
                label,
                fill="black",
                font=tick_font,
            )

        # ---------------------------------------------------------
        # X-axis ticks
        # ---------------------------------------------------------
        for threshold in thresholds:
            x = x_coordinate(threshold)

            draw.line(
                [(x, plot_top), (x, plot_bottom)],
                fill=(235, 235, 235),
                width=1,
            )

            label = f"{threshold:.2f}"
            label_width, _ = text_size(label, tick_font)

            draw.text(
                (
                    x - label_width / 2,
                    plot_bottom + 18,
                ),
                label,
                fill="black",
                font=tick_font,
            )

        # ---------------------------------------------------------
        # Main axes
        # ---------------------------------------------------------
        draw.line(
            [(plot_left, plot_top), (plot_left, plot_bottom)],
            fill="black",
            width=3,
        )

        draw.line(
            [(plot_left, plot_bottom), (plot_right, plot_bottom)],
            fill="black",
            width=3,
        )

        # ---------------------------------------------------------
        # Axis labels
        # ---------------------------------------------------------
        x_axis_label = "IoU Threshold"
        x_label_width, _ = text_size(x_axis_label, axis_font)

        draw.text(
            (
                plot_left + (plot_width - x_label_width) / 2,
                image_height - 60,
            ),
            x_axis_label,
            fill="black",
            font=axis_font,
        )

        y_axis_label = "AP / mAP"

        # Draw the Y label on a temporary image, then rotate it.
        y_label_bbox = draw.textbbox(
            (0, 0),
            y_axis_label,
            font=axis_font,
        )

        y_label_width = y_label_bbox[2] - y_label_bbox[0]
        y_label_height = y_label_bbox[3] - y_label_bbox[1]

        y_label_image = Image.new(
            "RGBA",
            (y_label_width + 20, y_label_height + 20),
            (255, 255, 255, 0),
        )

        y_label_draw = ImageDraw.Draw(y_label_image)

        y_label_draw.text(
            (10, 5),
            y_axis_label,
            fill="black",
            font=axis_font,
        )

        y_label_image = y_label_image.rotate(
            90,
            expand=True,
        )

        image.paste(
            y_label_image,
            (
                20,
                int(
                    plot_top
                    + (plot_height - y_label_image.height) / 2
                ),
            ),
            y_label_image,
        )

        # ---------------------------------------------------------
        # Colors for each class
        # ---------------------------------------------------------
        colors = [
            (31, 119, 180),    # blue
            (255, 127, 14),    # orange
            (44, 160, 44),     # green
            (214, 39, 40),     # red
            (148, 103, 189),   # purple
            (140, 86, 75),     # brown
            (227, 119, 194),   # pink
            (127, 127, 127),   # gray
        ]

        legend_entries = []

        # ---------------------------------------------------------
        # Class-wise AP curves
        # ---------------------------------------------------------
        per_class = ap_results["per_class"]

        for class_index, class_id in enumerate(sorted(per_class)):
            class_result = per_class[class_id]
            color = colors[class_index % len(colors)]

            if isinstance(class_names, dict):
                class_name = class_names.get(
                    class_id,
                    f"Class {class_id}",
                )
            else:
                class_name = (
                    class_names[class_id]
                    if class_id < len(class_names)
                    else f"Class {class_id}"
                )

            points = []

            for threshold in thresholds:
                ap_value = class_result["ap_by_iou"].get(
                    threshold,
                    float("nan"),
                )

                if np.isnan(ap_value):
                    continue

                points.append(
                    (
                        x_coordinate(threshold),
                        y_coordinate(ap_value),
                    )
                )

            if len(points) >= 2:
                draw.line(
                    points,
                    fill=color,
                    width=4,
                )

            for x, y in points:
                radius = 6

                draw.ellipse(
                    [
                        x - radius,
                        y - radius,
                        x + radius,
                        y + radius,
                    ],
                    fill=color,
                    outline="white",
                    width=1,
                )

            legend_entries.append(
                (str(class_name), color, 4)
            )

        # ---------------------------------------------------------
        # Overall mAP curve
        # ---------------------------------------------------------
        overall_map_values = []

        for threshold in thresholds:
            valid_values = []

            for class_id in per_class:
                ap_value = per_class[class_id]["ap_by_iou"].get(
                    threshold,
                    float("nan"),
                )

                if not np.isnan(ap_value):
                    valid_values.append(float(ap_value))

            if valid_values:
                overall_map_values.append(
                    (
                        threshold,
                        float(np.mean(valid_values)),
                    )
                )

        overall_points = [
            (
                x_coordinate(threshold),
                y_coordinate(map_value),
            )
            for threshold, map_value in overall_map_values
        ]

        if len(overall_points) >= 2:
            draw.line(
                overall_points,
                fill=(20, 20, 20),
                width=7,
            )

        for x, y in overall_points:
            radius = 8

            draw.ellipse(
                [
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                ],
                fill=(20, 20, 20),
                outline="white",
                width=2,
            )

        legend_entries.append(
            ("Overall mAP", (20, 20, 20), 7)
        )

        # ---------------------------------------------------------
        # Legend
        # ---------------------------------------------------------
        legend_x = plot_right + 40
        legend_y = plot_top + 10

        draw.text(
            (legend_x, legend_y),
            "Class",
            fill="black",
            font=legend_title_font,
        )

        legend_y += 48

        for label, color, line_width in legend_entries:
            line_y = legend_y + 10

            draw.line(
                [
                    (legend_x, line_y),
                    (legend_x + 55, line_y),
                ],
                fill=color,
                width=line_width,
            )

            draw.ellipse(
                [
                    legend_x + 23,
                    line_y - 5,
                    legend_x + 33,
                    line_y + 5,
                ],
                fill=color,
            )

            draw.text(
                (legend_x + 72, legend_y),
                label,
                fill="black",
                font=legend_font,
            )

            legend_y += 42

        # ---------------------------------------------------------
        # Explanatory note
        # ---------------------------------------------------------
        note = (
            "Confidence-ranked\n"
            "predictions are used\n"
            "for AP calculation."
        )

        draw.multiline_text(
            (legend_x, legend_y + 25),
            note,
            fill=(80, 80, 80),
            font=tick_font,
            spacing=5,
        )

        # Ensure the parent output directory exists.
        output_directory = os.path.dirname(output_path)

        if output_directory:
            os.makedirs(
                output_directory,
                exist_ok=True,
            )

        image.save(
            output_path,
            format="PNG",
        )

        logger.info(
            "AP-by-IoU graph saved to: %s",
            output_path,
        )

    @staticmethod
    def _save_confusion_plot(
        matrix,
        row_labels,
        column_labels,
        output_path,
        title,
        value_format,
    ):
        """
        Save a confusion-matrix heatmap using Pillow.

        Pillow is intentionally used instead of Matplotlib to avoid loading a
        second OpenMP runtime alongside CPU PyTorch on Windows.
        """
        rows, columns = matrix.shape
        cell_size = 105
        left_margin = 230
        top_margin = 215
        right_margin = 60
        bottom_margin = 90
        image_width = left_margin + columns * cell_size + right_margin
        image_height = top_margin + rows * cell_size + bottom_margin

        image = Image.new("RGB", (image_width, image_height), "white")
        draw = ImageDraw.Draw(image)

        def load_font(size, bold=False):
            font_candidates = [
                "arialbd.ttf" if bold else "arial.ttf",
                "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
            ]
            for candidate in font_candidates:
                try:
                    return ImageFont.truetype(candidate, size=size)
                except OSError:
                    continue
            return ImageFont.load_default()

        title_font = load_font(25, bold=True)
        axis_font = load_font(19, bold=True)
        label_font = load_font(16)
        value_font = load_font(17, bold=True)

        title_box = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_box[2] - title_box[0]
        draw.text(
            ((image_width - title_width) / 2, 20),
            title,
            fill="black",
            font=title_font,
        )

        predicted_label = "Predicted label"
        predicted_box = draw.textbbox((0, 0), predicted_label, font=axis_font)
        predicted_width = predicted_box[2] - predicted_box[0]
        draw.text(
            (
                left_margin + (columns * cell_size - predicted_width) / 2,
                62,
            ),
            predicted_label,
            fill="black",
            font=axis_font,
        )

        true_label = "True label"
        true_label_image = Image.new("RGBA", (180, 40), (255, 255, 255, 0))
        true_draw = ImageDraw.Draw(true_label_image)
        true_draw.text((0, 5), true_label, fill="black", font=axis_font)
        true_label_image = true_label_image.rotate(90, expand=True)
        image.paste(
            true_label_image,
            (
                15,
                top_margin + (rows * cell_size - true_label_image.height) // 2,
            ),
            true_label_image,
        )

        # Draw column labels vertically so long labels remain readable.
        for column_index, label in enumerate(column_labels):
            label_box = draw.textbbox((0, 0), label, font=label_font)
            label_width = max(1, label_box[2] - label_box[0] + 12)
            label_image = Image.new(
                "RGBA",
                (label_width, 30),
                (255, 255, 255, 0),
            )
            label_draw = ImageDraw.Draw(label_image)
            label_draw.text((6, 4), label, fill="black", font=label_font)
            label_image = label_image.rotate(90, expand=True)
            x_position = (
                left_margin
                + column_index * cell_size
                + (cell_size - label_image.width) // 2
            )
            y_position = top_margin - label_image.height - 8
            image.paste(
                label_image,
                (x_position, y_position),
                label_image,
            )

        maximum = float(np.max(matrix)) if matrix.size else 0.0
        for row_index in range(rows):
            row_label = row_labels[row_index]
            label_box = draw.textbbox((0, 0), row_label, font=label_font)
            label_width = label_box[2] - label_box[0]
            label_height = label_box[3] - label_box[1]
            draw.text(
                (
                    left_margin - label_width - 12,
                    top_margin
                    + row_index * cell_size
                    + (cell_size - label_height) / 2,
                ),
                row_label,
                fill="black",
                font=label_font,
            )

            for column_index in range(columns):
                value = float(matrix[row_index, column_index])
                ratio = value / maximum if maximum > 0 else 0.0
                # A restrained approximation of the Matplotlib "Blues" map.
                red = int(247 - 220 * ratio)
                green = int(251 - 145 * ratio)
                blue = int(255 - 55 * ratio)
                fill_color = (
                    max(0, red),
                    max(0, green),
                    max(0, blue),
                )

                x1 = left_margin + column_index * cell_size
                y1 = top_margin + row_index * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                draw.rectangle(
                    (x1, y1, x2, y2),
                    fill=fill_color,
                    outline=(185, 195, 205),
                    width=1,
                )

                value_text = (
                    str(int(value))
                    if value_format == "d"
                    else f"{value:.2f}"
                )
                value_box = draw.textbbox(
                    (0, 0),
                    value_text,
                    font=value_font,
                )
                value_width = value_box[2] - value_box[0]
                value_height = value_box[3] - value_box[1]
                draw.text(
                    (
                        x1 + (cell_size - value_width) / 2,
                        y1 + (cell_size - value_height) / 2,
                    ),
                    value_text,
                    fill="white" if ratio > 0.52 else "black",
                    font=value_font,
                )

        image.save(output_path, format="PNG")

# ================= RUN =================
if __name__ == "__main__":
    log_system_info()
    root = tk.Tk()
    app = ADaCITraApp(root)
    root.mainloop()
