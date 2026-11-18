import os
# Configure MKL before importing NumPy/PyTorch: Conda's Intel runtime can
# otherwise collide with PyTorch's OpenMP runtime when plotting validation.
if os.name == "nt":
    os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
# Report plots are saved to files; they must never create Tk windows in a worker.
os.environ["MPLBACKEND"] = "Agg"
import queue
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
import sys

# ================= CONFIG =================
from utils.logger import logger
from core.ultralytics_evaluator import UltralyticsEvaluator
from core.inference_timing import run_experiment, summarize, completion_message

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
APP_VERSION = "0.4.3"
INFERENCE_IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25
WARMUP_RUNS = 3
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def synchronize_cuda():
    """Synchronize CUDA before timing when a CUDA device is active."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
# ================= UTIL =================
def get_os_info():
    """Return detailed OS information for Windows and Linux."""
    system_name = platform.system()

    if system_name == "Windows":
        try:
            import winreg

            registry_path = (
                r"SOFTWARE\Microsoft\Windows NT\CurrentVersion"
            )

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                registry_path,
            ) as registry_key:
                product_name = winreg.QueryValueEx(
                    registry_key,
                    "ProductName",
                )[0]

                display_version = winreg.QueryValueEx(
                    registry_key,
                    "DisplayVersion",
                )[0]

            build_number = sys.getwindowsversion().build

            # Windows 11 may still be identified as Windows 10
            # in legacy registry fields.
            if build_number >= 22000:
                product_name = product_name.replace(
                    "Windows 10",
                    "Windows 11",
                )

            architecture = platform.machine()

            return (
                f"{product_name} {display_version}, "
                f"build {build_number}, {architecture}"
            )

        except (OSError, FileNotFoundError):
            build_number = sys.getwindowsversion().build
            windows_name = (
                "Windows 11"
                if build_number >= 22000
                else "Windows 10"
            )
            return (
                f"{windows_name}, build {build_number}, "
                f"{platform.machine()}"
            )

    if system_name == "Linux":
        return (
            f"Linux {platform.release()}, "
            f"{platform.machine()}"
        )

    return platform.platform()

def log_system_info():
    logger.info("=" * 60)
    logger.info("%s v%s", APP_NAME, APP_VERSION)
    logger.info("OS      : %s", get_os_info())
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
        self.yaml_path = None

        self.model = None
        self.image_dir = None
        self.gt_dir = None
        self.output_dir = None
        self.input_source = None

        self.images = []
        self.idx = 0
        self.pred_results = {}
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

        self.yaml_path = os.path.abspath(yaml_path)
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
        # Removed self.progress.pack(before=self.title_label, pady=5) from here
        self.progress_label = tk.Label(self.root, text="")
        tk.Button(f1,text="USB Camera",width=18,command=self.load_USBcam).pack(side=tk.LEFT, padx=5)
        tk.Button(f2, text="RUN INFERENCE", width=22, bg="green",fg="white", command=self.run_inference).pack(side=tk.LEFT, padx=10)
        self.report_button = tk.Button(f2, text="GENERATE REPORT", width=22, bg="blue",fg="white", command=self.generate_report)
        self.report_button.pack(side=tk.LEFT, padx=10)
        tk.Button(f2, text="RUN LATENCY", width=16,
                  command=self.run_timing_experiment).pack(side=tk.LEFT, padx=5)
        self.report_status = tk.Label(self.root, text="")
        self.report_status.pack()
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
        # Use the same internal inference timing source as RUN LATENCY.
        results = self.model(
            frame,
            imgsz=INFERENCE_IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )[0]

        preds = []
        for b in results.boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            box = b.xywhn[0].tolist()
            preds.append((cls, conf, box))
        annotated = draw_pred(frame.copy(), preds, self.class_names)
        inference_ms = results.speed.get("inference")
        self.current_fps = (
            1000.0 / inference_ms
            if inference_ms is not None and np.isfinite(inference_ms) and inference_ms > 0
            else None
        )
        # ================= HUD =================
        # background
        cv2.rectangle(annotated,(10, 10),(310, 70),(40, 40, 40),-1)
        # border
        cv2.rectangle(annotated,(10, 10),(310, 70),(255,255,255),1)
        # title
        cv2.putText(annotated,"VIDEO INFERENCE",(20, 32),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)
        # fps
        fps_text = f"{self.current_fps:.2f}" if self.current_fps is not None else "N/A"
        cv2.putText(annotated,f"Inference FPS : {fps_text}",(20, 58),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)

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
            self.model.to(DEVICE)
            logger.info("Inference device: %s", self.model.device)
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
            for sub in ["images", "video", "csv", "evaluation"]:
                os.makedirs(os.path.join(self.output_dir, sub), exist_ok=True)
            self.show_popup("Output Path Set")

    # ---------- CORE ----------
    def run_inference(self):
        logger.info("RUN INFERENCE clicked | images=%s | image_dir=%s | output=%s",
                    len(self.images), self.image_dir, self.output_dir)
        if getattr(self, "_inference_running", False):
            self.show_popup("Image inference is still running")
            return
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

        self.progress.pack(before=self.title_label, pady=5)
        self.progress_label.pack(before=self.title_label)
        self.progress["maximum"] = len(self.images)
        self.progress["value"] = 0
        self.progress_label.config(text="Starting image inference; warming up model...")
        self._inference_running = True
        try:
            threading.Thread(
                target=self.run_inference_worker,
                daemon=True
            ).start()
        except Exception as error:
            logger.exception("Could not start image inference worker.")
            self._finish_inference(str(error))

    def _update_inference_progress(self, value, total):
        self.progress["value"] = value
        self.progress_label.config(text=f"Processing {value}/{total}")

    def _finish_inference(self, error_message=None):
        self._inference_running = False
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
        logger.info("Image inference worker started.")
        self.pred_results.clear()
        self.gt_all.clear()
        total = len(self.images)
        csv_dir = os.path.join(self.output_dir, "csv")
        image_output_dir = os.path.join(self.output_dir, "images")

        prediction_rows = [["image", "class_id", "confidence", "x", "y", "w", "h"]]
        measured_times = []
        timing_rows = [["image", "pipeline_ms", "preprocess_ms", "inference_ms", "postprocess_ms"]]

        try:
            os.makedirs(csv_dir, exist_ok=True)
            os.makedirs(image_output_dir, exist_ok=True)
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
                    device=DEVICE,
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
                    device=DEVICE,
                    verbose=False,
                )[0]
                synchronize_cuda()
                measured_times.append(time.perf_counter() - start_time)
                timing_rows.append([name, measured_times[-1] * 1000.0,
                                    result.speed.get("preprocess"), result.speed.get("inference"),
                                    result.speed.get("postprocess")])

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
            timing_stats = summarize([value * 1000.0 for value in measured_times])
            with open(os.path.join(csv_dir, "raw_inference_times.csv"), "w",
                      newline="", encoding="utf-8") as file:
                csv.writer(file).writerows(timing_rows)
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
                writer.writerow(["Warm-up Runs", WARMUP_RUNS])
                writer.writerow(["Model Pipeline Time (s)", round(self.total_inference_time, 4)])
                writer.writerow(["FPS", round(self.fps, 2)])
                writer.writerow(["Latency (ms/image)", round(latency_ms, 2)])
                writer.writerow(["Pipeline Mean (ms)", timing_stats["mean_ms"]])
                writer.writerow(["Pipeline Median (ms)", timing_stats["median_ms"]])
                writer.writerow(["Pipeline P95 (ms, linear)", timing_stats["p95_ms"]])
                writer.writerow([
                    "Timing Scope",
                    "Ultralytics preprocessing + inference + post-processing; "
                    "excludes disk read, drawing, GUI update, and file write",
                ])

            logger.info("Image inference completed | measured=%s | output=%s",
                        measured_count, self.output_dir)
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
        self.progress.pack(before=self.title_label, pady=5)
        self.progress_label.pack(before=self.title_label)
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

    def run_timing_experiment(self):
        if getattr(self, "_timing_running", False):
            return
        if (getattr(self, "_report_running", False) or self.video_running
                or self.progress.winfo_ismapped()):
            messagebox.showwarning("Latency", "Wait for active inference/export/report or stop video first.")
            return
        if not self.model or not self.images or not self.image_dir or not self.output_dir:
            messagebox.showwarning("Latency", "Load a model, images, and output folder first.")
            return
        settings = dict(model_path=self.model.ckpt_path, device=str(self.model.device),
                        image_dir=self.image_dir, images=tuple(self.images),
                        output_dir=self.output_dir, imgsz=INFERENCE_IMAGE_SIZE,
                        conf=CONFIDENCE_THRESHOLD, warmups=WARMUP_RUNS, repeats=10)
        self._timing_queue = queue.Queue()
        self._timing_running = True
        self._timing_buttons = []

        def disable_buttons(widget):
            for child in widget.winfo_children():
                if isinstance(child, tk.Button):
                    self._timing_buttons.append((child, child.cget("state")))
                    child.config(state="disabled")
                disable_buttons(child)

        disable_buttons(self.root)
        self.report_status.config(text="Latency test: warming up (10 runs)...")
        try:
            threading.Thread(target=self._timing_worker,
                             args=(settings, self._timing_queue), daemon=True).start()
        except Exception as error:
            self._timing_queue.put(("error", str(error)))
        self.root.after(100, self._poll_timing)

    @staticmethod
    def _timing_worker(settings, events):
        try:
            def progress(run, index, total):
                events.put(("progress", f"Latency run {run}/10: image {index}/{total}"))
            output = run_experiment(**settings, progress=progress)
            events.put(("done", completion_message(output)))
        except Exception as error:
            logger.exception("Timing experiment failed.")
            events.put(("error", str(error)))

    def _poll_timing(self):
        try:
            while True:
                kind, value = self._timing_queue.get_nowait()
                if kind == "progress":
                    self.report_status.config(text=value)
                    continue
                self._timing_running = False
                for button, state in self._timing_buttons:
                    button.config(state=state)
                self.report_status.config(text="Latency completed: 10/10 runs." if kind == "done" else "Latency failed.")
                if kind == "done":
                    messagebox.showinfo("Latency Results", value)
                else:
                    messagebox.showerror("Latency Error", value)
                return
        except queue.Empty:
            self.root.after(100, self._poll_timing)

    def _create_evaluator(self):
        return UltralyticsEvaluator(
            model_path=getattr(self.model, "ckpt_path", None),
            yaml_path=self.yaml_path,
            output_dir=self.output_dir,
            app_version=APP_VERSION,
        )

    def run_ultralytics_evaluation(self):
        """Compatibility entry point; evaluation lives in core."""
        metrics = self._create_evaluator().evaluate()
        self.ultralytics_metrics = metrics
        return metrics

    def generate_report(self):
        if getattr(self, "_report_running", False):
            return
        # Freeze report inputs so changing a GUI selection cannot mix datasets.
        evaluation = self._create_evaluator()
        self._report_queue = queue.Queue()
        self._report_running = True
        self.report_button.config(state="disabled")
        self.report_status.config(text="Evaluating YAML test split; please wait...")
        try:
            threading.Thread(target=self._report_worker,
                             args=(evaluation, self._report_queue), daemon=True).start()
        except Exception as error:
            self._report_queue.put((None, str(error)))
        self.root.after(100, self._poll_report)

    @staticmethod
    def _report_worker(evaluation, results):
        # No Tk calls in this thread, including root.after().
        try:
            results.put((evaluation.evaluate(), None))
        except Exception as error:
            logger.exception("Ultralytics evaluation failed.")
            results.put((None, str(error)))

    def _poll_report(self):
        try:
            metrics, error = self._report_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_report)
            return
        self._report_running = False
        self.report_button.config(state="normal")
        self.report_status.config(text="Evaluation failed." if error else "Evaluation completed.")
        if error:
            messagebox.showerror("Evaluation Error", error)
        else:
            self.ultralytics_metrics = metrics
            messagebox.showinfo(
                "Ultralytics Evaluation",
                f"Evaluation completed for the YAML test split.\nReports: {metrics.save_dir}",
            )

    def save_confusion_matrix(self):
        """Retain the report entry point; all metrics and CM come from Ultralytics."""
        return self.run_ultralytics_evaluation()

# ================= RUN =================
if __name__ == "__main__":
    log_system_info()
    root = tk.Tk()
    app = ADaCITraApp(root)
    root.mainloop()


