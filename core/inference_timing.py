"""Raw model-call timing and repeated experiments; no accuracy evaluation."""
import csv
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO


def summarize(values):
    """Milliseconds; P95 uses NumPy's linear percentile interpolation."""
    values = np.asarray(values, dtype=float)
    if not values.size:
        raise ValueError("No successful inference measurements.")
    return dict(count=int(values.size), mean_ms=float(values.mean()),
                median_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values, 95)),
                fps=float(1000.0 / values.mean()) if values.mean() > 0 else 0.0)


def run_experiment(model_path, device, image_dir, images, output_dir,
                   imgsz=640, conf=0.25, warmups=3, repeats=10, progress=None):
    """Ten sequential warmed dataset passes with durable raw rows and run summaries."""
    if not images:
        raise ValueError("Please load images first.")
    folder_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    destination = Path(output_dir) / 'timing' / folder_name
    duplicate = 1
    while True:
        try:
            destination.mkdir(parents=True)
            break
        except FileExistsError:
            duplicate += 1
            destination = Path(output_dir) / 'timing' / f'{folder_name}_{duplicate:02d}'
    metadata = dict(status='running', model=str(model_path), device=str(device),
                    image_dir=str(image_dir), images=list(images), repeats=repeats,
                    warmups_per_run=warmups, imgsz=imgsz, conf=conf,
                    ultralytics=ultralytics.__version__, torch=torch.__version__,
                    numpy=np.__version__, python=platform.python_version(),
                    platform=platform.platform(),
                    mkl_threading_layer=__import__('os').environ.get('MKL_THREADING_LAYER'),
                    scope='Pipeline wall time includes preprocessing/inference/postprocessing; excludes disk read, drawing, UI and CSV writes.',
                    internal_inference='Ultralytics result.speed[inference], separate from pipeline wall time.',
                    aggregation='Per-run and pooled image samples; P95 uses linear interpolation.',
                    primary_metric='Ultralytics inference latency (result.speed[inference])',
                    primary_summary='latency_summary.csv',
                    supporting_summary='pipeline_timing_summary.csv',
                    fps_definition='Equivalent inference FPS = 1000 / mean inference_ms; not application FPS.',
                    input_shape_analysis='Original image dimensions and actual processed inference tensor shape are recorded per sample.',
                    input_shape_capture='Observe predictor.preprocess output shape without modifying the tensor or running extra preprocessing/inference. Observer overhead is in preprocessing/pipeline, not the internal inference timer.',
                    error=None)
    meta_path = destination / 'metadata.json'

    def save_metadata():
        meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    save_metadata()
    try:
        model = YOLO(model_path)
        model.to(device)

        # Results exposes original image shape, but not the actual input tensor.
        # Observe the existing preprocessing call; return its exact tensor unchanged.
        tensor_shape = None
        observed_predictor = None

        def observe_preprocessing(predictor):
            nonlocal observed_predictor
            if observed_predictor is predictor:
                return
            original_preprocess = predictor.preprocess

            def capture_shape(*args, **kwargs):
                nonlocal tensor_shape
                tensor = original_preprocess(*args, **kwargs)
                tensor_shape = (int(tensor.shape[-2]), int(tensor.shape[-1]))
                return tensor

            predictor.preprocess = capture_shape
            observed_predictor = predictor

        model.add_callback('on_predict_start', observe_preprocessing)

        def sync():
            if model.device.type == 'cuda':
                torch.cuda.synchronize(model.device)

        first = cv2.imread(str(Path(image_dir) / images[0]))
        if first is None:
            raise ValueError(f'Cannot read image: {images[0]}')
        pooled = []
        shape_groups = {}
        with (destination / 'raw_inference_times.csv').open('w', newline='', encoding='utf-8') as raw_file, \
             (destination / 'latency_summary.csv').open('w', newline='', encoding='utf-8') as latency_file, \
             (destination / 'pipeline_timing_summary.csv').open('w', newline='', encoding='utf-8') as pipeline_file:
            raw = csv.writer(raw_file)
            latency = csv.writer(latency_file)
            pipeline = csv.writer(pipeline_file)
            raw.writerow(['run', 'image_index', 'image', 'original_width', 'original_height',
                          'aspect_ratio', 'tensor_height', 'tensor_width', 'tensor_pixels', 'pipeline_ms',
                          'preprocess_ms', 'inference_ms', 'postprocess_ms'])
            latency.writerow(['run', 'count', 'inference_mean_ms', 'inference_median_ms',
                              'inference_p95_ms', 'equivalent_inference_fps'])
            pipeline.writerow(['run', 'count', 'pipeline_mean_ms', 'pipeline_median_ms',
                               'pipeline_p95_ms', 'equivalent_pipeline_fps'])
            for run in range(1, repeats + 1):
                for _ in range(warmups):
                    model(first, imgsz=imgsz, conf=conf, verbose=False)
                sync()
                samples, internal = [], []
                for index, name in enumerate(images, 1):
                    frame = cv2.imread(str(Path(image_dir) / name))
                    if frame is None:
                        raise ValueError(f'Cannot read image in run {run}: {name}')
                    original_height, original_width = frame.shape[:2]
                    tensor_shape = None  # Never reuse a previous sample's shape.
                    sync()
                    start = time.perf_counter()
                    result = model(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
                    sync()
                    elapsed = (time.perf_counter() - start) * 1000.0
                    if tensor_shape is None:
                        raise RuntimeError('Actual processed tensor shape was not captured.')
                    tensor_height, tensor_width = tensor_shape
                    speed = result.speed
                    samples.append(elapsed)
                    internal.append(float(speed['inference']))
                    group = shape_groups.setdefault(
                        (original_width, original_height, tensor_height, tensor_width),
                        {'images': set(), 'inference_ms': []})
                    group['images'].add(name)
                    group['inference_ms'].append(float(speed['inference']))
                    raw.writerow([run, index, name, original_width, original_height,
                                  original_width / original_height, tensor_height, tensor_width,
                                  tensor_height * tensor_width, elapsed, speed['preprocess'],
                                  speed['inference'], speed['postprocess']])
                    raw_file.flush()
                    if progress:
                        progress(run, index, len(images))
                pooled.extend(zip(samples, internal))
                for writer, values in [(pipeline, samples), (latency, internal)]:
                    stats = summarize(values)
                    writer.writerow([run, *stats.values()])
                latency_file.flush()
                pipeline_file.flush()
            for column, writer in enumerate([pipeline, latency]):
                stats = summarize([row[column] for row in pooled])
                writer.writerow(['pooled', *stats.values()])
            metadata['inference_pooled'] = summarize([row[1] for row in pooled])
        with (destination / 'input_shape_latency_summary.csv').open('w', newline='', encoding='utf-8') as shape_file:
            writer = csv.writer(shape_file)
            writer.writerow(['original_resolution', 'aspect_ratio', 'tensor_shape', 'tensor_pixels',
                             'unique_images', 'measurement_count', 'inference_mean_ms',
                             'inference_median_ms', 'inference_p95_ms', 'equivalent_inference_fps'])
            for (width, height, tensor_height, tensor_width), group in sorted(shape_groups.items()):
                stats = summarize(group['inference_ms'])
                writer.writerow([f'{width}x{height}', width / height,
                                 f'{tensor_height}x{tensor_width}', tensor_height * tensor_width,
                                 len(group['images']), stats['count'], stats['mean_ms'],
                                 stats['median_ms'], stats['p95_ms'], stats['fps']])
        metadata['status'] = 'complete'
    except Exception as error:
        metadata.update(status='failed', error=str(error))
        raise RuntimeError(f'{error}\nPartial timing data: {destination}') from error
    finally:
        save_metadata()
    return destination


def completion_message(destination):
    """Display the same pooled measurements exported in the primary summary."""
    metadata = json.loads((Path(destination) / 'metadata.json').read_text(encoding='utf-8'))
    stats = metadata['inference_pooled']
    repeats = metadata['repeats']
    return (f"Latency test completed: {repeats}/{repeats} runs\n"
            f"Measured images: {stats['count']:,}\n\n"
            "Ultralytics inference latency\n"
            f"Mean: {stats['mean_ms']:.2f} ms\n"
            f"Median: {stats['median_ms']:.2f} ms\n"
            f"P95: {stats['p95_ms']:.2f} ms\n"
            f"Equivalent inference FPS: {stats['fps']:.2f}\n\n"
            f"Primary report: {Path(destination) / 'latency_summary.csv'}")
