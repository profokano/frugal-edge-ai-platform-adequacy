#!/usr/bin/env python3
"""
edgeai_onnx_benchmark.py

Portable ONNX Runtime benchmark for YOLOv8 object-detection models on SBCs.

Outputs:
  - per_inference.csv
  - system_monitor.csv
  - summary.json
  - predictions_yolo/*.txt

Example:
  python edgeai_onnx_benchmark.py \
    --model ./best8n.onnx \
    --images ./DeepPCB_yolo/images/test \
    --labels ./DeepPCB_yolo/labels/test \
    --output ./results_pi500_yolov8n \
    --imgsz 640 \
    --warmup 30 \
    --repeats 3 \
    --monitor-interval 1.0
"""

import argparse
import csv
import json
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import onnxruntime as ort
import psutil

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def run_cmd(cmd: List[str], timeout: float = 2.0) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=timeout).decode().strip()
    except Exception:
        return ""


def get_platform_info() -> Dict:
    return {
        "hostname": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.replace("\n", " "),
        "onnxruntime": ort.__version__,
        "opencv": cv2.__version__,
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
    }


def get_temperature_c() -> Optional[float]:
    values = []
    try:
        temps = psutil.sensors_temperatures(fahrenheit=False)
        for entries in temps.values():
            for e in entries:
                if e.current is not None and 0 < e.current < 130:
                    values.append(float(e.current))
    except Exception:
        pass
    try:
        for p in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
            raw = p.read_text().strip()
            if raw:
                v = float(raw)
                if v > 1000:
                    v = v / 1000.0
                if 0 < v < 130:
                    values.append(v)
    except Exception:
        pass
    out = run_cmd(["vcgencmd", "measure_temp"])
    if "temp=" in out:
        try:
            values.append(float(out.split("temp=")[1].split("'")[0]))
        except Exception:
            pass
    return max(values) if values else None


def get_rpi_throttled_raw() -> str:
    return run_cmd(["vcgencmd", "get_throttled"])


def parse_power_file(power_file: Optional[str]) -> Optional[float]:
    if not power_file:
        return None
    try:
        txt = Path(power_file).read_text().strip().splitlines()[-1]
        val = txt.split(",")[-1].strip()
        return float(val)
    except Exception:
        return None


class SystemMonitor:
    def __init__(self, output_csv: Path, interval: float = 1.0,
                 power_file: Optional[str] = None, fixed_power_w: Optional[float] = None):
        self.output_csv = output_csv
        self.interval = interval
        self.power_file = power_file
        self.fixed_power_w = fixed_power_w
        self.stop_event = threading.Event()
        self.thread = None
        self.rows = []

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        psutil.cpu_percent(interval=None)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        self._write_csv()

    def _run(self):
        while not self.stop_event.is_set():
            vm = psutil.virtual_memory()
            temp = get_temperature_c()
            throttled = get_rpi_throttled_raw()
            power = self.fixed_power_w if self.fixed_power_w is not None else parse_power_file(self.power_file)
            self.rows.append({
                "timestamp": time.time(),
                "cpu_percent": psutil.cpu_percent(interval=None),
                "ram_used_mb": vm.used / (1024 * 1024),
                "ram_percent": vm.percent,
                "temperature_c": temp,
                "rpi_throttled_raw": throttled,
                "power_w": power,
            })
            time.sleep(self.interval)

    def _write_csv(self):
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = ["timestamp", "cpu_percent", "ram_used_mb", "ram_percent",
                  "temperature_c", "rpi_throttled_raw", "power_w"]
        with self.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)


def letterbox(img: np.ndarray, new_shape: int = 640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = new_shape - new_unpad[0]
    dh = new_shape - new_unpad[1]
    dw /= 2
    dh /= 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def preprocess(img_path: Path, imgsz: int):
    img0 = cv2.imread(str(img_path))
    if img0 is None:
        raise ValueError(f"Could not read image: {img_path}")
    img, ratio, pad = letterbox(img0, imgsz)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    return x, img0.shape[:2], ratio, pad


def xywh_to_xyxy(x):
    y = np.empty_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def scale_boxes(boxes, orig_shape, ratio, pad):
    dw, dh = pad
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, :4] /= ratio
    h, w = orig_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes


def nms_classwise(boxes, scores, classes, iou_thres):
    keep = []
    for c in np.unique(classes):
        idxs = np.where(classes == c)[0]
        b = boxes[idxs]
        s = scores[idxs]
        xywh = []
        for box in b:
            x1, y1, x2, y2 = box
            xywh.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
        selected = cv2.dnn.NMSBoxes(xywh, s.astype(float).tolist(), score_threshold=0.0, nms_threshold=float(iou_thres))
        if len(selected) > 0:
            selected = np.array(selected).reshape(-1)
            keep.extend(idxs[selected].tolist())
    return keep


def postprocess_yolov8(outputs, orig_shape, ratio, pad, conf_thres=0.25, iou_thres=0.45):
    pred = outputs[0]
    pred = np.squeeze(pred)
    if pred.ndim == 2 and pred.shape[0] < pred.shape[1]:
        pred = pred.T
    if pred.ndim == 3:
        pred = pred[0]
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T
    if pred.ndim != 2:
        return np.empty((0, 6), dtype=np.float32)

    if pred.shape[1] > 6:
        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:]
        classes = np.argmax(class_scores, axis=1).astype(np.int32)
        scores = class_scores[np.arange(class_scores.shape[0]), classes]
        mask = scores >= conf_thres
        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        classes = classes[mask]
        if len(scores) == 0:
            return np.empty((0, 6), dtype=np.float32)
        boxes = xywh_to_xyxy(boxes_xywh)
        boxes = scale_boxes(boxes, orig_shape, ratio, pad)
    elif pred.shape[1] == 6:
        boxes = pred[:, :4]
        scores = pred[:, 4]
        classes = pred[:, 5].astype(np.int32)
        mask = scores >= conf_thres
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]
        if len(scores) == 0:
            return np.empty((0, 6), dtype=np.float32)
        boxes = scale_boxes(boxes, orig_shape, ratio, pad)
    else:
        return np.empty((0, 6), dtype=np.float32)

    keep = nms_classwise(boxes, scores, classes, iou_thres)
    if not keep:
        return np.empty((0, 6), dtype=np.float32)
    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]
    out = np.column_stack([boxes, scores, classes]).astype(np.float32)
    out = out[np.argsort(-out[:, 4])]
    return out


def save_predictions_yolo(pred_dir: Path, img_path: Path, dets: np.ndarray, orig_shape):
    pred_dir.mkdir(parents=True, exist_ok=True)
    h, w = orig_shape
    out_file = pred_dir / f"{img_path.stem}.txt"
    with out_file.open("w") as f:
        for x1, y1, x2, y2, conf, cls in dets:
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.6f}\n")


def load_yolo_labels(label_path: Path, img_shape) -> np.ndarray:
    h, w = img_shape
    if not label_path.exists():
        return np.empty((0, 5), dtype=np.float32)
    rows = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls, cx, cy, bw, bh = map(float, parts[:5])
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        rows.append([cls, x1, y1, x2, y2])
    return np.array(rows, dtype=np.float32) if rows else np.empty((0, 5), dtype=np.float32)


def box_iou(box, boxes):
    if boxes.size == 0:
        return np.array([])
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area2 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-9)


def compute_prf1_at_iou(preds_by_image: Dict[str, np.ndarray], gts_by_image: Dict[str, np.ndarray], iou_thr: float = 0.5) -> Dict:
    tp = fp = fn = 0
    for img_id, gt in gts_by_image.items():
        pred = preds_by_image.get(img_id, np.empty((0, 6), dtype=np.float32))
        matched = set()
        for p in pred:
            p_box = p[:4]
            p_cls = int(p[5])
            candidates = np.where(gt[:, 0].astype(int) == p_cls)[0] if len(gt) else np.array([])
            best_iou, best_j = 0.0, None
            for j in candidates:
                if j in matched:
                    continue
                iou = box_iou(p_box, gt[j:j+1, 1:5])[0]
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j is not None and best_iou >= iou_thr:
                tp += 1
                matched.add(best_j)
            else:
                fp += 1
        fn += max(0, len(gt) - len(matched))
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {"precision_at_0.5": precision, "recall_at_0.5": recall, "f1_at_0.5": f1, "tp": tp, "fp": fp, "fn": fn}


def average_precision(recalls, precisions):
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_map(preds_by_image: Dict[str, np.ndarray], gts_by_image: Dict[str, np.ndarray], iou_thresholds: List[float]) -> Dict:
    classes = set()
    for gt in gts_by_image.values():
        if len(gt):
            classes.update(gt[:, 0].astype(int).tolist())
    for pred in preds_by_image.values():
        if len(pred):
            classes.update(pred[:, 5].astype(int).tolist())

    aps_by_thr = {}
    for iou_thr in iou_thresholds:
        aps = []
        for cls in sorted(classes):
            n_gt = sum(int(np.sum(gt[:, 0].astype(int) == cls)) for gt in gts_by_image.values())
            if n_gt == 0:
                continue
            pred_records = []
            for img_id, pred in preds_by_image.items():
                for p in pred:
                    if int(p[5]) == cls:
                        pred_records.append((img_id, float(p[4]), p[:4].copy()))
            pred_records.sort(key=lambda x: x[1], reverse=True)
            tp = np.zeros(len(pred_records))
            fp = np.zeros(len(pred_records))
            matched = {img_id: set() for img_id in gts_by_image.keys()}
            for i, (img_id, score, p_box) in enumerate(pred_records):
                gt = gts_by_image.get(img_id, np.empty((0, 5), dtype=np.float32))
                gt_cls_idx = np.where(gt[:, 0].astype(int) == cls)[0] if len(gt) else np.array([])
                if len(gt_cls_idx) == 0:
                    fp[i] = 1
                    continue
                gt_boxes = gt[gt_cls_idx, 1:5]
                ious = box_iou(p_box, gt_boxes)
                best_local = int(np.argmax(ious)) if len(ious) else -1
                best_iou = float(ious[best_local]) if len(ious) else 0.0
                best_global = int(gt_cls_idx[best_local]) if best_local >= 0 else None
                if best_iou >= iou_thr and best_global not in matched[img_id]:
                    tp[i] = 1
                    matched[img_id].add(best_global)
                else:
                    fp[i] = 1
            if len(pred_records) == 0:
                aps.append(0.0)
                continue
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recalls = tp_cum / (n_gt + 1e-9)
            precisions = tp_cum / (tp_cum + fp_cum + 1e-9)
            aps.append(average_precision(recalls, precisions))
        aps_by_thr[round(iou_thr, 2)] = float(np.mean(aps)) if aps else None
    map50 = aps_by_thr.get(0.5)
    valid_aps = [v for v in aps_by_thr.values() if v is not None]
    map5095 = float(np.mean(valid_aps)) if valid_aps else None
    return {"mAP@0.5": map50, "mAP@0.5:0.95": map5095, "AP_by_IoU": {str(k): v for k, v in aps_by_thr.items()}}


def percentile(values, p):
    return float(np.percentile(np.array(values), p)) if values else None


def list_images(images_dir: Path) -> List[Path]:
    return sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--images", required=True, help="Directory with test images")
    parser.add_argument("--labels", default=None, help="Optional YOLO labels directory for accuracy metrics")
    parser.add_argument("--output", required=True, help="Output directory for logs and results")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--monitor-interval", type=float, default=1.0)
    parser.add_argument("--providers", default="CPUExecutionProvider", help="Comma-separated ONNX Runtime providers")
    parser.add_argument("--power-file", default=None, help="Optional file whose last value is current board power in watts")
    parser.add_argument("--fixed-power-w", type=float, default=None, help="Optional fixed average power in watts")
    parser.add_argument("--save-predictions", action="store_true", default=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    images_dir = Path(args.images)
    labels_dir = Path(args.labels) if args.labels else None
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(images_dir)
    if not images:
        raise RuntimeError(f"No images found in {images_dir}")

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=providers)
    input_name = session.get_inputs()[0].name

    platform_info = get_platform_info()
    platform_info.update({
        "onnx_providers_requested": providers,
        "onnx_providers_active": session.get_providers(),
        "model": str(model_path),
        "images_dir": str(images_dir),
        "num_images": len(images),
        "imgsz": args.imgsz,
    })

    x, _, _, _ = preprocess(images[0], args.imgsz)
    for _ in range(args.warmup):
        _ = session.run(None, {input_name: x})

    monitor = SystemMonitor(out_dir / "system_monitor.csv", args.monitor_interval, args.power_file, args.fixed_power_w)
    monitor.start()

    per_rows = []
    preds_by_image = {}
    gts_by_image = {}
    failures = 0
    latencies_ms = []
    pred_dir = out_dir / "predictions_yolo"
    wall_start = time.perf_counter()

    try:
        for rep in range(args.repeats):
            for img_path in images:
                try:
                    x, orig_shape, ratio, pad = preprocess(img_path, args.imgsz)
                    t0 = time.perf_counter()
                    outputs = session.run(None, {input_name: x})
                    t1 = time.perf_counter()
                    latency_ms = (t1 - t0) * 1000.0
                    dets = postprocess_yolov8(outputs, orig_shape, ratio, pad, args.conf, args.iou)
                    latencies_ms.append(latency_ms)
                    img_id = img_path.stem
                    if rep == 0:
                        preds_by_image[img_id] = dets.copy()
                        if labels_dir:
                            gts_by_image[img_id] = load_yolo_labels(labels_dir / f"{img_path.stem}.txt", orig_shape)
                        if args.save_predictions:
                            save_predictions_yolo(pred_dir, img_path, dets, orig_shape)
                    per_rows.append({"timestamp": time.time(), "repeat": rep + 1, "image": str(img_path),
                                     "latency_ms": latency_ms, "detections": int(len(dets)), "status": "ok"})
                except Exception as e:
                    failures += 1
                    per_rows.append({"timestamp": time.time(), "repeat": rep + 1, "image": str(img_path),
                                     "latency_ms": "", "detections": "", "status": f"error: {e}"})
    finally:
        monitor.stop()

    wall_elapsed = time.perf_counter() - wall_start
    ok_count = len(latencies_ms)

    with (out_dir / "per_inference.csv").open("w", newline="") as f:
        fields = ["timestamp", "repeat", "image", "latency_ms", "detections", "status"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_rows)

    temps = [r["temperature_c"] for r in monitor.rows if r.get("temperature_c") is not None]
    cpus = [r["cpu_percent"] for r in monitor.rows if r.get("cpu_percent") is not None]
    ram_mb = [r["ram_used_mb"] for r in monitor.rows if r.get("ram_used_mb") is not None]
    powers = [r["power_w"] for r in monitor.rows if r.get("power_w") is not None]
    throttled_nonempty = [r["rpi_throttled_raw"] for r in monitor.rows if r.get("rpi_throttled_raw")]
    avg_power = float(np.mean(powers)) if powers else None
    fps_wall = ok_count / wall_elapsed if wall_elapsed > 0 else None

    summary = {
        "platform": platform_info,
        "benchmark": {
            "total_successful_inferences": ok_count,
            "runtime_failures": failures,
            "wall_time_s": wall_elapsed,
            "fps_wall": fps_wall,
            "mean_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else None,
            "median_latency_ms": percentile(latencies_ms, 50),
            "p95_latency_ms": percentile(latencies_ms, 95),
            "p99_latency_ms": percentile(latencies_ms, 99),
            "std_latency_ms": float(np.std(latencies_ms)) if latencies_ms else None,
            "min_latency_ms": float(np.min(latencies_ms)) if latencies_ms else None,
            "max_latency_ms": float(np.max(latencies_ms)) if latencies_ms else None,
        },
        "system": {
            "mean_cpu_percent": float(np.mean(cpus)) if cpus else None,
            "max_cpu_percent": float(np.max(cpus)) if cpus else None,
            "mean_ram_used_mb": float(np.mean(ram_mb)) if ram_mb else None,
            "peak_ram_used_mb": float(np.max(ram_mb)) if ram_mb else None,
            "mean_temperature_c": float(np.mean(temps)) if temps else None,
            "peak_temperature_c": float(np.max(temps)) if temps else None,
            "rpi_throttling_samples": throttled_nonempty,
            "mean_power_w": avg_power,
            "energy_per_inference_j": (avg_power / fps_wall) if (avg_power and fps_wall and fps_wall > 0) else None,
            "fps_per_watt": (fps_wall / avg_power) if (avg_power and avg_power > 0 and fps_wall) else None,
        }
    }

    if labels_dir:
        prf1 = compute_prf1_at_iou(preds_by_image, gts_by_image, iou_thr=0.5)
        map_metrics = compute_map(preds_by_image, gts_by_image,
                                  [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
        summary["accuracy"] = {**prf1, **map_metrics}

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
