#!/usr/bin/env python3
"""HailoRT 4.24.0 YOLOv8 DFL detector benchmark for KolektorSDD2.

Assumes two pre-NMS output vstreams: (1,3549,1) class and (1,3549,64) DFL.
Grid/stride order P3=52x52/8, P4=26x26/16, P5=13x13/32 has been
validated for DeepPCB; its applicability to KolektorSDD2 must be checked on annotated images.

Results: per_image.csv, predictions_{operational,ap}.jsonl, summary.json,
settings.json and per_class_ap.csv. No HEF or dataset is modified.

AP calculation is VOC/COCO-style 101-point interpolation across IoU thresholds
0.50:0.05:0.95, with greedy one-to-one assignment per class per image,
but is NOT the official COCO evaluator (no area ranges/crowd/maxDet semantics).
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from hailo_platform import (HEF, VDevice, HailoStreamInterface,
    ConfigureParams, InputVStreamParams, OutputVStreamParams,
    InferVStreams, FormatType)

NUM_CLASSES = 1
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
GRID_SPECS = ((52, 8), (26, 16), (13, 32))
IOU_THRESHOLDS = np.round(np.arange(.50, .951, .05), 2)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def anchors():
    ctr, st = [], []
    for grid, stride in GRID_SPECS:
        yy, xx = np.meshgrid(np.arange(grid), np.arange(grid), indexing='ij')
        ctr.append(np.stack((xx.ravel() + .5, yy.ravel() + .5), 1) * stride)
        st.append(np.full(grid * grid, stride))
    return np.concatenate(ctr).astype(np.float32), np.concatenate(st).astype(np.float32)

CENTERS, STRIDES = anchors()


def letterbox(bgr, size=416):
    h, w = bgr.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), (scale, left, top, w, h)


def output_tensors(outputs):
    arrays = [np.asarray(v, dtype=np.float32) for v in outputs.values()]
    cls = [v.reshape(-1, NUM_CLASSES) for v in arrays if v.size == 3549 * NUM_CLASSES]
    dfl = [v.reshape(-1, 64) for v in arrays if v.size == 3549 * 64]
    if len(cls) != 1 or len(dfl) != 1:
        raise ValueError('Unexpected HEF outputs: ' + str([(k, v.shape) for k, v in outputs.items()]))
    if not (np.isfinite(cls[0]).all() and np.isfinite(dfl[0]).all()):
        raise ValueError('Non-finite model output')
    return cls[0], dfl[0]


def decode(cls, dfl, transform):
    logits = dfl.reshape(-1, 4, 16)
    logits = logits - logits.max(axis=2, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=2, keepdims=True)
    distances = (probs * np.arange(16, dtype=np.float32)).sum(axis=2) * STRIDES[:, None]
    boxes = np.column_stack((
        CENTERS[:, 0] - distances[:, 0], CENTERS[:, 1] - distances[:, 1],
        CENTERS[:, 0] + distances[:, 2], CENTERS[:, 1] + distances[:, 3]
    ))
    scale, left, top, w, h = transform
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / scale
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
    scores = cls.max(1)
    classes = cls.argmax(1).astype(np.int32)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes, scores, classes, valid


def iou_one_to_many(box, boxes):
    if len(boxes) == 0:
        return np.empty(0, np.float32)
    tl = np.maximum(box[:2], boxes[:, :2])
    br = np.minimum(box[2:], boxes[:, 2:])
    wh = np.maximum(br - tl, 0)
    intersections = wh[:, 0] * wh[:, 1]
    a = np.maximum(box[2] - box[0], 0) * np.maximum(box[3] - box[1], 0)
    b = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    return intersections / np.maximum(a + b - intersections, 1e-9)


def nms_per_class(boxes, scores, classes, valid, conf, nms_iou, max_det=300):
    keep = []
    for c in np.unique(classes):
        idx = np.where((classes == c) & valid & (scores >= conf))[0]
        idx = idx[np.argsort(-scores[idx], kind='stable')]
        while idx.size:
            first = int(idx[0]); keep.append(first)
            if idx.size == 1:
                break
            overlap = iou_one_to_many(boxes[first], boxes[idx[1:]])
            idx = idx[1:][overlap <= nms_iou]
    keep = sorted(keep, key=lambda idx: -float(scores[idx]))[:max_det]
    return keep


def get_label(path, width, height):
    if not path.exists():
        raise FileNotFoundError(f'Missing label: {path}')
    objs = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f'Expected 5 YOLO fields at {path}:{n}: {line}')
        c = int(values[0]); x, y, w, h = map(float, values[1:])
        if not (0 <= c < NUM_CLASSES and all(np.isfinite([x, y, w, h])) and 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            raise ValueError(f'Invalid YOLO annotation: {path}:{n}')
        box = np.array([(x - w/2)*width, (y - h/2)*height,
                        (x + w/2)*width, (y + h/2)*height], np.float32)
        box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
        objs.append((c, box))
    return objs


def make_detections(boxes, scores, classes, keep):
    return [(int(classes[k]), float(scores[k]), boxes[k].copy()) for k in keep]


def match_image(gt, pred, iou_t=.5):
    """Greedy score-descending matching; returns TP FP FN for this image."""
    matched = set(); tp = 0; fp = 0
    for c, score, box in sorted(pred, key=lambda v: -v[1]):
        candidates = [(j, iou_one_to_many(box, b[None, :])[0]) for j, (gc, b) in enumerate(gt)
                      if gc == c and j not in matched]
        if candidates:
            j, iou = max(candidates, key=lambda x: x[1])
            if iou >= iou_t:
                matched.add(j); tp += 1; continue
        fp += 1
    return tp, fp, len(gt) - tp


def ap101(recalls, precisions):
    """101-point interpolated AP, without official COCO-specific exclusions."""
    recalls = np.asarray(recalls)
    precisions = np.asarray(precisions)
    if recalls.size == 0:
        return 0.0
    interpolated = np.maximum.accumulate(precisions[::-1])[::-1]
    thresholds = np.linspace(0, 1, 101)
    return float(np.mean([interpolated[recalls >= r].max(initial=0.0) for r in thresholds]))


def aggregate_ap(gt_by_image, preds_by_image):
    rows = []
    all_aps_50, all_aps = [], []
    for c in range(NUM_CLASSES):
        gt_c = {name: np.array([box for gc, box in gts if gc == c], dtype=np.float32).reshape(-1, 4)
                for name, gts in gt_by_image.items()}
        n_gt = sum(len(boxes) for boxes in gt_c.values())
        preds_c = [(name, score, box) for name, preds in preds_by_image.items()
                   for pc, score, box in preds if pc == c]
        preds_c.sort(key=lambda x: -x[1])
        class_aps = []
        for thr in IOU_THRESHOLDS:
            used = {name: np.zeros(len(boxes), dtype=bool) for name, boxes in gt_c.items()}
            tp, fp = [], []
            for name, score, box in preds_c:
                candidates = gt_c[name]
                ious = iou_one_to_many(box, candidates)
                if ious.size:
                    ious[used[name]] = -1
                best = int(np.argmax(ious)) if ious.size else -1
                correct = best >= 0 and ious[best] >= thr
                if correct:
                    used[name][best] = True
                tp.append(int(correct)); fp.append(int(not correct))
            tps = np.cumsum(tp)
            fps = np.cumsum(fp)
            recall = tps / max(n_gt, 1)
            precision = tps / np.maximum(tps + fps, 1)
            class_aps.append(ap101(recall, precision) if n_gt else float('nan'))
        row = {'class_id': c, 'gt_count': n_gt, 'prediction_count': len(preds_c),
               'AP50': class_aps[0], 'AP50_95': float(np.nanmean(class_aps)) if n_gt else float('nan')}
        rows.append(row)
        if n_gt:
            all_aps_50.append(class_aps[0]); all_aps.extend(class_aps)
    return rows, float(np.mean(all_aps_50)) if all_aps_50 else None, float(np.mean(all_aps)) if all_aps else None


def benchmark(args):
    hef_path = args.hef.resolve()
    images = sorted(p for p in args.images.iterdir() if p.is_file() and p.suffix.lower() in EXTS and not p.stem.upper().endswith('_GT'))
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise ValueError(f'No images in {args.images}')
    if not hef_path.exists():
        raise FileNotFoundError(hef_path)
    args.output.mkdir(parents=True, exist_ok=True)
    for p in images:
        if not (args.labels / (p.stem + '.txt')).exists():
            raise FileNotFoundError('Missing annotation: ' + str(args.labels / (p.stem + '.txt')))
    # Refuse overwriting earlier experiments, requiring a new --output directory.
    if (args.output / 'summary.json').exists():
        raise FileExistsError(f'Existing results in {args.output}; choose a new --output')
    cv2.setNumThreads(1)
    settings = {'hef': str(hef_path), 'hef_sha256': sha256(hef_path),
                'images': str(args.images.resolve()), 'labels': str(args.labels.resolve()),
                'image_count': len(images), 'warmup': args.warmup, 'num_classes': NUM_CLASSES, 'exclude_mask_suffix': '_GT',
                'operational_conf': .25, 'operational_nms_iou': .45,
                'ap_conf': .001, 'ap_nms_iou': .70,
                'ap_method': 'custom 101-point interpolation (not official COCO evaluation)',
                'max_detections_per_image': 300, 'grid_specs': GRID_SPECS,
                'model_output_assumption': '3549 x 1 class, 3549 x 64 DFL, P3/P4/P5',
                'python': sys.version, 'numpy': np.__version__, 'opencv': cv2.__version__}
    (args.output / 'settings.json').write_text(json.dumps(settings, indent=2, default=str))

    hef = HEF(str(hef_path))
    print('HEF input:', [(s.name, tuple(s.shape)) for s in hef.get_input_vstream_infos()])
    print('HEF outputs:', [(s.name, tuple(s.shape)) for s in hef.get_output_vstream_infos()])
    inputs = hef.get_input_vstream_infos()
    if len(inputs) != 1 or tuple(inputs[0].shape) != (416, 416, 3):
        raise ValueError('This diagnostic script supports only UINT8 NHWC 416x416x3 HEFs')
    outputs_info = hef.get_output_vstream_infos()
    output_shapes = {tuple(info.shape) for info in outputs_info}
    if (1, 3549, NUM_CLASSES) not in output_shapes or (1, 3549, 64) not in output_shapes:
        raise ValueError(f'Unexpected KolektorSDD2 output shapes: {output_shapes}')
    in_name = inputs[0].name

    timings = []
    gt_by_image, ap_by_image = {}, {}
    op_total = Counter()
    outputs_operational = args.output / 'predictions_operational.jsonl'
    outputs_ap = args.output / 'predictions_ap.jsonl'
    with VDevice() as device:
        config = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        group = device.configure(hef, config)[0]
        ip = InputVStreamParams.make(group, format_type=FormatType.UINT8)
        op = OutputVStreamParams.make(group, format_type=FormatType.FLOAT32)
        with InferVStreams(group, ip, op) as pipeline:
            with group.activate(group.create_params()):
                # Warmup includes real images; excluded from 500-image measurements.
                for i in range(args.warmup):
                    bgr = cv2.imread(str(images[i % len(images)]))
                    rgb, _ = letterbox(bgr)
                    pipeline.infer({in_name: rgb[None, ...]})
                with open(outputs_operational, 'w') as fop, open(outputs_ap, 'w') as fap:
                    for n, path in enumerate(images, 1):
                        start = time.perf_counter()
                        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                        read_end = time.perf_counter()
                        if bgr is None:
                            raise ValueError(f'Unable to read {path}')
                        h, w = bgr.shape[:2]
                        rgb, transf = letterbox(bgr)
                        prep_end = time.perf_counter()
                        out = pipeline.infer({in_name: rgb[None, ...]})
                        infer_end = time.perf_counter()
                        cls, dfl = output_tensors(out)
                        boxes, scores, classes, valid = decode(cls, dfl, transf)
                        op_keep = nms_per_class(boxes, scores, classes, valid, .25, .45)
                        ap_keep = nms_per_class(boxes, scores, classes, valid, .001, .70)
                        op_pred = make_detections(boxes, scores, classes, op_keep)
                        ap_pred = make_detections(boxes, scores, classes, ap_keep)
                        gt = get_label(args.labels / (path.stem + '.txt'), w, h)
                        post_end = time.perf_counter()
                        name = path.name
                        gt_by_image[name] = gt
                        ap_by_image[name] = ap_pred
                        tp, fp, fn = match_image(gt, op_pred)
                        op_total.update({'tp': tp, 'fp': fp, 'fn': fn})
                        timings.append({'image': name, 'read_ms': (read_end - start)*1000,
                                        'preprocess_ms': (prep_end - read_end)*1000,
                                        'infer_call_ms': (infer_end - prep_end)*1000,
                                        'postprocess_and_labels_ms': (post_end - infer_end)*1000,
                                        'total_ms': (post_end - start)*1000,
                                        'operational_detections': len(op_pred),
                                        'ap_detections': len(ap_pred), 'tp': tp, 'fp': fp, 'fn': fn})
                        for file, detections in ((fop, op_pred), (fap, ap_pred)):
                            file.write(json.dumps({'image': name, 'detections': [
                                {'class_id': c, 'confidence': score, 'xyxy': box.tolist()}
                                for c, score, box in detections]}) + '\n')
                        if n % 50 == 0 or n == len(images):
                            print(f'{n}/{len(images)} images; cumulative TP={op_total["tp"]} FP={op_total["fp"]} FN={op_total["fn"]}', flush=True)

    per_class, map50, map50_95 = aggregate_ap(gt_by_image, ap_by_image)
    p = op_total['tp'] / max(op_total['tp'] + op_total['fp'], 1)
    r = op_total['tp'] / max(op_total['tp'] + op_total['fn'], 1)
    f1 = 2*p*r / max(p+r, 1e-12)
    total_ms = sum(t['total_ms'] for t in timings)
    avg_inf = float(np.mean([t['infer_call_ms'] for t in timings]))
    summary = {
        'image_count': len(images), 'status': 'complete', 'conf_op': .25, 'nms_op': .45,
        'conf_ap': .001, 'nms_ap': .70, 'iou_match_op': .50,
        'tp': op_total['tp'], 'fp': op_total['fp'], 'fn': op_total['fn'],
        'precision': p, 'recall': r, 'f1': f1,
        'mAP50_custom101': map50, 'mAP50_95_custom101': map50_95,
        'mean_infer_call_ms': avg_inf,
        'mean_end_to_end_ms': total_ms / len(images),
        'per_image_end_to_end_fps': len(images)*1000 / total_ms,
        'timing_note': 'per-image timer: imread+letterbox+inference call+decode+NMS+label read; excludes setup, warmup, aggregate TP/FP/FN matching, output CSV/JSON writing and aggregate AP calculation; not a camera/video throughput measurement',
        'warning': 'Kolektor labels reconstructed as ONE enclosing bounding box per positive mask; not proven geometrically identical to original manuscript annotations. AP is custom COCO-like, not COCOeval. Do not directly compare timings with manuscript until time boundaries are reconciled.',
        'settings': settings
    }
    with open(args.output / 'per_image.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(timings[0]))
        writer.writeheader(); writer.writerows(timings)
    with open(args.output / 'per_class_ap.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_class[0]))
        writer.writeheader(); writer.writerows(per_class)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
    print('\n=== SUMMARY ===')
    for key in ('image_count','tp','fp','fn','precision','recall','f1','mAP50_custom101','mAP50_95_custom101','mean_infer_call_ms','mean_end_to_end_ms','per_image_end_to_end_fps'):
        print(f'{key}: {summary[key]}')
    print('Results:', args.output.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hef', type=Path, default=Path('hailo_benchmark/models/kolektor_yolov8n_416_hailo8l.hef'))
    parser.add_argument('--images', type=Path, default=Path('datasets/KolektorSDD2_yolo/images/test'))
    parser.add_argument('--labels', type=Path, default=Path('datasets/KolektorSDD2_yolo/labels/test'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--limit', type=int, default=None, help='Diagnostic run using first N images (choose separate output)')
    args = parser.parse_args()
    if args.warmup < 0 or (args.limit is not None and args.limit < 1):
        parser.error('warmup must be >=0 and limit must be >=1')
    benchmark(args)


if __name__ == '__main__':
    main()
