#!/usr/bin/env python3
"""Cross-check KolektorSDD2 exported Hailo detections using COCOeval.

Reads original test images only for width/height, reconstructed YOLO labels,
and saved predictions_ap.jsonl; does NOT rerun Hailo inference or edit labels.
COCOeval bbox AP: IoU .50:.05:.95, area=all, maxDets=300.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load_ground_truth(images_dir: Path, labels_dir: Path):
    names = sorted(p for p in images_dir.glob('*.png') if not p.stem.endswith('_GT'))
    if len(names) != 1004:
        raise ValueError(f'Expected 1004 original images, found {len(names)}')
    images, annotations, name_to_id = [], [], {}
    annotation_id = 1
    for index, path in enumerate(names, 1):
        label = labels_dir / (path.stem + '.txt')
        if not label.is_file():
            raise FileNotFoundError(label)
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f'Unable to read image {path}')
        height, width = image.shape[:2]
        images.append({'id': index, 'file_name': path.name, 'width': width, 'height': height})
        name_to_id[path.name] = (index, width, height)
        for line in label.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5 or int(parts[0]) != 0:
                raise ValueError(f'Invalid single-class YOLO label in {label}: {line}')
            _, cx, cy, bw, bh = map(float, parts)
            x, y = (cx - bw/2) * width, (cy - bh/2) * height
            w, h = bw * width, bh * height
            if min(w, h) <= 0:
                raise ValueError(f'Non-positive GT box in {label}')
            annotations.append({'id': annotation_id, 'image_id': index,
                                'category_id': 1, 'bbox': [x, y, w, h],
                                'area': w*h, 'iscrowd': 0})
            annotation_id += 1
    if len(annotations) != 110:
        raise ValueError(f'Expected 110 reconstructed ground-truth boxes, found {len(annotations)}')
    return {'info': {'description': 'KolektorSDD2 reconstructed YOLO detection labels'},
            'licenses': [], 'images': images, 'annotations': annotations,
            'categories': [{'id': 1, 'name': 'defect', 'supercategory': 'defect'}]}, name_to_id


def load_predictions(path: Path, lookup):
    predictions = []
    seen = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        name = item['image']
        if name in seen or name not in lookup:
            raise ValueError(f'Duplicate or unexpected image: {name}')
        seen.add(name)
        image_id, width, height = lookup[name]
        for det in item['detections']:
            if int(det['class_id']) != 0:
                raise ValueError(f'Unexpected class: {det}')
            x1, y1, x2, y2 = (float(v) for v in det['xyxy'])
            score = float(det['confidence'])
            if not np.isfinite([x1, y1, x2, y2, score]).all():
                raise ValueError(f'Nonfinite detection: {det}')
            if not (x2 > x1 and y2 > y1):
                raise ValueError(f'Invalid detection box: {det}')
            predictions.append({'image_id': image_id, 'category_id': 1,
                                'bbox': [x1, y1, x2-x1, y2-y1], 'score': score})
    if len(seen) != len(lookup):
        raise ValueError(f'Predictions cover {len(seen)} images, expected {len(lookup)}')
    return predictions


def extract_ap(evaluation, iou_index=None):
    # precision shape: [IoU thresholds, recall thresholds, categories, area ranges, max detections]
    precision = evaluation.eval['precision']
    values = precision if iou_index is None else precision[iou_index:iou_index+1]
    values = values[:, :, :, 0, -1]  # area='all', maxDets=300
    valid = values[values >= 0]
    if valid.size == 0:
        raise RuntimeError('COCOeval returned no valid precision cells')
    return float(valid.mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=Path('datasets/KolektorSDD2/test'))
    parser.add_argument('--labels', type=Path, default=Path('datasets/KolektorSDD2_yolo/labels/test'))
    parser.add_argument('--results', type=Path, default=Path('hailo_benchmark/results/kolektor_e2e'))
    args = parser.parse_args()
    coco_dataset, lookup = load_ground_truth(args.images, args.labels)
    coco_gt = COCO()
    coco_gt.dataset = coco_dataset
    coco_gt.createIndex()
    for model in ('yolov8n', 'yolov8s'):
        folder = args.results / f'{model}_1004'
        pred = load_predictions(folder / 'predictions_ap.jsonl', lookup)
        if not pred:
            raise ValueError(f'No predictions: {folder}')
        coco_dt = coco_gt.loadRes(pred)
        evaluation = COCOeval(coco_gt, coco_dt, 'bbox')
        evaluation.params.imgIds = sorted([x['id'] for x in coco_dataset['images']])
        evaluation.params.catIds = [1]
        evaluation.params.maxDets = [1, 10, 300]
        evaluation.params.iouThrs = np.linspace(0.50, 0.95, 10)
        evaluation.evaluate()
        evaluation.accumulate()
        ap50 = extract_ap(evaluation, 0)
        ap5095 = extract_ap(evaluation)
        summary = json.loads((folder / 'summary.json').read_text())
        result = {
            'evaluator': 'pycocotools COCOeval', 'iouType': 'bbox',
            'scope': 'area=all, maxDets=300, IoU 0.50:0.05:0.95',
            'image_count': len(lookup), 'gt_count': len(coco_dataset['annotations']),
            'prediction_count': len(pred), 'AP50': ap50, 'AP50_95': ap5095,
            'custom_AP50': summary['mAP50_custom101'],
            'custom_AP50_95': summary['mAP50_95_custom101'],
            'difference_AP50': ap50 - summary['mAP50_custom101'],
            'difference_AP50_95': ap5095 - summary['mAP50_95_custom101'],
            'annotation_note': 'One enclosing box per positive original mask; not verified against historic manuscript geometry.'
        }
        dest = folder / 'cocoeval_validation.json'
        dest.write_text(json.dumps(result, indent=2) + '\n')
        print('\n===', model, '===')
        print(json.dumps(result, indent=2))
        print('Saved:', dest)


if __name__ == '__main__':
    main()
