#!/usr/bin/env python3
"""Independent DeepPCB COCOeval validation of previously saved Hailo predictions.
No Hailo inference or modifications to original benchmark outputs.
Requires: pycocotools, opencv-python, numpy.
"""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--images', type=Path, default=Path('datasets/DeepPCB_yolo/images/test'))
    p.add_argument('--labels', type=Path, default=Path('datasets/DeepPCB_yolo/labels/test'))
    p.add_argument('--results-root', type=Path, default=Path('hailo_benchmark/results/deeppcb_e2e'))
    args = p.parse_args()
    images = sorted(x for x in args.images.iterdir() if x.is_file() and x.suffix.lower() in EXTS)
    assert len(images) == 500, f'Expected 500 DeepPCB images, found {len(images)}'
    image_ids = {}
    gt = {'info': {'description': 'DeepPCB YOLO ground truth converted to COCO'}, 'licenses': [],
          'images': [], 'annotations': [], 'categories': [{'id': k + 1, 'name': str(k)} for k in range(6)]}
    aid = 1
    for iid, image_path in enumerate(images, 1):
        im = cv2.imread(str(image_path))
        if im is None:
            raise RuntimeError(f'Cannot read image: {image_path}')
        height, width = im.shape[:2]
        image_ids[image_path.name] = iid
        gt['images'].append({'id': iid, 'file_name': image_path.name, 'width': width, 'height': height})
        label_path = args.labels / f'{image_path.stem}.txt'
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        for line_num, line in enumerate(label_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            values = line.split()
            if len(values) != 5:
                raise ValueError(f'Bad annotation {label_path}:{line_num}')
            c = int(values[0]); xc, yc, bw, bh = map(float, values[1:])
            if not 0 <= c < 6:
                raise ValueError(f'Invalid class in {label_path}:{line_num}')
            x1 = max(0., min(float(width), (xc-bw/2)*width))
            y1 = max(0., min(float(height), (yc-bh/2)*height))
            x2 = max(0., min(float(width), (xc+bw/2)*width))
            y2 = max(0., min(float(height), (yc+bh/2)*height))
            bw_px, bh_px = x2-x1, y2-y1
            if bw_px <= 0 or bh_px <= 0:
                raise ValueError(f'Degenerate box in {label_path}:{line_num}')
            gt['annotations'].append({'id': aid, 'image_id': iid, 'category_id': c+1,
                  'bbox': [x1,y1,bw_px,bh_px], 'area': bw_px*bh_px, 'iscrowd': 0})
            aid += 1
    print(f'Ground truth: {len(images)} images, {len(gt["annotations"])} boxes')
    if len(gt['annotations']) != 3140:
        raise ValueError('Ground-truth total differs from the 3140 reported in benchmark')
    for model in ('yolov8n_500', 'yolov8s_500'):
        folder = args.results_root / model
        path = folder / 'predictions_ap.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != len(images):
            raise ValueError(f'{model}: {len(rows)} rows; expected {len(images)}')
        seen = set()
        predictions = []
        for row in rows:
            name = row['image']
            if name not in image_ids or name in seen:
                raise ValueError(f'Unknown or repeated image name: {name}')
            seen.add(name)
            for d in row['detections']:
                c = int(d['class_id']); score = float(d['confidence'])
                x1, y1, x2, y2 = map(float, d['xyxy'])
                if c not in range(6) or not 0 <= score <= 1 or x2 <= x1 or y2 <= y1:
                    raise ValueError(f'Invalid prediction in {model}, image {name}: {d}')
                predictions.append({'image_id': image_ids[name], 'category_id': c+1,
                                    'bbox': [x1,y1,x2-x1,y2-y1], 'score': score})
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(predictions)
        ev = COCOeval(coco_gt, coco_dt, iouType='bbox')
        ev.params.imgIds = sorted(image_ids.values())
        ev.params.catIds = list(range(1,7))
        ev.params.maxDets = [1,10,300]
        ev.evaluate(); ev.accumulate()
        # Extract AP from accumulated precision tensor [IoU, Recall, Class, Area, MaxDets].
        # COCOeval.summarize() hard-codes 100 detections for stats[0], whereas
        # this experiment deliberately evaluates up to 300 detections per image.
        precision = ev.eval['precision']
        p = ev.params
        area_idx = p.areaRngLbl.index('all')
        det_idx = p.maxDets.index(300)
        all_values = precision[:, :, :, area_idx, det_idx]
        valid = all_values[all_values > -1]
        ap50_95 = float(np.mean(valid)) if valid.size else -1.0
        iou_50_idx = int(np.argmin(np.abs(p.iouThrs - 0.50)))
        if not np.isclose(p.iouThrs[iou_50_idx], 0.50):
            raise ValueError('COCOeval does not include IoU=0.50')
        values_50 = precision[iou_50_idx, :, :, area_idx, det_idx]
        valid_50 = values_50[values_50 > -1]
        ap50 = float(np.mean(valid_50)) if valid_50.size else -1.0
        if ap50_95 < 0 or ap50 < 0:
            raise ValueError('COCOeval AP undefined: check ground truth and predictions')
        custom = json.loads((folder / 'summary.json').read_text())
        print('\n===',model,'===')
        print('COCO AP50:95:', f'{ap50_95:.8f}')
        print('COCO AP50:', f'{ap50:.8f}')
        print('Custom AP50:95:', f'{custom["mAP50_95_custom101"]:.8f}')
        print('Custom AP50:', f'{custom["mAP50_custom101"]:.8f}')
        print('COCO - custom (AP50:95):', f'{ap50_95-custom["mAP50_95_custom101"]:+.8f}')
        print('COCO - custom (AP50):', f'{ap50-custom["mAP50_custom101"]:+.8f}')
        out = folder / 'cocoeval_validation.json'
        out.write_text(json.dumps({'evaluator':'pycocotools COCOeval', 'iouType':'bbox',
           'maxDets':[1,10,300], 'extraction':'COCOeval accumulated precision tensor, area=all, maxDets=300', 'image_count':len(images),'gt_count':len(gt['annotations']),
           'prediction_count':len(predictions), 'AP50_95':ap50_95,
           'AP50':ap50, 'custom_AP50_95':custom['mAP50_95_custom101'],
           'custom_AP50':custom['mAP50_custom101']},indent=2)+'\n')
        print('Saved:',out)

if __name__ == '__main__':
    main()
