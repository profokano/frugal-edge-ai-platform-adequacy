from pathlib import Path
import csv
import hashlib
import cv2
import numpy as np

root = Path("datasets/KolektorSDD2/test")
out = Path("datasets/KolektorSDD2_yolo")
images_out = out / "images"
labels_out = out / "labels/test"

images_out.mkdir(parents=True, exist_ok=True)
labels_out.mkdir(parents=True, exist_ok=True)

# Usamos links simbólicos para não duplicar imagens.
image_link = images_out / "test"
if not image_link.exists() and not image_link.is_symlink():
    image_link.symlink_to(root.resolve(), target_is_directory=True)

images = sorted(
    p for p in root.glob("*.png")
    if not p.stem.endswith("_GT")
)

assert len(images) == 1004, (
    f"Esperadas 1004 imagens; encontradas {len(images)}"
)

positive = 0
negative = 0
rows = []

for image_path in images:
    mask_path = root / f"{image_path.stem}_GT.png"
    assert mask_path.is_file(), f"Máscara ausente: {mask_path}"

    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        raise RuntimeError(f"Erro ao ler: {image_path.name}")

    H, W = image.shape[:2]
    assert mask.shape == (H, W), image_path.name

    ys, xs = np.where(mask > 0)

    label_path = labels_out / f"{image_path.stem}.txt"

    if len(xs) == 0:
        label_path.write_text("")
        negative += 1
        box_count = 0
    else:
        # x2/y2 são limites exclusivos em coordenadas de pixels.
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1

        xc = (x1 + x2) / (2 * W)
        yc = (y1 + y2) / (2 * H)
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H

        label_path.write_text(
            f"0 {xc:.10f} {yc:.10f} "
            f"{bw:.10f} {bh:.10f}\n"
        )

        positive += 1
        box_count = 1

    rows.append({
        "image": image_path.name,
        "image_sha256": hashlib.sha256(
            image_path.read_bytes()
        ).hexdigest(),
        "mask": mask_path.name,
        "mask_sha256": hashlib.sha256(
            mask_path.read_bytes()
        ).hexdigest(),
        "label": label_path.name,
        "label_sha256": hashlib.sha256(
            label_path.read_bytes()
        ).hexdigest(),
        "boxes": box_count
    })

assert positive == 110, f"Positivas: {positive}"
assert negative == 894, f"Negativas: {negative}"

manifest = out / "dataset_manifest.csv"

with manifest.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("=== CONVERSÃO CONCLUÍDA ===")
print("Imagens:", len(images))
print("Positivas:", positive)
print("Negativas:", negative)
print("Caixas de referência:", positive)
print("Manifesto:", manifest)
print("Imagens:", image_link)
print("Anotações:", labels_out)
