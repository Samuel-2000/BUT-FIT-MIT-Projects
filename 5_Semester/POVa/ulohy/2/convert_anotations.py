#!/usr/bin/env python3
"""
convert_annotations.py

Converts the provided annotation.json (list of items, each with a "label" list)
to Ultralytics/YOLO text label files:

    <class_id> <cx> <cy> <w> <h>

Coordinates must be normalized (0..1) and center-based.
The annotation.json uses percentage coordinates (0..100), x,y = top-left.
"""
import json
import os
import argparse
import shutil

# You can add more candidate keys here if your json uses a different name
FILENAME_KEYS = ("image", "filename", "name", "original_filename", "file", "image_id", "external_id")

def find_filename(item):
    for k in FILENAME_KEYS:
        if k in item and item[k]:
            return os.path.basename(item[k])
    # fallback: maybe item has a 'id' numeric -> assume <id>.jpg
    if "id" in item:
        return str(item["id"]) + ".jpg"
    return None

def parse_annotation_item(item):
    # `item` is expected to contain a "label" field which is a list of object dicts
    boxes = []
    labels = item.get("label", []) or []
    for obj in labels:
        # If your annotations are nested differently, you may need to adjust these keys
        try:
            x = float(obj["x"])        # left (%)
            y = float(obj["y"])        # top (%)
            w = float(obj["width"])    # width (%)
            h = float(obj["height"])   # height (%)
            class_list = obj.get("rectanglelabels", obj.get("labels", []))
            if isinstance(class_list, list) and len(class_list) > 0:
                class_name = class_list[0]
            else:
                class_name = str(class_list)
        except Exception:
            # skip malformed annotation
            continue

        # convert percentage -> normalized center coords (0..1)
        cx = (x + w / 2.0) / 100.0
        cy = (y + h / 2.0) / 100.0
        nw = w / 100.0
        nh = h / 100.0

        boxes.append((class_name, cx, cy, nw, nh))
    return boxes

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", "-a", required=True, help="Path to annotation.json")
    p.add_argument("--images", "-i", required=True, help="Path to images folder (train or val)")
    p.add_argument("--labels-out", "-l", required=True, help="Path where label txt files will be written (e.g. data/labels/train)")
    p.add_argument("--classes", "-c", nargs="+", default=["mouse"], help="List of class names in order (mouse first -> class 0)")
    p.add_argument("--copy-images", action="store_true", help="If set, copies images to an output images folder matching labels_out/../images")
    args = p.parse_args()

    with open(args.annotations, "r", encoding="utf-8") as f:
        data = json.load(f)

    # create lookup dict from filename -> boxes
    lookup = {}
    for item in data:
        fname = find_filename(item)
        if fname is None:
            continue
        boxes = parse_annotation_item(item)
        lookup[fname] = boxes

    os.makedirs(args.labels_out, exist_ok=True)

    # Optionally copy images into a sibling images/ folder (so dataset dir is self-contained).
    if args.copy_images:
        images_out_dir = os.path.join(os.path.dirname(args.labels_out), "images")
        os.makedirs(images_out_dir, exist_ok=True)
    else:
        images_out_dir = None

    # iterate over image files in provided images folder
    for img_name in os.listdir(args.images):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(args.images, img_name)

        # write label file (even if empty)
        base = os.path.splitext(img_name)[0]
        label_path = os.path.join(args.labels_out, base + ".txt")

        entries = lookup.get(img_name, [])
        # convert class names -> class ids
        with open(label_path, "w", encoding="utf-8") as lf:
            for (class_name, cx, cy, w, h) in entries:
                if class_name in args.classes:
                    class_id = args.classes.index(class_name)
                else:
                    # If unseen class, append to classes list dynamically (prints a warning)
                    print(f"Warning: found class '{class_name}' not in provided classes list; adding as new entry.")
                    args.classes.append(class_name)
                    class_id = args.classes.index(class_name)
                lf.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        if images_out_dir:
            shutil.copy2(img_path, os.path.join(images_out_dir, img_name))

    print("Done. Labels written to:", args.labels_out)
    print("Final classes order (index -> name):")
    for i, n in enumerate(args.classes):
        print(f"  {i}: {n}")

if __name__ == "__main__":
    main()
