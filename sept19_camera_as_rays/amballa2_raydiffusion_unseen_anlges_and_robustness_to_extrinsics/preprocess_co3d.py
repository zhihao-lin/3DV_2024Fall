"""
Script to pre-process camera poses and bounding boxes for CO3Dv2 dataset. This is
important because computing the bounding boxes from the masks is a significant
bottleneck.

First, you should pre-compute the bounding boxes since this takes a long time.

Usage:
    python -m preprocess.preprocess_co3d --category all --precompute_bbox \
        --co3d_v2_dir /path/to/co3d_v2
    python -m preprocess.preprocess_co3d --category all \
        --co3d_v2_dir /path/to/co3d_v2
"""

import argparse
import gzip
import json
import os.path as osp
from glob import glob

import ipdb
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

# fmt: off
CATEGORIES = [
    "chair",
]
# fmt: on


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="chair")
    parser.add_argument("--output_dir", type=str, default="data/chair/")
    parser.add_argument("--co3d_v2_dir", type=str, default="data/")
    parser.add_argument(
        "--min_quality",
        type=float,
        default=0.5,
        help="Minimum viewpoint quality score.",
    )
    parser.add_argument("--precompute_bbox", action="store_true")
    return parser


def mask_to_bbox(mask):
    """
    xyxy format
    """
    mask = mask > 0.4
    if not np.any(mask):
        return []
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax) + 1, int(rmax) + 1]


def precompute_bbox(co3d_dir, category, output_dir):
    """
    Precomputes bounding boxes for all frames using the masks. This can be an expensive
    operation because it needs to load every mask in the dataset. Thus, we only want to
    run this once, whereas processing the rest of the dataset is fast.
    """
    category_dir = osp.join(co3d_dir, category)
    print("Precomputing bbox for:", category)
    all_masks = sorted(glob(osp.join(category_dir, "*", "masks", "*.png")))
    print(all_masks)
    bboxes = {}
    for mask_filename in tqdm(all_masks):
        mask = plt.imread(mask_filename)
        # /Dataset/category/sequence/masks/mask.png -> category/sequence/mask/mask.png
        mask_filename = mask_filename.replace(osp.dirname(category_dir), "")[1:]
        try:
            bboxes[mask_filename] = mask_to_bbox(mask)
        except IndexError:
            ipdb.set_trace()
    output_file = osp.join(output_dir, f"{category}_bbox.jgz")
    with gzip.open(output_file, "w") as f:
        f.write(json.dumps(bboxes).encode("utf-8"))


def process_poses(co3d_dir, category, output_dir, min_quality):
    category_dir = osp.join(co3d_dir, args.category)
    print("Processing category:", category)
    frame_file = osp.join(category_dir, "frame_annotations.jgz")
    sequence_file = osp.join(category_dir, "sequence_annotations.jgz")
    # subset_lists_file = osp.join(category_dir, "set_lists/set_lists_fewview_dev.json")
    subset_lists_file = osp.join(category_dir, "set_lists/set_lists_manyview_test_0.json")

    bbox_file = osp.join(output_dir, f"{category}_bbox.jgz")

    with open(subset_lists_file) as f:
        subset_lists_data = json.load(f)

    with gzip.open(sequence_file, "r") as fin:
        sequence_data = json.loads(fin.read())
        print(f"Number of sequences: {(sequence_data)}")

    with gzip.open(frame_file, "r") as fin:
        frame_data = json.loads(fin.read())
        print(f"Number of frames: {len(frame_data)}")

    with gzip.open(bbox_file, "r") as fin:
        bbox_data = json.loads(fin.read())

    frame_data_processed = {}
    for f_data in frame_data:
        sequence_name = f_data["sequence_name"]
        if sequence_name not in frame_data_processed:
            frame_data_processed[sequence_name] = {}
        frame_data_processed[sequence_name][f_data["frame_number"]] = f_data
        # print(f_data)

    good_quality_sequences = set()
    for seq_data in sequence_data:
        # print(seq_data)
        # print(seq_data["viewpoint_quality_score"])
        # if seq_data["viewpoint_quality_score"] > min_quality:
        good_quality_sequences.add(seq_data["sequence_name"])
        print(seq_data["sequence_name"])

    for subset in ["train", "test"]:
        category_data = {}  # {sequence_name: [{filepath, R, T}]}
        for seq_name, frame_number, filepath in subset_lists_data[subset]:
            if seq_name not in good_quality_sequences:
                print(f"Skipping {seq_name} because quality is too low.")
                continue

            if seq_name not in category_data:
                category_data[seq_name] = []

            mask_path = filepath.replace("images", "masks").replace(".jpg", ".png")
            bbox = bbox_data[mask_path]
            if bbox == []:
                # Mask did not include any object.
                continue

            frame_data = frame_data_processed[seq_name][frame_number]
            # print(bbox)
            category_data[seq_name].append(
                {
                    "filepath": filepath,
                    "R": frame_data["viewpoint"]["R"],
                    "T": frame_data["viewpoint"]["T"],
                    "focal_length": frame_data["viewpoint"]["focal_length"],
                    "principal_point": frame_data["viewpoint"]["principal_point"],
                    "bbox": bbox,
                }
            )

        output_file = osp.join(args.output_dir, f"{args.category}_{subset}.jgz")
        with gzip.open(output_file, "w") as f:
            f.write(json.dumps(category_data).encode("utf-8"))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    if args.category == "all":
        categories = CATEGORIES
    else:
        categories = [args.category]
    if args.precompute_bbox:
        for category in categories:
            precompute_bbox(
                co3d_dir=args.co3d_v2_dir,
                category=category,
                output_dir=args.output_dir,
            )
    else:
        for category in categories:
            process_poses(
                co3d_dir=args.co3d_v2_dir,
                category=category,
                output_dir=args.output_dir,
                min_quality=args.min_quality,
            )
