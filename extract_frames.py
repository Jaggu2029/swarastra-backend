"""
Extract labeled frames from a Gujarati Sign Language alphabet video
using a CSV of {index, label, start(mm:ss), end(mm:ss)}.

Usage:
    python extract_frames.py --video gsl_source.mp4 --csv timestamps.csv \
        --outdir dataset --frames_per_label 15

Output structure (ready for image-classification training):
    dataset/
        ka/
            ka_000.jpg
            ka_001.jpg
            ...
        kha/
            kha_000.jpg
            ...
        ...
"""

import argparse
import csv
import os
import cv2


def mmss_to_seconds(ts: str) -> float:
    """Convert 'mm:ss' string to seconds (float)."""
    ts = ts.strip()
    parts = ts.split(":")
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    elif len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    else:
        raise ValueError(f"Unrecognized timestamp format: {ts}")


def read_labels(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "index": r["index"].strip(),
                    "label": r["label"].strip(),
                    "start": mmss_to_seconds(r["start"]),
                    "end": mmss_to_seconds(r["end"]),
                }
            )
    return rows


def sharpness(frame) -> float:
    """Higher = sharper. Uses variance of the Laplacian (standard blur metric)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extract_for_label(cap, fps, row, outdir, frames_per_label, margin,
                       candidate_multiplier=3, blur_percentile=40):
    """
    Sample `frames_per_label * candidate_multiplier` candidate frames from the
    [start+margin, end-margin] window, score each for blur (Laplacian variance),
    drop the blurriest `blur_percentile`% of candidates (motion frames), then
    keep `frames_per_label` frames evenly spread across what's left.
    """
    label = row["label"]
    start = row["start"] + margin
    end = row["end"] - margin
    if end <= start:
        start, end = row["start"], row["end"]

    label_dir = os.path.join(outdir, label)
    os.makedirs(label_dir, exist_ok=True)

    n_candidates = max(frames_per_label * candidate_multiplier, frames_per_label)
    if n_candidates == 1:
        cand_ts = [(start + end) / 2]
    else:
        step = (end - start) / (n_candidates - 1)
        cand_ts = [start + i * step for i in range(n_candidates)]

    candidates = []  # (timestamp, frame, sharpness)
    for t in cand_ts:
        frame_no = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            continue
        candidates.append((t, frame, sharpness(frame)))

    if not candidates:
        print(f"{label:>6}: [!] no frames could be read")
        return 0

    # drop the blurriest fraction (motion / transition frames)
    scores = sorted(c[2] for c in candidates)
    cutoff_idx = int(len(scores) * blur_percentile / 100)
    cutoff = scores[min(cutoff_idx, len(scores) - 1)]
    sharp_candidates = [c for c in candidates if c[2] >= cutoff]
    if len(sharp_candidates) < frames_per_label:
        sharp_candidates = candidates  # not enough left, fall back to all

    # pick frames_per_label evenly spread (by time) across the surviving candidates
    sharp_candidates.sort(key=lambda c: c[0])
    if len(sharp_candidates) <= frames_per_label:
        chosen = sharp_candidates
    else:
        idxs = [round(i * (len(sharp_candidates) - 1) / (frames_per_label - 1))
                for i in range(frames_per_label)]
        idxs = sorted(set(idxs))
        chosen = [sharp_candidates[i] for i in idxs]

    saved = 0
    for i, (t, frame, sc) in enumerate(chosen):
        fname = f"{label}_{i:03d}.jpg"
        cv2.imwrite(os.path.join(label_dir, fname), frame)
        saved += 1

    print(f"{label:>6}: saved {saved}/{frames_per_label} frames "
          f"(window {row['start']:.1f}s-{row['end']:.1f}s, "
          f"{len(candidates)} candidates, blur cutoff kept {len(sharp_candidates)})")
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to the downloaded mp4")
    ap.add_argument("--csv", required=True, help="Path to timestamps.csv")
    ap.add_argument("--outdir", default="dataset", help="Output dataset directory")
    ap.add_argument("--frames_per_label", type=int, default=20,
                     help="How many frames to keep per sign (more = bigger dataset)")
    ap.add_argument("--margin", type=float, default=0.15,
                     help="Seconds to trim off each side of the window to avoid "
                          "transition/motion blur frames")
    ap.add_argument("--blur_percentile", type=float, default=40,
                     help="Percent of blurriest candidate frames to discard per sign "
                          "(0 = keep everything, higher = stricter sharpness filtering)")
    ap.add_argument("--labels", default=None,
                     help="Comma-separated list of labels to (re-)extract, e.g. 'cha,Pa,gyna'. "
                          "If omitted, all labels in the CSV are processed.")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Could not open video file. Is it a valid mp4?")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps if fps else 0
    print(f"Video: {args.video}")
    print(f"FPS: {fps:.2f}, duration: {duration:.1f}s, total frames: {int(total_frames)}\n")

    rows = read_labels(args.csv)

    if args.labels:
        wanted = {s.strip() for s in args.labels.split(",")}
        rows = [r for r in rows if r["label"] in wanted]
        missing = wanted - {r["label"] for r in rows}
        if missing:
            print(f"[!] Labels not found in CSV: {missing}")
        print(f"Re-extracting only: {[r['label'] for r in rows]}\n")

    os.makedirs(args.outdir, exist_ok=True)

    total_saved = 0
    skipped = []
    for row in rows:
        if row["end"] > duration + 1:  # sanity check against actual video length
            print(f"  [!] WARNING: '{row['label']}' end time {row['end']:.1f}s "
                  f"exceeds video duration {duration:.1f}s — check your CSV/video match")
            skipped.append(row["label"])
        total_saved += extract_for_label(cap, fps, row, args.outdir, args.frames_per_label,
                                          args.margin, blur_percentile=args.blur_percentile)

    cap.release()
    print(f"\nDone. Total frames saved: {total_saved}")
    if skipped:
        print(f"Labels with out-of-range timestamps: {skipped}")


if __name__ == "__main__":
    main()
