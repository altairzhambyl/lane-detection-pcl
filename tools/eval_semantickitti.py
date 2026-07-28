#!/usr/bin/env python3
r"""
Evaluate ground segmentation against SemanticKITTI's per-point labels.

This is the only benchmark in the repo that runs on real sensor data. Every
other figure comes from `tools/benchmarks.py`, which builds its own clouds and
therefore measures the implementation, not the world. Real Velodyne ground is
not one plane — the road is cambered, the sidewalk rises over a kerb and the
terrain class rolls — so a single global plane underfits it by construction.
That gap is the point of measuring here.

Ground truth follows the class set used by the published ground-segmentation
benchmarks (Patchwork, Patchwork++), so the numbers are comparable to theirs:

    40 road, 44 parking, 48 sidewalk, 49 other-ground,
    60 lane-marking, 72 terrain

Points labelled 0 (unlabeled) are excluded from scoring, not counted as
negatives — the standard protocol.

Usage:
    python3 tools/eval_semantickitti.py --root ~/datasets/semantickitti \\
        --sequences 00 05 08 --limit 500

    python3 tools/eval_semantickitti.py --root ~/datasets/semantickitti \\
        --sequences 00 01 02 03 04 05 06 07 08 09 10 --json results.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ransac_seg.ground_plane import (  # noqa: E402
    DEFAULT_DISTANCE_THRESHOLD,
    segment_ground,
)
from ransac_seg.plane_filter import PlaneFilter  # noqa: E402
from ransac_seg.point_files import read_kitti_bin  # noqa: E402

GROUND_CLASSES = (40, 44, 48, 49, 60, 72)
UNLABELED = 0

# The fixed-height baseline is scored at every threshold in this range and
# reported at its best one, chosen on the test data itself. That is deliberately
# generous — an oracle no deployed system could have — because a baseline tuned
# to lose proves nothing. RANSAC has to beat the best a fixed cut can ever do.
HEIGHT_CANDIDATES = np.arange(-2.10, -0.79, 0.05)

# Semantic ids are 16-bit, so a full lookup table is 64 KB and indexing it is
# always in range. Measured 2.5x faster than np.isin on a 123k-point frame,
# which is worth having across 23k frames.
GROUND_LUT = np.zeros(1 << 16, dtype=bool)
GROUND_LUT[list(GROUND_CLASSES)] = True


def sweep_counts(z, is_ground):
    """
    Return (tp, fp) for every candidate height, one boolean scan each.

    A cumulative-histogram version of this was tried and measured *slower*
    (2.8 ms vs 1.3 ms on 123k points): numpy's boolean reductions beat
    np.histogram at this size, so the obvious-looking loop stays.
    """
    ground_z, other_z = z[is_ground], z[~is_ground]
    tp = np.array([np.count_nonzero(ground_z < h) for h in HEIGHT_CANDIDATES])
    fp = np.array([np.count_nonzero(other_z < h) for h in HEIGHT_CANDIDATES])
    return tp, fp


def read_labels(path):
    """Read a SemanticKITTI ``.label`` file and return its semantic class ids."""
    # Upper 16 bits carry the instance id; the semantic class is the lower half.
    return np.fromfile(path, dtype=np.uint32) & 0xFFFF


class Confusion:
    """Accumulates TP/FP/FN over frames and derives the summary metrics."""

    def __init__(self):
        """Start an empty accumulator."""
        self.tp = self.fp = self.fn = self.tn = 0
        self.latencies_ms = []
        self.frames = 0
        self.fallbacks = 0

    def add(self, predicted, truth, scored):
        """Fold one frame's masks into the running totals."""
        pred, true = predicted[scored], truth[scored]
        self.tp += int(np.count_nonzero(pred & true))
        self.fp += int(np.count_nonzero(pred & ~true))
        self.fn += int(np.count_nonzero(~pred & true))
        self.tn += int(np.count_nonzero(~pred & ~true))
        self.frames += 1

    @property
    def metrics(self):
        """Return IoU/precision/recall/F1 plus latency percentiles."""
        tp, fp, fn = self.tp, self.fp, self.fn
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        lat = np.asarray(self.latencies_ms) if self.latencies_ms else np.zeros(1)
        return {
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "frames": self.frames,
            "fallback_frames": self.fallbacks,
            "latency_p50_ms": float(np.percentile(lat, 50)),
            "latency_p95_ms": float(np.percentile(lat, 95)),
            "latency_p99_ms": float(np.percentile(lat, 99)),
            "latency_mean_ms": float(lat.mean()),
        }


def evaluate_sequence(
    root,
    sequence,
    limit=None,
    temporal=False,
    seed=0,
    distance_threshold=0.08,
    max_range=None,
    forward_only=True,
):
    """Run both the RANSAC pipeline and the legacy height rule over one sequence."""
    base = os.path.join(root, "dataset", "sequences", sequence)
    velo_dir, label_dir = os.path.join(base, "velodyne"), os.path.join(base, "labels")
    if not os.path.isdir(velo_dir):
        raise FileNotFoundError(
            f"missing velodyne data for sequence {sequence}: {velo_dir}"
        )
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"missing labels for sequence {sequence}: {label_dir}")

    names = sorted(f[:-4] for f in os.listdir(velo_dir) if f.endswith(".bin"))
    if limit:
        # Even stride rather than the first N, so a sequence is sampled across
        # its whole route instead of only its opening street.
        names = names[:: max(1, len(names) // limit)][:limit]

    ransac = Confusion()
    # One accumulator per candidate height, so the baseline can be reported at
    # its best threshold rather than at one somebody picked in advance.
    height_counts = np.zeros((HEIGHT_CANDIDATES.size, 3), dtype=np.int64)
    plane_filter = PlaneFilter() if temporal else None

    for name in names:
        points = read_kitti_bin(os.path.join(velo_dir, name + ".bin"))
        labels = read_labels(os.path.join(label_dir, name + ".label"))
        if labels.shape[0] != points.shape[0]:
            raise ValueError(
                f"{sequence}/{name}: {points.shape[0]} points but {labels.shape[0]} labels"
            )

        truth = GROUND_LUT[labels]
        scored = labels != UNLABELED

        if max_range is not None:
            # Match PerceptionNode._range_gate exactly, including its x > 0 term.
            # The nodes never segment the rear half: those returns are a
            # different piece of road and drag the single plane fit off true.
            # Scoring the full sweep measured a configuration the pipeline does
            # not run, and understated it by 3.5 IoU points on seq 08.
            radius = np.linalg.norm(points[:, :2], axis=1)
            keep = (radius >= 0.5) & (radius <= max_range)
            if forward_only:
                keep &= points[:, 0] > 0.0
            points, truth, scored = points[keep], truth[keep], scored[keep]

        start = time.perf_counter()
        mask, model = segment_ground(
            points, distance_threshold=distance_threshold, seed=seed
        )
        if plane_filter is not None:
            plane_filter.update(model)
        elapsed = (time.perf_counter() - start) * 1000.0

        ransac.latencies_ms.append(elapsed)
        if model is None:
            ransac.fallbacks += 1
        ransac.add(mask, truth, scored)

        z, t = points[scored, 2], truth[scored]
        tp_h, fp_h = sweep_counts(z, t)
        height_counts[:, 0] += tp_h
        height_counts[:, 1] += fp_h
        # Whatever ground the cut did not take is a false negative.
        height_counts[:, 2] += int(np.count_nonzero(t)) - tp_h

    tp, fp, fn = height_counts.T
    iou_per_height = tp / np.maximum(tp + fp + fn, 1)
    best = int(np.argmax(iou_per_height))

    return {
        "ransac": ransac.metrics,
        "legacy": {
            "iou": float(iou_per_height[best]),
            "precision": float(tp[best] / max(tp[best] + fp[best], 1)),
            "recall": float(tp[best] / max(tp[best] + fn[best], 1)),
            "best_height": float(HEIGHT_CANDIDATES[best]),
        },
        "n_frames": len(names),
        # Raw counts travel with the summary so the overall row can be a true
        # micro-average over points rather than a mean of per-sequence ratios.
        "counts": {
            "ransac": (ransac.tp, ransac.fp, ransac.fn),
            "legacy": (int(tp[best]), int(fp[best]), int(fn[best])),
        },
        "latencies_ms": ransac.latencies_ms,
    }


def micro_average(results, key):
    """Pool every sequence's counts and derive one IoU/precision/recall."""
    tp = sum(r["counts"][key][0] for r in results.values())
    fp = sum(r["counts"][key][1] for r in results.values())
    fn = sum(r["counts"][key][2] for r in results.values())
    union = tp + fp + fn
    return {
        "iou": tp / union if union else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
    }


def main():
    """Evaluate the requested sequences and print a comparison table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default=os.path.expanduser("~/datasets/semantickitti")
    )
    parser.add_argument("--sequences", nargs="+", default=["08"])
    parser.add_argument("--limit", type=int, default=None, help="frames per sequence")
    parser.add_argument("--temporal", action="store_true", help="run the plane filter")
    parser.add_argument("--json", default=None, help="write raw results here")
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=DEFAULT_DISTANCE_THRESHOLD,
        help=(
            "RANSAC inlier band in metres; follows the package default "
            f"({DEFAULT_DISTANCE_THRESHOLD}) so this script cannot drift from it"
        ),
    )
    parser.add_argument(
        "--max-range",
        type=float,
        default=60.0,
        help="range gate in metres; defaults to the nodes' own max_range",
    )
    parser.add_argument(
        "--full-sweep",
        action="store_true",
        help="score the whole 360 sweep instead of the forward half the nodes gate to",
    )
    args = parser.parse_args()

    results = {}
    for sequence in args.sequences:
        print(f"seq {sequence} ...", end=" ", flush=True)
        results[sequence] = evaluate_sequence(
            args.root,
            sequence,
            limit=args.limit,
            temporal=args.temporal,
            distance_threshold=args.distance_threshold,
            max_range=args.max_range,
            forward_only=not args.full_sweep,
        )
        r = results[sequence]["ransac"]
        print(f"{results[sequence]['n_frames']} frames, IoU {r['iou']:.3f}")

    print(f"\n{'seq':>5} {'frames':>7} | {'RANSAC':^31} | {'best fixed height':^24}")
    print(
        f"{'':>5} {'':>7} | {'IoU':>6} {'prec':>6} {'rec':>6} {'p95 ms':>8} "
        f"| {'IoU':>6} {'rec':>6} {'at z<':>7}"
    )
    print("-" * 86)
    for seq, res in results.items():
        r, g = res["ransac"], res["legacy"]
        print(
            f"{seq:>5} {res['n_frames']:>7} | {r['iou']:>6.3f} {r['precision']:>6.3f} "
            f"{r['recall']:>6.3f} {r['latency_p95_ms']:>8.1f} "
            f"| {g['iou']:>6.3f} {g['recall']:>6.3f} {g['best_height']:>7.2f}"
        )

    # Micro-average over every scored point, which is what published tables report.
    tot_frames = sum(r["n_frames"] for r in results.values())
    overall = micro_average(results, "ransac")
    overall_legacy = micro_average(results, "legacy")
    all_lat = np.asarray([ms for r in results.values() for ms in r["latencies_ms"]])
    mean_p95 = float(np.percentile(all_lat, 95)) if all_lat.size else 0.0
    print("-" * 86)
    print(
        f"{'ALL':>5} {tot_frames:>7} | {overall['iou']:>6.3f} {overall['precision']:>6.3f} "
        f"{overall['recall']:>6.3f} {mean_p95:>8.1f} "
        f"| {overall_legacy['iou']:>6.3f} {overall_legacy['recall']:>6.3f}"
    )

    fallbacks = sum(r["ransac"]["fallback_frames"] for r in results.values())
    if fallbacks:
        print(
            f"\nRANSAC found no plane on {fallbacks}/{tot_frames} frames (height fallback used)"
        )

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
