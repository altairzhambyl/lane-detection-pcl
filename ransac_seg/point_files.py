#!/usr/bin/env python3
"""
ROS-free readers for on-disk LiDAR sweeps (KITTI ``.bin`` and PCD).

Every cloud in this package used to be generated procedurally, so nothing
proved the pipeline works on recorded sensor data. Parsing lives here rather
than in ``tools/`` for two reasons: the publishers import it as a library, and
the tests exercise it without ROS on the path, which is what lets CI run the
real-data path inside a plain Python container.
"""

import os
import re
from glob import glob

import numpy as np

# KITTI Velodyne sweeps are a flat float32 stream of (x, y, z, reflectance).
_KITTI_STRIDE = 4
_KITTI_POINT_BYTES = _KITTI_STRIDE * 4

_PCD_HEADER_KEYS = (
    "FIELDS",
    "SIZE",
    "TYPE",
    "COUNT",
    "WIDTH",
    "HEIGHT",
    "POINTS",
    "DATA",
)

# PCD encodes element type as a letter plus a byte width; numpy wants a kind
# character, and only these three kinds are legal in the PCD spec.
_PCD_KIND = {"I": "i", "U": "u", "F": "f"}


def read_kitti_bin(path):
    """
    Read a KITTI/nuScenes ``.bin`` sweep and return its ``(N, 3)`` xyz.

    The reflectance channel is dropped: nothing downstream consumes it, and
    keeping the array at three columns matches ``cloud_io`` conventions.
    """
    raw = np.fromfile(path, dtype=np.float32)
    n_bytes = raw.nbytes
    if n_bytes == 0 or n_bytes % _KITTI_POINT_BYTES != 0:
        raise ValueError(
            f"{path}: {n_bytes} bytes is not a whole number of "
            f"{_KITTI_POINT_BYTES}-byte (x, y, z, intensity) float32 points; "
            "the file is truncated or is not a KITTI .bin sweep"
        )
    return np.ascontiguousarray(raw.reshape(-1, _KITTI_STRIDE)[:, :3], dtype=np.float64)


def _parse_pcd_header(handle):
    """Consume the PCD header from a binary handle and return its fields."""
    header = {}
    while True:
        line = handle.readline()
        if not line:
            raise ValueError("PCD header ended before the DATA line")
        text = line.decode("ascii", errors="replace").strip()
        if not text or text.startswith("#"):
            continue
        key, _, value = text.partition(" ")
        key = key.upper()
        if key in _PCD_HEADER_KEYS:
            header[key] = value.split()
        if key == "DATA":
            break

    missing = [k for k in ("FIELDS", "SIZE", "TYPE", "DATA") if k not in header]
    if missing:
        raise ValueError(f"PCD header is missing {', '.join(missing)}")
    return header


def _pcd_layout(header):
    """Return (names, numpy dtype strings, per-field element counts)."""
    names = header["FIELDS"]
    sizes = [int(s) for s in header["SIZE"]]
    types = [t.upper() for t in header["TYPE"]]
    counts = [int(c) for c in header.get("COUNT", ["1"] * len(names))]

    if not (len(sizes) == len(types) == len(counts) == len(names)):
        raise ValueError(
            "PCD header is inconsistent: FIELDS/SIZE/TYPE/COUNT have "
            f"{len(names)}/{len(sizes)}/{len(types)}/{len(counts)} entries"
        )

    dtypes = []
    for name, size, kind in zip(names, sizes, types):
        if kind not in _PCD_KIND:
            raise ValueError(f"PCD field {name!r} has unsupported TYPE {kind!r}")
        dtypes.append(f"<{_PCD_KIND[kind]}{size}")
    return names, dtypes, counts


def _xyz_indices(names):
    """Return the positions of x, y and z inside the field list."""
    lowered = [n.lower() for n in names]
    try:
        # Located by name, not by position: PCDs in the wild reorder or pad the
        # field list, and a reader that assumes x/y/z come first silently
        # returns permuted coordinates.
        return [lowered.index(axis) for axis in ("x", "y", "z")]
    except ValueError:
        raise ValueError(
            f"PCD has no x/y/z fields; FIELDS were {' '.join(names)}"
        ) from None


def _n_points(header, fallback):
    """Return the declared point count, preferring POINTS over WIDTH*HEIGHT."""
    if "POINTS" in header:
        return int(header["POINTS"][0])
    if "WIDTH" in header and "HEIGHT" in header:
        return int(header["WIDTH"][0]) * int(header["HEIGHT"][0])
    return fallback


def read_pcd(path):
    """
    Read an ASCII or uncompressed-binary PCD and return its ``(N, 3)`` xyz.

    ``binary_compressed`` is rejected rather than mis-parsed, because its LZF
    payload is column-major and reading it as binary yields plausible-looking
    garbage.
    """
    with open(path, "rb") as handle:
        header = _parse_pcd_header(handle)
        data_format = header["DATA"][0].lower()
        names, dtypes, counts = _pcd_layout(header)
        ix, iy, iz = _xyz_indices(names)

        if data_format == "binary_compressed":
            raise NotImplementedError(
                f"{path}: PCD DATA format 'binary_compressed' is not supported; "
                "re-save the file as 'ascii' or 'binary' (pcl_convert2ascii)"
            )

        if data_format == "ascii":
            table = np.loadtxt(handle, dtype=np.float64, ndmin=2)
            if table.size == 0:
                return np.zeros((0, 3), dtype=np.float64)
            # COUNT > 1 fields occupy several columns each, so the column of a
            # field is the sum of the counts before it.
            starts = np.cumsum([0] + counts[:-1])
            cols = [int(starts[i]) for i in (ix, iy, iz)]
            return np.ascontiguousarray(table[:, cols], dtype=np.float64)

        if data_format == "binary":
            struct_dtype = np.dtype(
                [
                    (f"f{i}", dtypes[i], (counts[i],))
                    if counts[i] > 1
                    else (f"f{i}", dtypes[i])
                    for i in range(len(names))
                ]
            )
            payload = handle.read()
            expected = _n_points(header, len(payload) // struct_dtype.itemsize)
            usable = min(expected, len(payload) // struct_dtype.itemsize)
            table = np.frombuffer(payload, dtype=struct_dtype, count=usable)
            out = np.empty((usable, 3), dtype=np.float64)
            for col, index in enumerate((ix, iy, iz)):
                values = table[f"f{index}"]
                out[:, col] = values[:, 0] if values.ndim > 1 else values
            return out

    raise ValueError(f"{path}: unsupported PCD DATA format {data_format!r}")


def read_cloud(path):
    """Dispatch on the file suffix to the right reader."""
    suffix = os.path.splitext(str(path))[1].lower()
    if suffix == ".bin":
        return read_kitti_bin(path)
    if suffix == ".pcd":
        return read_pcd(path)
    raise ValueError(
        f"{path}: unsupported cloud extension {suffix!r}; expected .bin or .pcd"
    )


def _natural_key(path):
    """Sort key that orders ``sweep2`` before ``sweep10``."""
    name = os.path.basename(path)
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def iter_cloud_files(directory, pattern="*"):
    """
    Yield cloud file paths from ``directory`` in sorted order.

    Sorting is numeric-aware so an unpadded sweep index replays in capture
    order instead of lexicographic order.
    """
    matches = [
        path
        for path in glob(os.path.join(str(directory), pattern))
        if os.path.isfile(path)
        and os.path.splitext(path)[1].lower() in (".bin", ".pcd")
    ]
    for path in sorted(matches, key=_natural_key):
        yield path
