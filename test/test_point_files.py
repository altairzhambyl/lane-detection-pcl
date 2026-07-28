#!/usr/bin/env python3
"""
Round-trip and failure-mode tests for the on-disk cloud readers.

Pure numpy on purpose: these must run in a container without ROS, because the
real-data ingestion path is the part of the pipeline CI can actually verify.
"""

import struct

import numpy as np
import pytest
from ransac_seg.point_files import (
    iter_cloud_files,
    read_cloud,
    read_kitti_bin,
    read_pcd,
)


def write_bin(path, xyz, intensity=None):
    """Write an (N, 3) array as a KITTI-style float32 xyzi sweep."""
    xyz = np.asarray(xyz, dtype=np.float32)
    if intensity is None:
        intensity = np.linspace(0.0, 1.0, xyz.shape[0], dtype=np.float32)
    packed = np.column_stack([xyz, np.asarray(intensity, dtype=np.float32)])
    packed.astype(np.float32).tofile(str(path))
    return path


def pcd_header(fields, data, points):
    """Build a PCD header block for float32 fields."""
    return (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {' '.join(fields)}\n"
        f"SIZE {' '.join(['4'] * len(fields))}\n"
        f"TYPE {' '.join(['F'] * len(fields))}\n"
        f"COUNT {' '.join(['1'] * len(fields))}\n"
        f"WIDTH {points}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {points}\n"
        f"DATA {data}\n"
    )


SAMPLE = np.array(
    [
        [1.5, -2.25, 0.5],
        [10.0, 3.125, -1.75],
        [-4.5, 0.0, 2.0],
        [0.25, 0.75, -0.125],
    ],
    dtype=np.float64,
)


def test_read_kitti_bin_round_trip(tmp_path):
    """A written xyzi sweep reads back as exactly its xyz columns."""
    path = write_bin(tmp_path / "0000.bin", SAMPLE)
    out = read_kitti_bin(str(path))
    assert out.shape == (4, 3)
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, SAMPLE)


def test_read_kitti_bin_rejects_truncated_file(tmp_path):
    """A sweep cut mid-point raises rather than silently dropping data."""
    path = tmp_path / "truncated.bin"
    path.write_bytes(struct.pack("<fffff", 1.0, 2.0, 3.0, 0.5, 9.0))
    with pytest.raises(ValueError) as excinfo:
        read_kitti_bin(str(path))
    message = str(excinfo.value)
    assert "truncated.bin" in message
    assert "20 bytes" in message


def test_read_kitti_bin_rejects_empty_file(tmp_path):
    """An empty file is a corrupt sweep, not a zero-point cloud."""
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    with pytest.raises(ValueError):
        read_kitti_bin(str(path))


def test_read_pcd_ascii_round_trip(tmp_path):
    """An ASCII PCD reads back to the values that were written."""
    path = tmp_path / "cloud.pcd"
    rows = "\n".join(f"{x} {y} {z}" for x, y, z in SAMPLE)
    path.write_text(pcd_header(["x", "y", "z"], "ascii", len(SAMPLE)) + rows + "\n")
    np.testing.assert_allclose(read_pcd(str(path)), SAMPLE)


def test_read_pcd_binary_round_trip(tmp_path):
    """A binary PCD with an xyzi layout reads back its xyz columns."""
    path = tmp_path / "cloud_bin.pcd"
    payload = np.column_stack(
        [SAMPLE, np.arange(len(SAMPLE), dtype=np.float64)]
    ).astype(np.float32)
    with open(path, "wb") as handle:
        handle.write(
            pcd_header(["x", "y", "z", "intensity"], "binary", len(SAMPLE)).encode()
        )
        handle.write(payload.tobytes())
    np.testing.assert_allclose(read_pcd(str(path)), SAMPLE)


def test_read_pcd_honours_reordered_fields(tmp_path):
    """Fields declared as z y x are located by name, not assumed positional."""
    path = tmp_path / "reordered.pcd"
    flipped = SAMPLE[:, ::-1]  # stored as z, y, x
    rows = "\n".join(f"{a} {b} {c}" for a, b, c in flipped)
    path.write_text(pcd_header(["z", "y", "x"], "ascii", len(SAMPLE)) + rows + "\n")
    np.testing.assert_allclose(read_pcd(str(path)), SAMPLE)


def test_read_pcd_binary_honours_reordered_fields(tmp_path):
    """The binary path resolves field offsets by name as well."""
    path = tmp_path / "reordered_bin.pcd"
    flipped = SAMPLE[:, ::-1].astype(np.float32)
    with open(path, "wb") as handle:
        handle.write(pcd_header(["z", "y", "x"], "binary", len(SAMPLE)).encode())
        handle.write(np.ascontiguousarray(flipped).tobytes())
    np.testing.assert_allclose(read_pcd(str(path)), SAMPLE)


def test_read_pcd_rejects_binary_compressed(tmp_path):
    """binary_compressed is named in the error instead of being mis-parsed."""
    path = tmp_path / "compressed.pcd"
    with open(path, "wb") as handle:
        handle.write(pcd_header(["x", "y", "z"], "binary_compressed", 1).encode())
        handle.write(b"\x00" * 32)
    with pytest.raises(NotImplementedError) as excinfo:
        read_pcd(str(path))
    assert "binary_compressed" in str(excinfo.value)


def test_read_pcd_rejects_missing_xyz(tmp_path):
    """A cloud without x/y/z reports the fields it actually found."""
    path = tmp_path / "no_xyz.pcd"
    path.write_text(pcd_header(["u", "v", "w"], "ascii", 1) + "1 2 3\n")
    with pytest.raises(ValueError) as excinfo:
        read_pcd(str(path))
    assert "u v w" in str(excinfo.value)


def test_read_pcd_rejects_missing_data_line(tmp_path):
    """A header with no DATA line is a hard error, not an empty cloud."""
    path = tmp_path / "headerless.pcd"
    path.write_text("VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n")
    with pytest.raises(ValueError):
        read_pcd(str(path))


def test_read_cloud_dispatches_on_suffix(tmp_path):
    """read_cloud routes .bin and .pcd, and rejects anything else."""
    bin_path = write_bin(tmp_path / "sweep.bin", SAMPLE)
    pcd_path = tmp_path / "sweep.pcd"
    rows = "\n".join(f"{x} {y} {z}" for x, y, z in SAMPLE)
    pcd_path.write_text(pcd_header(["x", "y", "z"], "ascii", len(SAMPLE)) + rows + "\n")

    np.testing.assert_array_equal(read_cloud(str(bin_path)), SAMPLE)
    np.testing.assert_allclose(read_cloud(str(pcd_path)), SAMPLE)

    other = tmp_path / "sweep.ply"
    other.write_text("ply\n")
    with pytest.raises(ValueError) as excinfo:
        read_cloud(str(other))
    assert ".ply" in str(excinfo.value)


def test_iter_cloud_files_is_numerically_sorted(tmp_path):
    """Unpadded sweep indices replay in capture order, not lexicographic."""
    for index in (10, 2, 1):
        write_bin(tmp_path / f"sweep{index}.bin", SAMPLE)
    (tmp_path / "notes.txt").write_text("ignored")

    names = [path.rsplit("/", 1)[-1] for path in iter_cloud_files(str(tmp_path))]
    assert names == ["sweep1.bin", "sweep2.bin", "sweep10.bin"]


def test_disk_to_pointcloud2_round_trip(tmp_path):
    """
    A sweep read off disk survives the full ingestion path unchanged.

    This is the contract ``tools/kitti_publisher.py`` depends on: file ->
    read_cloud -> xyz_to_pointcloud2 -> the decode every node runs. float32 is
    the wire format, so the comparison is at float32 precision.
    """
    pytest.importorskip("sensor_msgs")
    from builtin_interfaces.msg import Time
    from ransac_seg.cloud_io import pointcloud2_to_xyz, xyz_to_pointcloud2

    path = write_bin(tmp_path / "sweep.bin", SAMPLE)
    points = read_cloud(str(path))
    msg = xyz_to_pointcloud2(points, "lidar_front", Time())

    assert msg.width == len(SAMPLE)
    assert msg.point_step == 12
    np.testing.assert_allclose(pointcloud2_to_xyz(msg), SAMPLE, rtol=0, atol=1e-6)


def test_iter_cloud_files_honours_pattern(tmp_path):
    """The glob pattern narrows the selection."""
    write_bin(tmp_path / "keep.bin", SAMPLE)
    write_bin(tmp_path / "drop.bin", SAMPLE)
    names = [
        path.rsplit("/", 1)[-1] for path in iter_cloud_files(str(tmp_path), "keep*")
    ]
    assert names == ["keep.bin"]
