#!/usr/bin/env python3
"""
Fit a plane from N 3D points and export it as a triangle mesh PLY.

Usage examples:

  - From whitespace TXT (x y z per line):
      python fit_plane_to_mesh.py --input points.txt --output plane.ply

  - From CSV:
      python fit_plane_to_mesh.py --input points.csv --output plane.ply

  - From PLY point cloud:
      python fit_plane_to_mesh.py --input cloud.ply --output plane.ply

  - Control size using point-projected bounds + margin (meters):
      python fit_plane_to_mesh.py --input points.txt --output plane.ply --margin 0.05

  - Or force a fixed square size (meters), centered at centroid:
      python fit_plane_to_mesh.py --input points.txt --output plane.ply --fixed-size 1.0

The script:
  1) Loads Nx3 points from TXT/CSV/NPY/PLY
  2) Fits plane via SVD, obtaining centroid C and normal n
  3) Builds an orthonormal basis (u, v) on the plane
  4) Either:
     - Uses projected point bounds (in u,v) + margin; or
     - Builds a fixed-size square
  5) Creates a quad (two triangles) mesh and writes PLY
  6) Also writes a sidecar TXT with plane equation ax+by+cz+d=0
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import open3d as o3d


# Edit this list to enter coordinates directly in code.
# Each inner list/tuple is a point [x, y, z] in meters.
# Example below defines points roughly on the plane z = 0.1x + 0.05y + 0.2
INLINE_POINTS = [
    [1, 0, 0],
    [0, 1, 0],
    [-1,1, 0],
    [-1,-1, 0]
]


def load_points(path: Path) -> np.ndarray:
    """Load Nx3 points from TXT/CSV/NPY/PLY. Raises on error."""
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    ext = path.suffix.lower()
    if ext in {".txt", ".xyz"}:
        pts = np.loadtxt(str(path))
    elif ext == ".csv":
        # Heuristic: try comma delimiter first, fall back to any whitespace
        try:
            pts = np.loadtxt(str(path), delimiter=",")
        except Exception:
            pts = np.loadtxt(str(path))
    elif ext == ".npy":
        pts = np.load(str(path))
    elif ext == ".ply":
        # Try point cloud; if empty, try mesh vertices
        pcd = o3d.io.read_point_cloud(str(path))
        if len(pcd.points) == 0:
            mesh = o3d.io.read_triangle_mesh(str(path))
            if len(mesh.vertices) == 0:
                raise ValueError(f"PLY has no points or vertices: {path}")
            pts = np.asarray(mesh.vertices)
        else:
            pts = np.asarray(pcd.points)
    else:
        # Attempt whitespace text as a generic fallback
        try:
            pts = np.loadtxt(str(path))
        except Exception as e:
            raise ValueError(f"Unsupported input format: {ext} ({e})")

    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got shape {pts.shape}")
    if pts.shape[0] < 3:
        raise ValueError("At least 3 non-collinear points are required to define a plane")
    return pts


def fit_plane_svd(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit plane via SVD. Returns (centroid C, unit normal n, d) with ax+by+cz+d=0.

    d is computed so that the plane passes through the centroid.
    """
    C = points.mean(axis=0)
    Q = points - C
    # SVD on centered coordinates; normal is last right-singular vector
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    n = Vt[-1, :]
    n = n / np.linalg.norm(n)
    d = -float(np.dot(n, C))
    return C, n, d


def plane_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct an orthonormal basis (u, v) on the plane orthogonal to n."""
    n = n / np.linalg.norm(n)
    # Pick a vector not parallel to n
    if abs(n[0]) < 0.9:
        a = np.array([1.0, 0.0, 0.0])
    else:
        a = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return u, v


def build_plane_corners(
    points: np.ndarray,
    C: np.ndarray,
    n: np.ndarray,
    margin: float | None,
    fixed_size: float | None,
) -> np.ndarray:
    """Compute 4 corner vertices of a planar quad in 3D.

    If fixed_size provided: returns square of that size centered at C.
    Otherwise: uses projected bounds of input points in (u,v) coordinates, expanded by margin.
    """
    u, v = plane_basis(n)

    if fixed_size is not None:
        half = float(fixed_size) / 2.0
        corners_local = np.array([
            [-half, -half],
            [ half, -half],
            [ half,  half],
            [-half,  half],
        ])
    else:
        # Project points onto plane basis
        Q = points - C
        U = Q @ u
        V = Q @ v
        umin, umax = float(U.min()), float(U.max())
        vmin, vmax = float(V.min()), float(V.max())
        m = float(margin or 0.0)
        umin -= m; umax += m
        vmin -= m; vmax += m
        corners_local = np.array([
            [umin, vmin],
            [umax, vmin],
            [umax, vmax],
            [umin, vmax],
        ])

    # Map local (u,v) to 3D: P = C + u*u_local + v*v_local
    corners3d = C + corners_local[:, 0:1] * u + corners_local[:, 1:2] * v
    return corners3d


def mesh_from_corners(corners: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Create a 2-triangle mesh (quad) from 4 corner vertices in CCW order."""
    if corners.shape != (4, 3):
        raise ValueError("Expected 4x3 corners array")
    vertices = o3d.utility.Vector3dVector(corners.astype(np.float64))
    triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32))
    mesh = o3d.geometry.TriangleMesh(vertices, triangles)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh


def write_plane_info(path: Path, n: np.ndarray, d: float, C: np.ndarray):
    """Write plane equation and centroid to a sidecar text file."""
    text = (
        "Plane fit result\n"
        "================\n"
        f"Normal (a,b,c): [{n[0]:.8f}, {n[1]:.8f}, {n[2]:.8f}]\n"
        f"d: {d:.8f}   (ax+by+cz+d=0)\n"
        f"Centroid: [{C[0]:.8f}, {C[1]:.8f}, {C[2]:.8f}]\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fit plane from 3D points and export as PLY mesh")
    p.add_argument("--input", "-i", type=Path, required=False, help="Input points file: TXT/CSV/NPY/PLY")
    p.add_argument("--output", "-o", type=Path, default=Path("output/ground.ply"), help="Output PLY mesh path")
    p.add_argument("--margin", type=float, default=0.0, help="Margin (meters) added around projected bounds")
    p.add_argument("--fixed-size", type=float, default=None, help="If set, create square of this size centered at centroid")
    p.add_argument("--orient-up", action="store_true", help="Flip normal to have positive Z component if needed")
    p.add_argument("--inline", action="store_true", help="Use INLINE_POINTS defined in this script instead of reading a file")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.inline or args.input is None:
        pts = np.asarray(INLINE_POINTS, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
            raise ValueError("INLINE_POINTS must be a list of Nx3 coordinates with N>=3")
        print(f"Using INLINE_POINTS with {pts.shape[0]} points")
    else:
        pts = load_points(args.input)
    C, n, d = fit_plane_svd(pts)

    if args.orient_up and n[2] < 0:
        n = -n
        d = -d

    corners = build_plane_corners(
        points=pts,
        C=C,
        n=n,
        margin=args.margin,
        fixed_size=args.fixed_size,
    )
    mesh = mesh_from_corners(corners)

    # Ensure parent directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(args.output), mesh):
        raise RuntimeError(f"Failed to write mesh: {args.output}")

    # Write plane equation info next to the mesh
    info_path = args.output.with_suffix(".txt")
    write_plane_info(info_path, n, d, C)

    print(f"Saved plane mesh: {args.output}")
    print(f"Plane info: {info_path}")


if __name__ == "__main__":
    sys.exit(main())
