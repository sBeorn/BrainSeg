"""
Thalamic Nuclei Morphometrics
==============================
data/subjects/<subject>/warped/labels/ maskelerinden:
  1. Her label için centroid (RAS mm) → 3D Slicer .fcsv pin noktaları
  2. Geometrik ölçümler → morphometrics_all.csv özet tablosu

Kullanım:
    python scripts/compute_morphometrics.py

Çıktı:
    outputs/morphometrics/<subject>.fcsv      ← 3D Slicer'da aç
    outputs/morphometrics/morphometrics_all.csv
"""

import csv
import os
import re
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

try:
    from skimage.measure import marching_cubes
    from scipy.ndimage import gaussian_filter
    _MC_AVAILABLE = True
except ImportError:
    _MC_AVAILABLE = False

# utils/metrics.py — geometrik morphometrics (volume, surface, compactness vb.)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.metrics import compute_geometric_morphometrics

# ─────────────────────────────────────────────────────────────────────────────
# Yol Ayarları
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(PROJECT_DIR, "data", "subjects")
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "morphometrics"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSV Sütun Sırası
# ─────────────────────────────────────────────────────────────────────────────
CSV_COLUMNS = [
    "subject", "label", "side",
    "centroid_x_ras_mm", "centroid_y_ras_mm", "centroid_z_ras_mm",
    "volume_mm3", "surface_area_mm2",
    "compactness", "elongation", "flatness",
    "bbox_x_mm", "bbox_y_mm", "bbox_z_mm",
    "bbox_fill_ratio", "connected_components",
    "skeleton_length_mm", "skeleton_max_radius_mm",
]


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı Fonksiyonlar
# ─────────────────────────────────────────────────────────────────────────────

def list_subjects() -> list:
    """DATA_DIR altındaki subject klasörlerini döndür."""
    entries = sorted(os.listdir(DATA_DIR))
    return [e for e in entries if os.path.isdir(os.path.join(DATA_DIR, e))]


def list_labels(subject: str) -> list:
    """
    Bir subjectin warped/labels/ klasöründeki maskeleri parse et.
    Döndürür: [(side, label_name, path), ...]
    """
    label_dir = os.path.join(DATA_DIR, subject, "warped", "labels")
    pattern   = re.compile(r"^(left|right)_(.+)_warped\.nii\.gz$")
    results   = []
    for fname in sorted(os.listdir(label_dir)):
        m = pattern.match(fname)
        if m:
            side, label = m.group(1), m.group(2)
            results.append((side, label, os.path.join(label_dir, fname)))
    return results


def load_mask(path: str):
    """
    NIfTI maskesini yükle, RAS+ orientasyonuna çevir.
    Döndürür: (affine, binary_mask, voxel_spacing)
    """
    img  = nib.load(path)
    img  = nib.as_closest_canonical(img)   # RAS+ yönelimi garantile
    data = img.get_fdata(dtype=np.float32)
    mask = (data > 0.5).astype(bool)
    zooms = img.header.get_zooms()[:3]
    return img.affine, mask, tuple(float(z) for z in zooms)


def surface_landmarks(mask: np.ndarray, affine: np.ndarray) -> dict:
    """
    Maskenin yuzeyindeki 6 anatomik uc noktayi RAS mm olarak hesapla.
    RAS: x=Sag(+)/Sol(-), y=On(+)/Arka(-), z=Ust(+)/Alt(-)

    Dondurulen anahtarlar:
      ant, post, sup, inf, lat, med
    Her deger [x, y, z] RAS mm listesi.
    """
    if not _MC_AVAILABLE or mask.sum() < 5:
        return {}

    smooth = gaussian_filter(mask.astype(float), sigma=0.8)
    try:
        verts_vox, _, _, _ = marching_cubes(smooth, level=0.5)
    except Exception:
        return {}

    if len(verts_vox) == 0:
        return {}

    ones      = np.ones((len(verts_vox), 1))
    verts_ras = (affine @ np.hstack([verts_vox, ones]).T).T[:, :3]

    def pt(v):
        return [round(float(v[0]), 3), round(float(v[1]), 3), round(float(v[2]), 3)]

    return {
        "ant":  pt(verts_ras[np.argmax(verts_ras[:, 1])]),   # max y = en on
        "post": pt(verts_ras[np.argmin(verts_ras[:, 1])]),   # min y = en arka
        "sup":  pt(verts_ras[np.argmax(verts_ras[:, 2])]),   # max z = en ust
        "inf":  pt(verts_ras[np.argmin(verts_ras[:, 2])]),   # min z = en alt
        "lat":  pt(verts_ras[np.argmax(verts_ras[:, 0])]),   # max x = en sag
        "med":  pt(verts_ras[np.argmin(verts_ras[:, 0])]),   # min x = en sol
    }


def centroid_ras(mask: np.ndarray, affine: np.ndarray):
    """
    Binary maskenin ağırlık merkezini RAS mm koordinatlarında hesapla.
    Boş maskede None döndürür.
    """
    if mask.sum() == 0:
        return None
    vox = np.array(ndimage.center_of_mass(mask))             # voxel [i,j,k]
    ras = affine @ np.array([vox[0], vox[1], vox[2], 1.0])  # mm RAS
    return [round(float(ras[0]), 3), round(float(ras[1]), 3), round(float(ras[2]), 3)]


# ─────────────────────────────────────────────────────────────────────────────
# .fcsv Yazma (3D Slicer Markup Fiducial)
# ─────────────────────────────────────────────────────────────────────────────

def write_fcsv(subject: str, landmarks: list):
    """
    landmarks = [(label_name, side, [x,y,z]), ...]
    Sıra tutarlı olsun diye label alfabetik, sonra left→right.

    Slicer 5.x formatı:
      CoordinateSystem = RAS   (nibabel affine → RAS mm)
    """
    path = os.path.join(OUT_DIR, f"{subject}.fcsv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("# Markups fiducial file version = 5.0\n")
        f.write("# CoordinateSystem = RAS\n")
        f.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
        for i, (label, side, coords) in enumerate(landmarks):
            if coords is None:
                x, y, z = 0.0, 0.0, 0.0
                print(f"    UYARI  {subject} / {label}_{side}: boş maske → 0,0,0")
            else:
                x, y, z = coords
            node_id  = f"vtkMRMLMarkupsFiducialNode_{i}"
            lm_label = f"{label}_{side}"
            f.write(
                f"{node_id},{x:.3f},{y:.3f},{z:.3f},"
                f"0,0,0,1,1,1,0,{lm_label},,\n"
            )
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Yuzey Landmark .fcsv Yazma
# ─────────────────────────────────────────────────────────────────────────────

DIRECTION_ORDER = ["ant", "post", "sup", "inf", "lat", "med"]
DIRECTION_TR    = {"ant": "On", "post": "Arka", "sup": "Ust",
                   "inf": "Alt", "lat": "Lat", "med": "Med"}

def write_surface_fcsv(subject: str, surface_data: list):
    """
    surface_data = [(label, side, {dir: [x,y,z]}), ...]
    Iki cikti uretir:
      1. outputs/morphometrics/<subject>_surface.fcsv  — tum labellar (Slicer scripti icin)
      2. outputs/morphometrics/per_label/<subject>/<label>_surface.fcsv — her label ayri
    """
    # ── 1. Tek buyuk fcsv (tum labellar) ──────────────────────────────────────
    all_path = os.path.join(OUT_DIR, f"{subject}_surface.fcsv")
    idx = 0
    with open(all_path, "w", encoding="utf-8", newline="") as f:
        f.write("# Markups fiducial file version = 5.0\n")
        f.write("# CoordinateSystem = RAS\n")
        f.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
        for label, side, lm_dict in surface_data:
            for direction in DIRECTION_ORDER:
                coords = lm_dict.get(direction)
                if coords is None:
                    continue
                x, y, z  = coords
                lm_label = f"{label}_{side}_{DIRECTION_TR[direction]}"
                node_id  = f"vtkMRMLMarkupsFiducialNode_{idx}"
                f.write(f"{node_id},{x:.3f},{y:.3f},{z:.3f},0,0,0,1,1,1,0,{lm_label},,\n")
                idx += 1

    # ── 2. Her label icin ayri fcsv ────────────────────────────────────────────
    per_label_dir = os.path.join(OUT_DIR, "per_label", subject)
    os.makedirs(per_label_dir, exist_ok=True)

    # label bazinda grupla
    from collections import defaultdict
    by_label = defaultdict(list)
    for label, side, lm_dict in surface_data:
        by_label[label].append((side, lm_dict))

    for label, sides in by_label.items():
        lpath = os.path.join(per_label_dir, f"{label}_surface.fcsv")
        with open(lpath, "w", encoding="utf-8", newline="") as f:
            f.write("# Markups fiducial file version = 5.0\n")
            f.write("# CoordinateSystem = RAS\n")
            f.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
            pt_idx = 0
            for side, lm_dict in sides:
                for direction in DIRECTION_ORDER:
                    coords = lm_dict.get(direction)
                    if coords is None:
                        continue
                    x, y, z  = coords
                    lm_label = f"{label}_{side}_{DIRECTION_TR[direction]}"
                    node_id  = f"vtkMRMLMarkupsFiducialNode_{pt_idx}"
                    f.write(f"{node_id},{x:.3f},{y:.3f},{z:.3f},0,0,0,1,1,1,0,{lm_label},,\n")
                    pt_idx += 1

    return all_path


# ─────────────────────────────────────────────────────────────────────────────
# CSV Yazma
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(all_rows: list):
    path = os.path.join(OUT_DIR, "morphometrics_all.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Ana Fonksiyon
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    subjects = list_subjects()
    print(f"\n{'='*62}")
    print(f"  Thalamic Nuclei Morphometrics  —  {len(subjects)} subject")
    print(f"{'='*62}")
    print(f"  Veri: {DATA_DIR}")
    print(f"  Çıktı: {OUT_DIR}\n")

    all_rows = []

    for subj in subjects:
        print(f"  [{subj}]")
        label_list = list_labels(subj)

        if not label_list:
            print("    Hiç label bulunamadı, atlanıyor.")
            continue

        # label_name alfabetik sıra, sonra left→right
        sorted_labels = sorted(
            label_list,
            key=lambda t: (t[1], 0 if t[0] == "left" else 1)
        )

        landmarks     = []   # (label, side, [x,y,z])
        surface_data  = []   # (label, side, {dir: [x,y,z]})
        rows          = []   # CSV satırları

        for side, label, fpath in sorted_labels:
            affine, mask, spacing = load_mask(fpath)

            # Centroid (RAS mm)
            c = centroid_ras(mask, affine)

            # Yuzey uc noktalari (mesh uzerinde)
            surf_lm = surface_landmarks(mask, affine)
            surface_data.append((label, side, surf_lm))

            # Geometrik metrikler
            geo = compute_geometric_morphometrics(mask, spacing)

            bbox = geo.get("bbox_size_mm", [None, None, None])

            row = {
                "subject":            subj,
                "label":              label,
                "side":               side,
                "centroid_x_ras_mm":  c[0] if c else "",
                "centroid_y_ras_mm":  c[1] if c else "",
                "centroid_z_ras_mm":  c[2] if c else "",
                "volume_mm3":         geo.get("volume_mm3", ""),
                "surface_area_mm2":   geo.get("surface_area_mm2", ""),
                "compactness":        geo.get("compactness", ""),
                "elongation":         geo.get("elongation", ""),
                "flatness":           geo.get("flatness", ""),
                "bbox_x_mm":          bbox[0] if bbox else "",
                "bbox_y_mm":          bbox[1] if bbox else "",
                "bbox_z_mm":          bbox[2] if bbox else "",
                "bbox_fill_ratio":    geo.get("bbox_fill_ratio", ""),
                "connected_components": geo.get("connected_components", ""),
                "skeleton_length_mm": geo.get("skeleton_length_mm", ""),
                "skeleton_max_radius_mm": geo.get("skeleton_max_radius_mm", ""),
            }

            rows.append(row)
            landmarks.append((label, side, c))

            status = "OK" if mask.sum() > 0 else "BOŞ"
            vol    = geo.get("volume_mm3", 0)
            print(f"    {status:4s}  {side:5s}  {label:<15s}  {vol:8.1f} mm³")

        all_rows.extend(rows)

        # Centroid .fcsv yaz
        fcsv_path = write_fcsv(subj, landmarks)
        print(f"    >> {os.path.basename(fcsv_path)}  ({len(landmarks)} centroid nokta)")

        # Yuzey landmark .fcsv yaz
        surf_path = write_surface_fcsv(subj, surface_data)
        n_surf = sum(len(d[2]) for d in surface_data)
        print(f"    >> {os.path.basename(surf_path)}  ({n_surf} yuzey nokta)\n")

    # Özet CSV yaz
    csv_path = write_csv(all_rows)
    print(f"{'='*62}")
    print(f"  TAMAMLANDI  —  {len(all_rows)} label işlendi")
    print(f"  CSV  : {csv_path}")
    print(f"{'='*62}\n")
    print("  3D Slicer'da nasil acilir:")
    print("  1. Slicer'i baslat, atlas_warped.nii.gz'yi yukle")
    print("  2. File > Add Data > .fcsv dosyasini sec")
    print("  3. Markups modulunde pin noktalari gorunur\n")


if __name__ == "__main__":
    main()
