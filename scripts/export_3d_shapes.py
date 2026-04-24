"""
3D Shape Export — Thalamic Nuclei
==================================
data/subjects/<subject>/warped/labels/ maskelerinden marching cubes ile VTK yüzey mesh'leri üretir.

Her label için ayrı bir .vtk (polydata) dosyası oluşturulur.
Koordinatlar RAS mm uzayında yazılır — 3D Slicer doğrudan açar.

Kullanım:
    python scripts/export_3d_shapes.py

Çıktı:
    outputs/shapes/<subject>/AD_left.vtk
    outputs/shapes/<subject>/AD_right.vtk
    ...
"""

import os
import re
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

try:
    from skimage.measure import marching_cubes
except ImportError:
    print("HATA: scikit-image yuklu degil.  pip install scikit-image")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Yol Ayarları
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(PROJECT_DIR, "data", "subjects")
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "shapes"
)

# Küçük label'lar için minimum voxel eşiği (altında mesh güvenilmez)
MIN_VOXELS = 5

# ─────────────────────────────────────────────────────────────────────────────
# VTK ASCII Polydata Yazıcı
# ─────────────────────────────────────────────────────────────────────────────

def write_vtk(verts: np.ndarray, faces: np.ndarray, path: str, label_name: str):
    """
    Triangulated surface mesh'i VTK ASCII polydata formatında yaz.
    verts: (N, 3) float — RAS mm koordinatları
    faces: (M, 3) int  — triangle vertex indisleri
    """
    n_pts  = len(verts)
    n_tri  = len(faces)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="ascii") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"{label_name}\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"\nPOINTS {n_pts} float\n")
        for v in verts:
            f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        f.write(f"\nPOLYGONS {n_tri} {4 * n_tri}\n")
        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Mesh Üretim
# ─────────────────────────────────────────────────────────────────────────────

def mask_to_mesh(mask: np.ndarray, affine: np.ndarray):
    """
    Binary maske → (verts_ras, faces) triangulated surface.
    marching_cubes voxel indis uzayında çalışır, affine ile RAS mm'ye çevrilir.
    Döndürür None çifti, mesh üretilemezse.
    """
    n = int(mask.sum())
    if n < MIN_VOXELS:
        return None, None

    # Küçük Gaussian ile hafif yumuşatma — voxel pikselasyonunu azaltır
    from scipy.ndimage import gaussian_filter
    smooth = gaussian_filter(mask.astype(float), sigma=0.8)

    try:
        verts_vox, faces, _, _ = marching_cubes(smooth, level=0.5)
    except (ValueError, RuntimeError):
        return None, None

    if len(verts_vox) == 0 or len(faces) == 0:
        return None, None

    # Voxel indis koordinatlarını RAS mm'ye çevir
    ones       = np.ones((len(verts_vox), 1))
    verts_h    = np.hstack([verts_vox, ones])          # (N, 4)
    verts_ras  = (affine @ verts_h.T).T[:, :3]         # (N, 3)

    return verts_ras, faces


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı Fonksiyonlar
# ─────────────────────────────────────────────────────────────────────────────

def list_subjects():
    return sorted(
        e for e in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, e))
    )


def list_labels(subject):
    label_dir = os.path.join(DATA_DIR, subject, "warped", "labels")
    pat = re.compile(r"^(left|right)_(.+)_warped\.nii\.gz$")
    out = []
    for fname in sorted(os.listdir(label_dir)):
        m = pat.match(fname)
        if m:
            out.append((m.group(1), m.group(2), os.path.join(label_dir, fname)))
    return out


def load_mask(path):
    img   = nib.load(path)
    img   = nib.as_closest_canonical(img)
    data  = img.get_fdata(dtype=np.float32)
    return img.affine, (data > 0.5).astype(bool)


def generate_brain_surface(subject: str, out_dir: str):
    """
    atlas_warped.nii.gz'den beyin yuzey mesh'i uret.
    Yuzey output/shapes/<subject>/brain_surface.vtk olarak kaydedilir.
    Slicer'da yari seffaf gosterilir — nuclei iceriden gorunur.
    """
    atlas_path = os.path.join(DATA_DIR, subject, "warped", "atlas_warped.nii.gz")
    if not os.path.exists(atlas_path):
        print(f"    UYARI: atlas_warped bulunamadi: {atlas_path}")
        return

    img  = nib.load(atlas_path)
    img  = nib.as_closest_canonical(img)
    data = img.get_fdata(dtype=np.float32)

    # Beyin maskesi: Otsu esigi ile beyaz madde + gri maddeyi yakala
    nz = data[data > 0]
    try:
        from skimage.filters import threshold_otsu
        thr = threshold_otsu(nz) * 0.35   # dusuk esik = beyin + meninks
    except Exception:
        thr = float(np.percentile(nz, 25))

    brain_mask = (data > thr).astype(bool)

    # En buyuk bileşeni al (kafa)
    from scipy.ndimage import binary_fill_holes, label as ndi_label, gaussian_filter, binary_erosion
    labeled, n = ndi_label(brain_mask)
    if n > 1:
        sizes  = [np.sum(labeled == i) for i in range(1, n + 1)]
        brain_mask = (labeled == (np.argmax(sizes) + 1))

    # Delikleri doldur, hafif erozyonla skull strip
    brain_mask = binary_fill_holes(brain_mask)
    brain_mask = binary_erosion(brain_mask, iterations=3)

    # Gaussian ile yumusatma — pürüzsüz yüzey
    smooth = gaussian_filter(brain_mask.astype(float), sigma=2.5)

    try:
        verts_vox, faces, _, _ = marching_cubes(smooth, level=0.5)
    except Exception as e:
        print(f"    UYARI: beyin yuzey marching cubes basarisiz: {e}")
        return

    # Affine ile RAS mm'ye
    ones      = np.ones((len(verts_vox), 1))
    verts_ras = (img.affine @ np.hstack([verts_vox, ones]).T).T[:, :3]

    out_path = os.path.join(out_dir, "brain_surface.vtk")
    write_vtk(verts_ras, faces, out_path, "brain_surface")
    print(f"    BEYIN  brain_surface.vtk  "
          f"{len(verts_ras):6d} nokta  {len(faces):7d} yuzey")


# ─────────────────────────────────────────────────────────────────────────────
# Ana Fonksiyon
# ─────────────────────────────────────────────────────────────────────────────

def main():
    subjects = list_subjects()
    print(f"\n{'='*62}")
    print(f"  3D Shape Export  —  {len(subjects)} subject")
    print(f"{'='*62}")
    print(f"  Cikti: {OUT_DIR}\n")

    total_ok   = 0
    total_skip = 0

    for subj in subjects:
        print(f"  [{subj}]")
        subj_dir = os.path.join(OUT_DIR, subj)
        os.makedirs(subj_dir, exist_ok=True)

        for side, label, fpath in list_labels(subj):
            affine, mask = load_mask(fpath)
            verts, faces = mask_to_mesh(mask, affine)

            out_name = f"{label}_{side}.vtk"
            out_path = os.path.join(subj_dir, out_name)

            if verts is None:
                n = int(mask.sum())
                print(f"    ATLANDI  {side:5s}  {label:<15s}  ({n} voxel < {MIN_VOXELS})")
                total_skip += 1
            else:
                write_vtk(verts, faces, out_path, f"{label}_{side}")
                print(f"    OK       {side:5s}  {label:<15s}  "
                      f"{len(verts):5d} nokta  {len(faces):6d} yuzey")
                total_ok += 1

        # Beyin yuzey mesh'i
        generate_brain_surface(subj, subj_dir)
        print()

    print(f"{'='*62}")
    print(f"  TAMAMLANDI  —  {total_ok} mesh uretildi, {total_skip} atlandi")
    print(f"{'='*62}\n")
    print("  Sonraki adim:")
    print("  3D Slicer'i ac ve Python konsolunda su komutu calistir:")
    print("  exec(open(r'scripts/slicer_load_scene.py').read())\n")


if __name__ == "__main__":
    main()
