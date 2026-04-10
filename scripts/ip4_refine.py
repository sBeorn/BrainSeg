"""
İP-4: Refine ve Damar İlişkisi
================================
Propagated label'ları T1/T2 intensite bilgisiyle iyileştirir
ve MRA'dan türetilen damar maskesiyle ilişki analizi yapar.

3 refine adayı:
  R1 — Gaussian intensity-based local boundary refinement
  R2 — Active contour (snake) T1+T2 joint
  R3 — Morphological correction only (baseline)

Her aday için:
  - Boundary score (pre vs post)
  - Volume drift oranı
  - Shape regularity değişimi (compactness delta)
  - Vessel overlap kontrolü
  - Vessel-label minimum mesafe
  - vessel_label_distances.csv
  - JSON raporu + QC görselleri

Kullanım:
    python scripts/ip4_refine.py
    python scripts/ip4_refine.py --subject IXI002-Guys-0828 --warp W1 --candidate R1
"""

import argparse
import csv
import json
import os
import sys

# Windows terminal encoding fix
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(_sys.stderr, "reconfigure"):
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import (
    SUBJECTS, OUTPUT_DIR, WARP_CANDIDATES, REFINE_CANDIDATES, THRESHOLDS, QC_DPI
)
from scripts.utils.reporter import StepReporter
from scripts.utils.nifti_utils import load_nifti, save_nifti, get_voxel_volume
from scripts.utils.metrics import compute_geometric_morphometrics


# ──────────────────────────────────────────────────────────────────────────────

def refine_subject(
    subject_id: str,
    warp_candidate: str,
    refine_candidate: str,
) -> dict:
    preproc_dir  = os.path.join(OUTPUT_DIR, subject_id, "preproc")
    prop_dir     = os.path.join(OUTPUT_DIR, subject_id, "propagate", warp_candidate)
    out_dir      = os.path.join(OUTPUT_DIR, subject_id, "refine", refine_candidate)
    ref_left     = os.path.join(out_dir, "refined_left_labels")
    ref_right    = os.path.join(out_dir, "refined_right_labels")
    os.makedirs(ref_left,  exist_ok=True)
    os.makedirs(ref_right, exist_ok=True)

    step_name = f"IP4_Refine_{warp_candidate}_{refine_candidate}"
    rep = StepReporter(subject_id, step_name, out_dir)

    try:
        # ── Girdi yolları ──────────────────────────────────────────────────
        t1_path   = os.path.join(preproc_dir, "T1_preproc.nii.gz")
        t2_path   = os.path.join(preproc_dir, "T2_to_T1_preproc.nii.gz")
        mra_path  = os.path.join(preproc_dir, "MRA_to_T1_preproc.nii.gz")
        mask_path = os.path.join(preproc_dir, "brain_mask.nii.gz")
        prop_left = os.path.join(prop_dir, "left_labels")
        prop_right= os.path.join(prop_dir, "right_labels")

        for p in [t1_path, mask_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Girdi yok: {p}")
        if not os.path.exists(prop_left):
            raise FileNotFoundError(f"Propagate çıktısı yok: {prop_left}")

        t1_nib   = load_nifti(t1_path)
        t2_nib   = load_nifti(t2_path) if os.path.exists(t2_path) else None
        mra_nib  = load_nifti(mra_path) if os.path.exists(mra_path) else None
        mask_nib = load_nifti(mask_path)

        t1_data  = t1_nib.get_fdata()
        t2_data  = t2_nib.get_fdata() if t2_nib else None
        mra_data = mra_nib.get_fdata() if mra_nib else None
        spacing  = tuple(t1_nib.header.get_zooms()[:3])
        voxel_vol= float(np.prod(spacing))

        # ── Damar Maskesi (MRA Frangi vesselness) ─────────────────────────
        print("  → Damar maskesi (Frangi vesselness)...")
        vessel_mask, vessel_skeleton = _compute_vessel_mask(mra_data, spacing)

        v_path  = os.path.join(out_dir, "vessel_mask.nii.gz")
        sk_path = os.path.join(out_dir, "vessel_skeleton.nii.gz")
        save_nifti(vessel_mask.astype(np.uint8),    t1_nib.affine, v_path)
        save_nifti(vessel_skeleton.astype(np.uint8), t1_nib.affine, sk_path)
        rep.add_file("vessel_mask",     v_path)
        rep.add_file("vessel_skeleton", sk_path)
        rep.log("vessel_mask_volume_mm3", round(float(vessel_mask.sum()) * voxel_vol, 1))

        # ── Label dosyalarını bul ──────────────────────────────────────────
        label_names = [
            os.path.splitext(os.path.splitext(f)[0])[0]
            for f in os.listdir(prop_left) if f.endswith(".nii.gz")
        ]

        cfg = REFINE_CANDIDATES[refine_candidate]
        rep.log("refine_method", cfg["description"])

        # ── Refine döngüsü ─────────────────────────────────────────────────
        vessel_distances = []
        refine_metrics   = {}

        for label_name in sorted(label_names):
            lp = os.path.join(prop_left,  f"{label_name}.nii.gz")
            rp = os.path.join(prop_right, f"{label_name}.nii.gz")

            for side, src_path, dst_dir in [
                ("left",  lp, ref_left),
                ("right", rp, ref_right),
            ]:
                if not os.path.exists(src_path):
                    continue

                orig_arr = load_nifti(src_path).get_fdata()

                # Refine
                refined_arr = _apply_refine(
                    orig_arr, t1_data, t2_data, spacing, cfg
                )

                # Boundary score
                bs_pre  = _boundary_score(orig_arr,    t1_data)
                bs_post = _boundary_score(refined_arr, t1_data)

                # Volume drift
                vol_pre  = float(orig_arr.sum())    * voxel_vol
                vol_post = float(refined_arr.sum()) * voxel_vol
                drift    = abs(vol_post - vol_pre) / (vol_pre + 1e-10)

                # Compactness delta
                g_pre  = compute_geometric_morphometrics(orig_arr,    spacing)
                g_post = compute_geometric_morphometrics(refined_arr, spacing)
                comp_delta = g_post["compactness"] - g_pre["compactness"]

                # Vessel overlap
                v_overlap = int(((refined_arr > 0) & (vessel_mask > 0)).sum()) \
                    if vessel_mask is not None else 0

                # Vessel-label minimum mesafe (EDT)
                if vessel_mask is not None and refined_arr.sum() > 0:
                    edt   = ndimage.distance_transform_edt(
                        1 - (vessel_mask > 0), sampling=spacing
                    )
                    dists = edt[refined_arr > 0]
                    min_d  = round(float(dists.min()), 2)
                    mean_d = round(float(dists.mean()), 2)
                    max_d  = round(float(dists.max()), 2)
                else:
                    min_d = mean_d = max_d = -1.0

                vessel_distances.append({
                    "label":       label_name,
                    "side":        side,
                    "min_dist_mm": min_d,
                    "mean_dist_mm":mean_d,
                    "max_dist_mm": max_d,
                    "vessel_overlap_voxels": v_overlap,
                })

                refine_metrics[f"{label_name}_{side}"] = {
                    "boundary_score_pre":  round(bs_pre,   4),
                    "boundary_score_post": round(bs_post,  4),
                    "boundary_improvement": round(bs_post - bs_pre, 4),
                    "volume_pre_mm3":  round(vol_pre,  2),
                    "volume_post_mm3": round(vol_post, 2),
                    "volume_drift":    round(float(drift), 4),
                    "compactness_delta": round(float(comp_delta), 4),
                    "vessel_overlap_voxels": v_overlap,
                    "vessel_min_dist_mm":    min_d,
                }

                # Uyarı / fail kontrolleri
                if drift > THRESHOLDS["volume_drift_fail"]:
                    rep.set_flag(
                        f"volume_drift_warn_{label_name}_{side}", True,
                        f"Volume drift = {drift:.1%}"
                    )
                if v_overlap > THRESHOLDS["vessel_overlap_fail"]:
                    rep.set_flag(
                        f"vessel_overlap_fail_{label_name}_{side}", True,
                        f"Label damar içine girdi ({v_overlap} voxel)"
                    )

                # Kaydet
                dst_path = os.path.join(dst_dir, f"{label_name}.nii.gz")
                save_nifti(refined_arr.astype(np.uint8), t1_nib.affine, dst_path)

        # ── vessel_label_distances.csv ─────────────────────────────────────
        csv_path = os.path.join(out_dir, "vessel_label_distances.csv")
        if vessel_distances:
            keys = list(vessel_distances[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(vessel_distances)
        rep.add_file("vessel_label_distances_csv", csv_path)

        # ── QC görseli ────────────────────────────────────────────────────
        _qc_vessel_label_overlay(
            t1_data, vessel_mask,
            load_nifti(os.path.join(ref_left, f"{label_names[0]}.nii.gz")).get_fdata()
            if label_names else np.zeros_like(t1_data),
            title=f"{subject_id} — {refine_candidate} | Vessel + Label",
            path=os.path.join(out_dir, "vessel_label_overlay.png"),
        )
        rep.add_file("QC_vessel_label_overlay",
                     os.path.join(out_dir, "vessel_label_overlay.png"))

        rep.add_metric_group("refine_metrics", refine_metrics)
        rep.add_file("refined_left_labels",  ref_left)
        rep.add_file("refined_right_labels", ref_right)

    except Exception as exc:
        rep.record_exception(exc)
        return rep.finish("FAILED")

    return rep.finish("SUCCESS")


# ──────────────────────────────────────────────────────────────────────────────
# Refine Yöntemleri
# ──────────────────────────────────────────────────────────────────────────────

def _apply_refine(
    mask: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    spacing: tuple,
    cfg: dict,
) -> np.ndarray:
    method = cfg["method"]

    if method == "gaussian_boundary":
        return _refine_gaussian(mask, t1, cfg)

    elif method == "active_contour":
        return _refine_active_contour(mask, t1, t2, cfg)

    elif method == "morphological":
        return _refine_morphological(mask, cfg)

    else:
        print(f"  ⚠  Bilinmeyen refine yöntemi: {method} → orijinal döndürülüyor")
        return mask.copy()


def _refine_gaussian(mask: np.ndarray, t1: np.ndarray, cfg: dict) -> np.ndarray:
    """
    R1: Gaussian smoothing ile intensite-tabanlı sınır iyileştirme.
    Maske içi yoğunluk ortalaması hesaplanır; bölgeyi iteratif genişlet/daralt.
    """
    sigma  = cfg.get("sigma", 1.0)
    n_iter = cfg.get("n_iter", 5)

    binary = (mask > 0).astype(float)
    roi    = t1[mask > 0]
    if len(roi) == 0:
        return mask.astype(np.uint8)

    # Minimum voksel sayisi kontrolu: cok kucuk label'larda
    # intensite istatistigi guvenilmez, orijinal dondur
    MIN_VOXELS = 50
    if len(roi) < MIN_VOXELS:
        return mask.astype(np.uint8)

    mu_in  = roi.mean()
    std_in = roi.std() + 1e-10

    # Gaussian smooth ile olasılık haritası
    smooth = ndimage.gaussian_filter(binary, sigma=sigma)
    prob   = np.exp(-((t1 - mu_in) ** 2) / (2 * std_in ** 2))

    # Birleşik skor → threshold
    combined = smooth * 0.5 + prob * 0.5
    threshold = combined[mask > 0].mean() * 0.8

    refined = (combined > threshold).astype(np.uint8)

    # Morfolojik temizleme
    struct  = ndimage.generate_binary_structure(3, 1)
    refined = ndimage.binary_closing(refined, structure=struct, iterations=2)
    refined = ndimage.binary_fill_holes(refined).astype(np.uint8)

    return refined


def _refine_active_contour(
    mask: np.ndarray, t1: np.ndarray, t2: np.ndarray, cfg: dict
) -> np.ndarray:
    """
    R2: Basit active contour (snake) yaklaşımı — T1+T2 joint.
    Gerçek graph-cut için maxflow/pygco gerekir; burada iteratif erozyon/dilasyon
    tabanlı yaklaşım uygulanır.
    """
    alpha  = cfg.get("alpha",  0.015)
    sigma  = cfg.get("sigma",  3.0)
    n_iter = cfg.get("n_iter", 200)

    binary = (mask > 0).astype(float)
    roi    = t1[mask > 0]
    if len(roi) == 0:
        return mask.astype(np.uint8)

    # Minimum voksel sayisi kontrolu
    MIN_VOXELS = 50
    if len(roi) < MIN_VOXELS:
        return mask.astype(np.uint8)

    mu_t1  = roi.mean()
    std_t1 = roi.std() + 1e-10

    if t2 is not None:
        roi_t2 = t2[mask > 0]
        mu_t2  = roi_t2.mean()
        std_t2 = roi_t2.std() + 1e-10
    else:
        mu_t2 = std_t2 = None

    # Edge map
    from scipy.ndimage import gaussian_filter
    smooth_t1 = gaussian_filter(t1, sigma=sigma)
    gx, gy, gz = np.gradient(smooth_t1)
    edge_map = np.sqrt(gx**2 + gy**2 + gz**2)
    edge_map = edge_map / (edge_map.max() + 1e-10)

    # Iteratif güncelleme (simplified level-set spirit)
    phi = binary.copy()
    struct = ndimage.generate_binary_structure(3, 1)
    for _ in range(min(n_iter, 50)):   # 50 iterasyon yeterli
        # Data term: T1 intensite uyumu
        data_t1 = np.exp(-((t1 - mu_t1) ** 2) / (2 * std_t1 ** 2))
        if t2 is not None and std_t2:
            data_t2 = np.exp(-((t2 - mu_t2) ** 2) / (2 * std_t2 ** 2))
            data = 0.5 * data_t1 + 0.5 * data_t2
        else:
            data = data_t1

        # Yeni phi
        phi_new = gaussian_filter(phi, sigma=0.5) * (1 - alpha * edge_map) + alpha * data
        phi = (phi_new > 0.5).astype(float)

    refined = ndimage.binary_fill_holes(phi > 0).astype(np.uint8)
    return refined


def _refine_morphological(mask: np.ndarray, cfg: dict) -> np.ndarray:
    """R3: Sadece morfolojik işlemler (baseline)."""
    op     = cfg.get("morph_op", "closing")
    radius = cfg.get("radius", 1)
    struct = ndimage.generate_binary_structure(3, 1)

    binary = (mask > 0)
    if op == "closing":
        refined = ndimage.binary_closing(binary, structure=struct, iterations=radius)
    elif op == "opening":
        refined = ndimage.binary_opening(binary, structure=struct, iterations=radius)
    elif op == "dilation":
        refined = ndimage.binary_dilation(binary, structure=struct, iterations=radius)
    else:
        refined = binary

    refined = ndimage.binary_fill_holes(refined)
    return refined.astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Vesselness
# ──────────────────────────────────────────────────────────────────────────────

def _compute_vessel_mask(
    mra: np.ndarray, spacing: tuple, threshold: float = 0.1
) -> tuple:
    """
    MRA görüntüsünden Frangi vesselness filtresi ile damar maskesi türet.
    """
    if mra is None:
        dummy = np.zeros((10, 10, 10), dtype=np.uint8)
        return dummy, dummy

    try:
        from skimage.filters import frangi
        # 2D dilim bazlı Frangi (3D Frangi için scikit-image ≥ 0.19 gerekir)
        vessel_prob = np.zeros_like(mra, dtype=float)
        for z in range(mra.shape[2]):
            sl = mra[:, :, z].astype(float)
            if sl.max() > sl.min():
                vessel_prob[:, :, z] = frangi(
                    sl / (sl.max() + 1e-10),
                    sigmas=range(1, 4),
                    alpha=0.5, beta=0.5, gamma=15,
                    black_ridges=False,
                )
    except Exception as e:
        print(f"  ⚠  Frangi filtresi başarısız ({e}); eşikleme yöntemi kullanılıyor.")
        p95 = np.percentile(mra[mra > 0], 95)
        vessel_prob = (mra > p95).astype(float)

    vessel_mask = (vessel_prob > threshold).astype(np.uint8)

    # Morfolojik temizleme
    struct = ndimage.generate_binary_structure(3, 2)
    vessel_mask = ndimage.binary_opening(vessel_mask, structure=struct, iterations=1)
    vessel_mask = vessel_mask.astype(np.uint8)

    # İskelet
    try:
        from skimage.morphology import skeletonize_3d
        vessel_skeleton = skeletonize_3d(vessel_mask).astype(np.uint8)
    except Exception:
        vessel_skeleton = np.zeros_like(vessel_mask)

    return vessel_mask, vessel_skeleton


# ──────────────────────────────────────────────────────────────────────────────
# Sınır Skoru
# ──────────────────────────────────────────────────────────────────────────────

def _boundary_score(mask: np.ndarray, t1: np.ndarray) -> float:
    """
    Label sınırındaki ortalama gradient magnitude (sınır keskinliği).
    Yüksek = iyi hizalanmış sınır.
    """
    if mask.sum() == 0:
        return 0.0

    struct = ndimage.generate_binary_structure(3, 1)
    dilated = ndimage.binary_dilation(mask > 0, structure=struct)
    eroded  = ndimage.binary_erosion(mask > 0, structure=struct)
    boundary = dilated ^ eroded  # symmetric difference → shell

    gx, gy, gz = np.gradient(t1.astype(float))
    grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)

    if boundary.sum() == 0:
        return 0.0
    return float(grad_mag[boundary].mean())


# ──────────────────────────────────────────────────────────────────────────────
# QC
# ──────────────────────────────────────────────────────────────────────────────

def _qc_vessel_label_overlay(
    t1: np.ndarray, vessel: np.ndarray, label: np.ndarray,
    title: str, path: str, n: int = 3
):
    z_indices = np.linspace(t1.shape[2] // 4, 3 * t1.shape[2] // 4, n, dtype=int)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.suptitle(title, fontsize=10)
    for i, z in enumerate(z_indices):
        sl_t1  = t1[:, :, z]
        axes[i].imshow(sl_t1.T, cmap="gray", origin="lower",
                       vmin=np.percentile(sl_t1, 1), vmax=np.percentile(sl_t1, 99))
        if vessel is not None and vessel[:, :, z].sum() > 0:
            axes[i].contour(vessel[:, :, z].T, levels=[0.5],
                            colors=["cyan"], linewidths=0.8)
        if label.sum() > 0 and label[:, :, z].sum() > 0:
            axes[i].contour(label[:, :, z].T, levels=[0.5],
                            colors=["red"], linewidths=1.2)
        axes[i].set_title(f"z={z}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=QC_DPI, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="İP-4 Refine + Vessel")
    parser.add_argument("--subject",   type=str, default=None)
    parser.add_argument("--warp",      type=str, default=None,
                        choices=list(WARP_CANDIDATES.keys()))
    parser.add_argument("--candidate", type=str, default=None,
                        choices=list(REFINE_CANDIDATES.keys()))
    args = parser.parse_args()

    subjects    = [args.subject]   if args.subject   else SUBJECTS
    warps       = [args.warp]      if args.warp      else list(WARP_CANDIDATES.keys())
    refines     = [args.candidate] if args.candidate else list(REFINE_CANDIDATES.keys())

    print(f"\n{'='*60}")
    print(f"  İP-4 Refine — {len(subjects)} hasta × {len(warps)} warp × {len(refines)} refine")
    print(f"{'='*60}")

    results = []
    for subj in subjects:
        for w in warps:
            for r in refines:
                rep = refine_subject(subj, w, r)
                results.append(rep)

    ok   = sum(1 for r in results if r["status"] == "SUCCESS")
    fail = len(results) - ok
    print(f"\n{'='*60}")
    print(f"  İP-4 TAMAMLANDI — ✓ {ok} başarılı, ✗ {fail} hata")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
