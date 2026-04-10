"""
İP-1: Veri Hazırlık ve Standardizasyon
=======================================
- Orientation kontrolü → RAS
- Voxel spacing kontrolü → 1mm iso resample
- N4 bias correction (T1, T2)
- Robust intensity normalization
- Brain mask üretimi (basit eşikleme; HD-BET varsa onu çağır)
- T2 → T1 rigid registration
- MRA → T1 rigid registration
- QC görselleri + JSON raporu

Kullanım:
    python scripts/ip1_preproc.py                      # tüm hastalar
    python scripts/ip1_preproc.py --subject IXI002-Guys-0828
    python scripts/ip1_preproc.py --subject IXI002-Guys-0828 --skip-n4
"""

import argparse
import os
import sys
import time

# Windows terminal encoding fix
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(_sys.stderr, "reconfigure"):
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import ants

# Proje kök dizinini Python path'ine ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import (
    SUBJECTS, PATIENT_DIR, ATLAS_T1, OUTPUT_DIR, TARGET_SPACING, THRESHOLDS, QC_DPI
)
from scripts.utils.reporter import StepReporter
from scripts.utils.nifti_utils import (
    load_nifti, save_nifti, get_voxel_volume,
    check_orientation, reorient_to_ras, resample_to_spacing,
    n4_bias_correction, normalize_intensity, compute_brain_mask_simple, compute_nmi,
    nib_to_ants,
)


# ──────────────────────────────────────────────────────────────────────────────
# Ana preprocessing fonksiyonu
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_subject(subject_id: str, skip_n4: bool = False) -> dict:
    """
    Tek bir hastanın tüm İP-1 adımlarını yürüt.
    Döndürür: rapor dict'i
    """
    out_dir = os.path.join(OUTPUT_DIR, subject_id, "preproc")
    qc_dir  = os.path.join(out_dir, "QC")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(qc_dir,  exist_ok=True)

    rep = StepReporter(subject_id, "IP1_Preproc", out_dir)

    try:
        # ── Girdi yolları ──────────────────────────────────────────────────
        t1_path  = os.path.join(PATIENT_DIR, f"{subject_id}-T1.nii.gz")
        t2_path  = os.path.join(PATIENT_DIR, f"{subject_id}-T2.nii.gz")
        mra_path = os.path.join(PATIENT_DIR, f"{subject_id}-MRA.nii.gz")

        for p in [t1_path, t2_path, mra_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Girdi bulunamadı: {p}")

        # ── T1 yükle ──────────────────────────────────────────────────────
        print(f"  → T1 yükleniyor: {t1_path}")
        t1_img = load_nifti(t1_path)

        # Orientation kontrolü
        orient = check_orientation(t1_img)
        rep.log("T1_orientation_original", orient)
        t1_img = reorient_to_ras(t1_img)
        rep.log("T1_orientation_after", check_orientation(t1_img))

        # Voxel spacing kontrolü
        zooms = t1_img.header.get_zooms()[:3]
        rep.log("T1_voxel_spacing_original_mm",
                [round(float(z), 3) for z in zooms])

        if not np.allclose(zooms, TARGET_SPACING, atol=0.01):
            print(f"  → Resample: {zooms} → {TARGET_SPACING}")
            t1_img = resample_to_spacing(t1_img, TARGET_SPACING, "linear")
        rep.log("T1_shape", list(t1_img.shape))
        rep.log("T1_voxel_spacing_mm", list(TARGET_SPACING))

        # ── N4 Bias Correction — T1 ────────────────────────────────────────
        if not skip_n4:
            print("  → N4 bias correction (T1)...")
            t1_img, n4_iters = n4_bias_correction(t1_img)
            rep.log("N4_T1_iterations", n4_iters)
        else:
            print("  → N4 atlandı (--skip-n4)")

        # ── Brain Mask ─────────────────────────────────────────────────────
        print("  → Brain mask üretiliyor...")
        brain_mask_img = _run_brain_extraction(t1_img)
        mask_vol = float(brain_mask_img.get_fdata().sum()) * get_voxel_volume(t1_img)
        rep.log("brain_mask_volume_mm3", round(mask_vol, 0),
                warn_if=lambda v: v < 800_000 or v > 2_200_000)

        # ── Intensity Normalization — T1 ───────────────────────────────────
        t1_norm = normalize_intensity(t1_img, mask=brain_mask_img)

        # ── T2 yükle & işle ────────────────────────────────────────────────
        t2_img  = load_nifti(t2_path)
        t2_img  = reorient_to_ras(t2_img)
        t2_img  = resample_to_spacing(t2_img, TARGET_SPACING, "linear")
        if not skip_n4:
            t2_img, _ = n4_bias_correction(t2_img)
        # T2 kendi space'inde normalize edilir; mask T1 space'inde olduğu için kullanılmaz
        t2_norm = normalize_intensity(t2_img)

        # T2 → T1 rigid registration
        print("  → T2 → T1 rigid registration...")
        t2_to_t1_img, t2_nmi = _register_to_t1(t1_norm, t2_norm, metric="mattes")
        rep.log("T2_to_T1_NMI", round(t2_nmi, 4),
                warn_if=lambda v: v < 0.5)

        # ── MRA yükle & işle ───────────────────────────────────────────────
        mra_img  = load_nifti(mra_path)
        mra_img  = reorient_to_ras(mra_img)
        mra_img  = resample_to_spacing(mra_img, TARGET_SPACING, "linear")
        mra_norm = normalize_intensity(mra_img)

        # MRA → T1 rigid registration
        print("  → MRA → T1 rigid registration...")
        mra_to_t1_img, mra_nmi = _register_to_t1(t1_norm, mra_norm, metric="mattes")
        rep.log("MRA_to_T1_NMI", round(mra_nmi, 4),
                warn_if=lambda v: v < 0.4)

        # ── QC flag'leri ───────────────────────────────────────────────────
        rep.set_flag("gross_misalignment",      t2_nmi < 0.4,
                     "T2→T1 NMI çok düşük")
        rep.set_flag("skull_strip_loss_warning", mask_vol < 800_000,
                     "Beyin maskesi çok küçük")

        # ── Çıktıları kaydet ───────────────────────────────────────────────
        paths = {
            "T1_preproc":       os.path.join(out_dir, "T1_preproc.nii.gz"),
            "T2_to_T1_preproc": os.path.join(out_dir, "T2_to_T1_preproc.nii.gz"),
            "MRA_to_T1_preproc":os.path.join(out_dir, "MRA_to_T1_preproc.nii.gz"),
            "brain_mask":       os.path.join(out_dir, "brain_mask.nii.gz"),
        }
        save_nifti(t1_norm.get_fdata(),    t1_norm.affine,    paths["T1_preproc"])
        save_nifti(t2_to_t1_img.get_fdata(), t2_to_t1_img.affine, paths["T2_to_T1_preproc"])
        save_nifti(mra_to_t1_img.get_fdata(), mra_to_t1_img.affine, paths["MRA_to_T1_preproc"])
        save_nifti(brain_mask_img.get_fdata().astype(np.uint8),
                   brain_mask_img.affine, paths["brain_mask"])

        for key, path in paths.items():
            rep.add_file(key, path)

        # ── QC Görselleri ──────────────────────────────────────────────────
        _qc_overlay_two(
            t1_norm.get_fdata(), t2_to_t1_img.get_fdata(),
            brain_mask_img.get_fdata(),
            title=f"{subject_id} — T1 vs T2",
            path=os.path.join(qc_dir, "overlay_T1_T2.png"),
        )
        _qc_overlay_two(
            t1_norm.get_fdata(), mra_to_t1_img.get_fdata(),
            brain_mask_img.get_fdata(),
            title=f"{subject_id} — T1 vs MRA",
            path=os.path.join(qc_dir, "overlay_T1_MRA.png"),
        )
        _qc_histogram(
            t1_norm.get_fdata(), t2_to_t1_img.get_fdata(),
            brain_mask_img.get_fdata(),
            path=os.path.join(qc_dir, "histogram_report.png"),
        )
        rep.add_file("QC_overlay_T1_T2",  os.path.join(qc_dir, "overlay_T1_T2.png"))
        rep.add_file("QC_overlay_T1_MRA", os.path.join(qc_dir, "overlay_T1_MRA.png"))
        rep.add_file("QC_histogram",      os.path.join(qc_dir, "histogram_report.png"))

    except Exception as exc:
        rep.record_exception(exc)
        return rep.finish("FAILED")

    return rep.finish("SUCCESS")


# ──────────────────────────────────────────────────────────────────────────────
# Yardımcı fonksiyonlar
# ──────────────────────────────────────────────────────────────────────────────

def _run_brain_extraction(t1_img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    ANTsPyNet derin ogrenme tabanli skull stripping.
    Kurulu degilse basit esikleme kullan.
    """
    try:
        import antspynet
        t1_ants = nib_to_ants(t1_img)
        prob    = antspynet.brain_extraction(t1_ants, modality="t1", verbose=False)
        mask_ants = ants.threshold_image(prob, 0.5, 1.0, 1, 0)
        # Kucuk izole parcalari temizle
        from scipy import ndimage as _ndi
        mask_data = mask_ants.numpy().astype(np.uint8)
        labeled, n_cc = _ndi.label(mask_data)
        if n_cc > 1:
            sizes = np.bincount(labeled.ravel()); sizes[0] = 0
            mask_data = (labeled == sizes.argmax()).astype(np.uint8)
        print(f"  [ANTsPyNet] Brain mask: {mask_data.sum()/1e6:.2f}M mm3")
        return nib.Nifti1Image(mask_data, t1_img.affine, t1_img.header)
    except Exception as e:
        print(f"  [ANTsPyNet] Kullanilamiyor ({e}) -> basit esikleme")
        return compute_brain_mask_simple(t1_img, threshold_percentile=15.0)


def _ants_to_nib(ants_img, ref_nib: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    ANTsPy registration çıktısını nibabel'e dönüştür.
    Affine olarak referans görüntünün (fixed) affine'i kullanılır —
    bu registration çıktıları için doğru davranıştır.
    nifti_utils.ants_to_nib()'den farklıdır: o kendi metadata'sını kullanır.
    """
    return nib.Nifti1Image(ants_img.numpy(), ref_nib.affine, ref_nib.header)


def _register_to_t1(
    fixed_img: nib.Nifti1Image,
    moving_img: nib.Nifti1Image,
    metric: str = "mattes",
) -> tuple:
    """
    ANTsPy ile rigid registration; (registered_nib_img, NMI_score) döndür.
    """
    fixed_ants  = nib_to_ants(fixed_img)
    moving_ants = nib_to_ants(moving_img)

    tx = ants.registration(
        fixed             = fixed_ants,
        moving            = moving_ants,
        type_of_transform = "Rigid",
        aff_metric        = metric,
        verbose           = False,
    )
    warped_ants = tx["warpedmovout"]
    warped_nib  = _ants_to_nib(warped_ants, fixed_img)

    from scripts.utils.nifti_utils import compute_nmi
    nmi = compute_nmi(fixed_img, warped_nib)

    return warped_nib, nmi


# ──────────────────────────────────────────────────────────────────────────────
# QC Görsel fonksiyonları
# ──────────────────────────────────────────────────────────────────────────────

def _qc_overlay_two(
    data1: np.ndarray, data2: np.ndarray, mask: np.ndarray,
    title: str, path: str, n_slices: int = 3
):
    """İki modaliteyi yan yana göster (axial orta dilim ± n_slices)."""
    z_center = data1.shape[2] // 2
    z_indices = np.linspace(z_center - 20, z_center + 20, n_slices, dtype=int)
    z_indices = np.clip(z_indices, 0, data1.shape[2] - 1)

    fig, axes = plt.subplots(2, n_slices, figsize=(4 * n_slices, 8))
    fig.suptitle(title, fontsize=12)

    for i, z in enumerate(z_indices):
        sl1 = data1[:, :, z]
        sl2 = data2[:, :, z]
        axes[0, i].imshow(sl1.T, cmap="gray", origin="lower",
                          vmin=np.percentile(sl1, 1), vmax=np.percentile(sl1, 99))
        axes[0, i].set_title(f"T1 z={z}")
        axes[0, i].axis("off")
        axes[1, i].imshow(sl2.T, cmap="gray", origin="lower",
                          vmin=np.percentile(sl2, 1), vmax=np.percentile(sl2, 99))
        axes[1, i].set_title(f"Other z={z}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=QC_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  → QC görsel kaydedildi: {path}")


def _qc_histogram(
    t1: np.ndarray, t2: np.ndarray, mask: np.ndarray, path: str
):
    """T1 ve T2 intensite histogramlarını kaydet."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Intensite Histogramları (beyin maskesi içi)", fontsize=11)

    m = mask > 0
    for ax, data, label in zip(axes, [t1, t2], ["T1", "T2"]):
        roi = data[m]
        ax.hist(roi, bins=128, color="steelblue", alpha=0.8)
        ax.set_title(f"{label} — mean={roi.mean():.2f}, std={roi.std():.2f}")
        ax.set_xlabel("Intensite")
        ax.set_ylabel("Voksel sayısı")

    plt.tight_layout()
    plt.savefig(path, dpi=QC_DPI, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="İP-1 Preprocessing Pipeline")
    parser.add_argument("--subject", type=str, default=None,
                        help="Tek hasta ID'si; belirtilmezse tüm hastalar çalışır.")
    parser.add_argument("--skip-n4", action="store_true",
                        help="N4 bias correction adımını atla (hızlı test için).")
    args = parser.parse_args()

    subjects = [args.subject] if args.subject else SUBJECTS

    print(f"\n{'='*60}")
    print(f"  İP-1 Preprocessing — {len(subjects)} hasta")
    print(f"{'='*60}")

    results = []
    for subj in subjects:
        report = preprocess_subject(subj, skip_n4=args.skip_n4)
        results.append(report)

    # Özet
    ok   = sum(1 for r in results if r["status"] == "SUCCESS")
    fail = len(results) - ok
    print(f"\n{'='*60}")
    print(f"  İP-1 TAMAMLANDI — ✓ {ok} başarılı, ✗ {fail} başarısız")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
