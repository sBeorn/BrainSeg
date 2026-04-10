"""
İP-2: Warp Keşfi
================
Atlas T1 → Subject T1 nonlineer registration; 3 aday: W1, W2, W3.

Her aday için:
- ANTsPy SyN registration
- Jacobian determinant haritası + negatif voksel oranı
- Inverse consistency hatası (atlas→subj→atlas roundtrip)
- NMI / CC similarity skoru
- QC overlay + Jacobian haritası görseli
- JSON raporu

Kullanım:
    python scripts/ip2_warp.py
    python scripts/ip2_warp.py --subject IXI002-Guys-0828 --candidate W1
"""

import argparse
import json
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
import numpy as np
import nibabel as nib
import ants

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import (
    SUBJECTS, OUTPUT_DIR, ATLAS_T1, WARP_CANDIDATES, THRESHOLDS, QC_DPI
)
from scripts.utils.reporter import StepReporter
from scripts.utils.nifti_utils import load_nifti, save_nifti, compute_nmi, ants_to_nib, win_path


# ──────────────────────────────────────────────────────────────────────────────

def warp_subject_candidate(
    subject_id: str, candidate_name: str
) -> dict:
    """
    Tek hasta + tek warp adayı için registration yürüt.
    """
    preproc_dir = os.path.join(OUTPUT_DIR, subject_id, "preproc")
    out_dir     = os.path.join(OUTPUT_DIR, subject_id, "warp", candidate_name)
    os.makedirs(out_dir, exist_ok=True)

    rep = StepReporter(subject_id, f"IP2_Warp_{candidate_name}", out_dir)

    try:
        # ── Girdi dosyaları ────────────────────────────────────────────────
        t1_path      = os.path.join(preproc_dir, "T1_preproc.nii.gz")
        mask_path    = os.path.join(preproc_dir, "brain_mask.nii.gz")
        atlas_preproc = _get_atlas_preproc()

        for p in [t1_path, mask_path, atlas_preproc]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Girdi yok: {p}")

        fixed_ants  = ants.image_read(t1_path)
        moving_ants = ants.image_read(atlas_preproc)
        mask_ants   = ants.image_read(mask_path)

        cfg = WARP_CANDIDATES[candidate_name]
        rep.log("warp_config_description", cfg["description"])

        # ── Registration ───────────────────────────────────────────────────
        print(f"  → ANTsPy registration ({candidate_name})...")
        t0 = time.time()

        tx = ants.registration(
            fixed              = fixed_ants,
            moving             = moving_ants,
            type_of_transform  = cfg["type_of_transform"],
            aff_metric         = cfg.get("aff_metric", "mattes"),
            syn_metric         = cfg.get("syn_metric", "CC"),
            syn_sampling       = cfg.get("syn_metric_params", [4])[0],
            grad_step          = cfg.get("grad_step", 0.1),
            flow_sigma         = cfg.get("flow_sigma", 3.0),
            total_sigma        = cfg.get("total_sigma", 0.0),
            reg_iterations     = tuple(cfg.get("syn_iterations", [100, 70, 50, 20])),
            outprefix          = win_path(os.path.join(out_dir, "fwd_transform_")),
            verbose            = False,
            mask               = mask_ants,
        )
        elapsed = round(time.time() - t0, 1)
        rep.log("registration_duration_sec", elapsed)

        warped_atlas = tx["warpedmovout"]

        # ── Similarity ─────────────────────────────────────────────────────
        warped_nib = ants_to_nib(warped_atlas)
        fixed_nib  = ants_to_nib(fixed_ants)
        nmi_score  = compute_nmi(fixed_nib, warped_nib)
        rep.log("NMI_post_registration", round(nmi_score, 4),
                warn_if=lambda v: v < 0.7)

        # ── Jacobian Determinant ───────────────────────────────────────────
        print("  → Jacobian determinant hesaplanıyor...")
        # fwdtransforms[0] = 1Warp.nii.gz, [1] = 0GenericAffine.mat
        warp_field_path = next(
            (f for f in tx["fwdtransforms"] if f.endswith(".nii.gz")), None
        )
        if warp_field_path is None:
            raise RuntimeError("Warp displacement field bulunamadı: " + str(tx["fwdtransforms"]))
        jac_ants = ants.create_jacobian_determinant_image(
            domain_image = fixed_ants,
            tx           = warp_field_path,
            do_log       = False,
        )
        jac_arr  = jac_ants.numpy()

        jac_min   = float(jac_arr.min())
        jac_max   = float(jac_arr.max())
        jac_mean  = float(jac_arr.mean())
        neg_ratio = float((jac_arr < 0).sum() / jac_arr.size)

        rep.log("jacobian_min",           round(jac_min, 4))
        rep.log("jacobian_max",           round(jac_max, 4))
        rep.log("jacobian_mean",          round(jac_mean, 4))
        rep.log("jacobian_negative_ratio", round(neg_ratio, 6),
                warn_if=lambda v: v > THRESHOLDS["jacobian_neg_ratio_warn"],
                fail_if=lambda v: v > THRESHOLDS["jacobian_neg_ratio_fail"])

        # ── Inverse Consistency ────────────────────────────────────────────
        print("  → Inverse consistency hatası hesaplanıyor...")
        inv_error_mm = _inverse_consistency(
            fixed_ants, moving_ants, tx["fwdtransforms"], tx["invtransforms"]
        )
        # inv_error: NMI kaybı 0-1 arası; > 0.1 uyarı (yüksek roundtrip kayıp)
        rep.log("inverse_consistency_nmi_loss", round(inv_error_mm, 4),
                warn_if=lambda v: v > 0.10)

        # ── Çıktıları kaydet ───────────────────────────────────────────────
        warped_path = os.path.join(out_dir, "warped_atlas_T1.nii.gz")
        jac_path    = os.path.join(out_dir, "jacobian_det.nii.gz")

        warped_atlas.to_file(warped_path)
        jac_ants.to_file(jac_path)

        # Transform matrislerini kaydet (ants zaten outprefix ile yazdı)
        # similarity_metric.json
        sim_json = {
            "subject_id":                  subject_id,
            "candidate":                   candidate_name,
            "NMI":                         round(nmi_score, 4),
            "jacobian_min":                round(jac_min, 4),
            "jacobian_max":                round(jac_max, 4),
            "jacobian_neg_ratio":          round(neg_ratio, 6),
            "inverse_consistency_nmi_loss": round(inv_error_mm, 4),
            "jacobian_fail":               neg_ratio > THRESHOLDS["jacobian_neg_ratio_fail"],
        }
        with open(os.path.join(out_dir, "similarity_metric.json"), "w") as f:
            json.dump(sim_json, f, indent=2)

        rep.add_file("warped_atlas_T1",   warped_path)
        rep.add_file("jacobian_det",      jac_path)
        rep.add_file("fwd_transforms",    out_dir)

        # ── QC Görselleri ──────────────────────────────────────────────────
        _qc_warp_overlay(
            fixed_nib.get_fdata(), warped_nib.get_fdata(),
            title=f"{subject_id} — {candidate_name} | Warped Atlas vs T1",
            path=os.path.join(out_dir, "QC_overlay.png"),
        )
        _qc_jacobian_map(
            jac_arr, title=f"{subject_id} — {candidate_name} | Jacobian Det.",
            path=os.path.join(out_dir, "jacobian_map_axial.png"),
        )
        rep.add_file("QC_overlay",        os.path.join(out_dir, "QC_overlay.png"))
        rep.add_file("QC_jacobian",       os.path.join(out_dir, "jacobian_map_axial.png"))

    except Exception as exc:
        rep.record_exception(exc)
        return rep.finish("FAILED")

    return rep.finish("SUCCESS")


# ──────────────────────────────────────────────────────────────────────────────
# Atlas preproc — İP-1'den alınan ya da atlasın kendisi
# ──────────────────────────────────────────────────────────────────────────────

def _get_atlas_preproc() -> str:
    """
    Ön işlenmiş atlas T1 yolunu döndür.
    outputs/atlas/atlas_T1_preproc.nii.gz varsa onu kullan,
    yoksa ham ATLAS_T1 kullan.
    """
    preproc_path = os.path.join(OUTPUT_DIR, "atlas", "atlas_T1_preproc.nii.gz")
    if os.path.exists(preproc_path):
        return preproc_path
    if os.path.exists(ATLAS_T1):
        return ATLAS_T1
    raise FileNotFoundError(f"Atlas T1 bulunamadı: {ATLAS_T1}")


# ──────────────────────────────────────────────────────────────────────────────
# Inverse Consistency
# ──────────────────────────────────────────────────────────────────────────────

def _inverse_consistency(
    fixed_ants, moving_ants, fwd_transforms: list, inv_transforms: list
) -> float:
    """
    Atlas → Subject → Atlas roundtrip NMI kaybı (0=mükemmel, 1=tam bozuk).

    NOT: Gerçek displacement-based mm ölçümü için ANTs'ın CreateJacobianDeterminantImage
    veya explicit displacement field analizi gereklidir. Bu metrik, iki görüntü arasındaki
    NMI benzerlik kaybını proxy olarak kullanır; 0'a yakın = iyi round-trip tutarlılık.
    """
    try:
        # Moving → Fixed (forward)
        step1 = ants.apply_transforms(
            fixed=fixed_ants, moving=moving_ants,
            transformlist=fwd_transforms, interpolator="linear",
        )
        # Fixed → Moving (inverse)
        step2 = ants.apply_transforms(
            fixed=moving_ants, moving=step1,
            transformlist=inv_transforms, interpolator="linear",
        )
        # NMI benzerliği: 1.0'a ne kadar yakın = o kadar tutarlı
        # ants.image_similarity: negatif NMI döndürür → abs al
        nmi_roundtrip = abs(ants.image_similarity(
            moving_ants, step2, metric_type="MattesMutualInformation"
        ))
        nmi_original = abs(ants.image_similarity(
            moving_ants, moving_ants, metric_type="MattesMutualInformation"
        ))
        if nmi_original < 1e-6:
            return 0.0
        # Loss = ne kadar NMI kaybedildi (0=kayıp yok, 1=tam kayıp)
        loss = max(0.0, 1.0 - nmi_roundtrip / (nmi_original + 1e-10))
        return round(float(loss), 4)
    except Exception as e:
        print(f"  ⚠  Inverse consistency hesaplanamadı: {e}")
        return -1.0


# ──────────────────────────────────────────────────────────────────────────────
# QC Görselleri
# ──────────────────────────────────────────────────────────────────────────────

def _qc_warp_overlay(
    t1: np.ndarray, warped: np.ndarray, title: str, path: str, n: int = 3
):
    z_indices = np.linspace(t1.shape[2] // 4, 3 * t1.shape[2] // 4, n, dtype=int)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    fig.suptitle(title, fontsize=11)
    for i, z in enumerate(z_indices):
        for row, arr, label in [(0, t1, "T1"), (1, warped, "Warped Atlas")]:
            sl = arr[:, :, z]
            axes[row, i].imshow(sl.T, cmap="gray", origin="lower",
                                vmin=np.percentile(sl, 1), vmax=np.percentile(sl, 99))
            axes[row, i].set_title(f"{label} z={z}")
            axes[row, i].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=QC_DPI, bbox_inches="tight")
    plt.close(fig)


def _qc_jacobian_map(jac: np.ndarray, title: str, path: str, n: int = 3):
    z_indices = np.linspace(jac.shape[2] // 4, 3 * jac.shape[2] // 4, n, dtype=int)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.suptitle(title, fontsize=11)
    vmin, vmax = np.percentile(jac, 2), np.percentile(jac, 98)
    for i, z in enumerate(z_indices):
        im = axes[i].imshow(jac[:, :, z].T, cmap="RdBu_r", origin="lower",
                            vmin=vmin, vmax=vmax)
        axes[i].set_title(f"Jacobian z={z}")
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=QC_DPI, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="İP-2 Warp Keşfi")
    parser.add_argument("--subject",   type=str, default=None)
    parser.add_argument("--candidate", type=str, default=None,
                        choices=list(WARP_CANDIDATES.keys()))
    args = parser.parse_args()

    subjects   = [args.subject]   if args.subject   else SUBJECTS
    candidates = [args.candidate] if args.candidate else list(WARP_CANDIDATES.keys())

    print(f"\n{'='*60}")
    print(f"  İP-2 Warp — {len(subjects)} hasta × {len(candidates)} aday")
    print(f"{'='*60}")

    results = []
    for subj in subjects:
        for cand in candidates:
            r = warp_subject_candidate(subj, cand)
            results.append(r)

    ok   = sum(1 for r in results if r["status"] == "SUCCESS")
    fail = len(results) - ok
    elim = sum(1 for r in results
               if r["qc_flags"].get("jacobian_negative_ratio_fail"))
    print(f"\n{'='*60}")
    print(f"  İP-2 TAMAMLANDI — ✓ {ok} başarılı, ✗ {fail} hata, ⊘ {elim} Jacobian elendi")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
