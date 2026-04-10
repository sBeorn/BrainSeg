"""
Veri Kontrol Scripti
====================
Pipeline başlamadan önce tüm girdi dosyalarının varlığını,
boyutlarını ve orientasyonlarını kontrol eder.

Kullanım:
    python scripts/01_check_data.py
"""

import os
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import (
    SUBJECTS, PATIENT_DIR, ATLAS_T1, LEFT_LABELS, RIGHT_LABELS
)


def check_nifti(path: str, label: str) -> dict:
    result = {"path": path, "label": label, "exists": os.path.exists(path)}
    if not result["exists"]:
        print(f"  ✗ YOK      : {label}")
        return result
    try:
        img = nib.load(path)
        zooms  = img.header.get_zooms()[:3]
        orient = "".join(nib.aff2axcodes(img.affine))
        result.update({
            "shape":   list(img.shape),
            "spacing": [round(float(z), 3) for z in zooms],
            "orient":  orient,
        })
        spacing_ok = np.allclose(zooms, [1.0, 1.0, 1.0], atol=0.1)
        orient_ok  = orient == "RAS"
        sp_flag = "✓" if spacing_ok else "⚠"
        or_flag = "✓" if orient_ok  else "⚠"
        print(
            f"  ✓ MEVCUT   : {label:50s} "
            f"shape={img.shape}  "
            f"spacing={[round(float(z),2) for z in zooms]} {sp_flag}  "
            f"orient={orient} {or_flag}"
        )
    except Exception as e:
        result["error"] = str(e)
        print(f"  ✗ HATA     : {label}  ({e})")
    return result


def main():
    print(f"\n{'='*70}")
    print("  BrainSeg — Veri Kontrol Raporu")
    print(f"{'='*70}\n")

    results = []

    # Atlas T1
    print("[ ATLAS ]")
    results.append(check_nifti(ATLAS_T1, "Atlas T1"))

    # Sol label'lar
    print(f"\n[ SOL LABEL'LAR — {LEFT_LABELS} ]")
    if os.path.isdir(LEFT_LABELS):
        for fname in sorted(os.listdir(LEFT_LABELS)):
            if fname.endswith(".nii.gz"):
                path = os.path.join(LEFT_LABELS, fname)
                results.append(check_nifti(path, f"left/{fname}"))
    else:
        print(f"  ✗ Dizin yok: {LEFT_LABELS}")

    # Sağ label'lar
    print(f"\n[ SAĞ LABEL'LAR — {RIGHT_LABELS} ]")
    if os.path.isdir(RIGHT_LABELS):
        for fname in sorted(os.listdir(RIGHT_LABELS)):
            if fname.endswith(".nii.gz"):
                path = os.path.join(RIGHT_LABELS, fname)
                results.append(check_nifti(path, f"right/{fname}"))
    else:
        print(f"  ✗ Dizin yok: {RIGHT_LABELS}")

    # Hasta verileri
    print(f"\n[ HASTA VERİLERİ — {PATIENT_DIR} ]")
    for subj in SUBJECTS:
        print(f"\n  {subj}:")
        for mod in ["T1", "T2", "MRA"]:
            path = os.path.join(PATIENT_DIR, f"{subj}-{mod}.nii.gz")
            results.append(check_nifti(path, f"  {subj}-{mod}"))

    # Özet
    total  = len(results)
    ok     = sum(1 for r in results if r.get("exists") and "error" not in r)
    missing= sum(1 for r in results if not r.get("exists"))
    err    = sum(1 for r in results if "error" in r)

    print(f"\n{'='*70}")
    print(f"  ÖZET: {total} dosya — ✓ {ok} mevcut, ✗ {missing} eksik, ⚠ {err} hatalı")
    print(f"{'='*70}\n")

    if missing > 0 or err > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
