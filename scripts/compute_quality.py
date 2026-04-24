"""
Warp Kalite ve Dogruluk Analizi — Thalamic Nuclei
===================================================
Atlas label hacimleri ile warped (subject uzayi) hacimleri karsilastirir.
Asimetri, cross-subject tutarlilik ve hacim koruma oranlarini hesaplar.

Kullanim:
    python scripts/compute_quality.py

Cikti:
    outputs/quality/quality_report.csv   -- label bazinda kalite metrikleri
    outputs/quality/summary.txt          -- okunabilir ozet rapor
"""

import csv
import io
import os
import sys

# Windows terminal UTF-8 zorla
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import nibabel as nib
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBJ_DIR    = os.path.join(PROJECT_DIR, "data", "subjects")
ATLAS_L_DIR = os.path.join(PROJECT_DIR, "data", "atlas", "labels", "left")
ATLAS_R_DIR = os.path.join(PROJECT_DIR, "data", "atlas", "labels", "right")
OUT_DIR     = os.path.join(PROJECT_DIR, "outputs", "quality")

SUBJECTS = sorted(
    e for e in os.listdir(SUBJ_DIR)
    if os.path.isdir(os.path.join(SUBJ_DIR, e))
)

# ─────────────────────────────────────────────────────────────────────────────

def voxel_volume_mm3(path: str) -> float:
    """NIfTI maskesindeki toplam hacmi mm3 olarak hesapla."""
    img   = nib.load(path)
    data  = img.get_fdata(dtype=np.float32)
    zooms = img.header.get_zooms()[:3]
    vvol  = float(np.prod(zooms))
    return float((data > 0.5).sum()) * vvol


def atlas_volume(label: str, side: str) -> float:
    """Atlas label hacmini oku."""
    d = ATLAS_L_DIR if side == "left" else ATLAS_R_DIR
    p = os.path.join(d, f"{label}.nii.gz")
    if not os.path.exists(p):
        return None
    return voxel_volume_mm3(p)


def warped_volume(subject: str, label: str, side: str) -> float:
    """Warped label hacmini oku."""
    p = os.path.join(
        SUBJ_DIR, subject, "warped", "labels",
        f"{side}_{label}_warped.nii.gz"
    )
    if not os.path.exists(p):
        return None
    return voxel_volume_mm3(p)


# ─────────────────────────────────────────────────────────────────────────────

def collect_label_names() -> list:
    """Birinci subjectten label listesini al."""
    import re
    label_dir = os.path.join(SUBJ_DIR, SUBJECTS[0], "warped", "labels")
    pat = re.compile(r"^left_(.+)_warped\.nii\.gz$")
    return sorted(
        m.group(1)
        for f in os.listdir(label_dir)
        if (m := pat.match(f))
    )


def volume_preservation_tier(vpi: float) -> str:
    """
    Hacim Koruma Indeksi (VPI = warped/atlas) yorumu:
      0.70 – 1.30  → iyi   (atlas hacminin ±30% icinde)
      0.50 – 1.50  → orta
      disi         → zayif
    """
    if vpi is None:
        return "N/A"
    if 0.70 <= vpi <= 1.30:
        return "IYI"
    if 0.50 <= vpi <= 1.50:
        return "ORTA"
    return "ZAYIF"


def symmetry_tier(lr_ratio: float) -> str:
    """
    Sol/Sag hacim orani yorumu (normal beyin ~1.0):
      0.75 – 1.25  → simetrik
      0.50 – 1.50  → hafif asimetrik
      disi         → belirgin asimetri
    """
    if lr_ratio is None:
        return "N/A"
    if 0.75 <= lr_ratio <= 1.25:
        return "SIMETRIK"
    if 0.50 <= lr_ratio <= 1.50:
        return "HAFIF_ASIM"
    return "ASIMETRIK"


# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    labels = collect_label_names()

    print(f"\n{'='*64}")
    print(f"  Warp Kalite Analizi  --  {len(SUBJECTS)} subject, {len(labels)} label")
    print(f"{'='*64}\n")

    rows = []  # CSV icin

    # Her label icin: atlas hacmi, warped hacimleri, VPI, simetri, cross-CV
    label_stats = {}

    for label in labels:
        atl_L = atlas_volume(label, "left")
        atl_R = atlas_volume(label, "right")

        warped_L = {}
        warped_R = {}
        for subj in SUBJECTS:
            wL = warped_volume(subj, label, "left")
            wR = warped_volume(subj, label, "right")
            if wL is not None:
                warped_L[subj] = wL
            if wR is not None:
                warped_R[subj] = wR

        # Cross-subject CV (sol taraf, daha istikrarlı)
        vols_L = list(warped_L.values())
        if len(vols_L) >= 2 and np.mean(vols_L) > 0:
            cv_L = float(np.std(vols_L) / np.mean(vols_L))
        else:
            cv_L = None

        for subj in SUBJECTS:
            wL = warped_L.get(subj)
            wR = warped_R.get(subj)

            # Hacim Koruma Indeksi
            vpi_L = (wL / atl_L) if (wL and atl_L and atl_L > 0) else None
            vpi_R = (wR / atl_R) if (wR and atl_R and atl_R > 0) else None

            # Sol/Sag Simetri
            lr_ratio = (wL / wR) if (wL and wR and wR > 0) else None

            row = {
                "subject":          subj,
                "label":            label,
                "atlas_vol_L_mm3":  round(atl_L, 2) if atl_L else "",
                "atlas_vol_R_mm3":  round(atl_R, 2) if atl_R else "",
                "warped_vol_L_mm3": round(wL, 2) if wL else "",
                "warped_vol_R_mm3": round(wR, 2) if wR else "",
                "VPI_left":         round(vpi_L, 3) if vpi_L else "",
                "VPI_right":        round(vpi_R, 3) if vpi_R else "",
                "VPI_tier":         volume_preservation_tier(vpi_L),
                "LR_ratio":         round(lr_ratio, 3) if lr_ratio else "",
                "symmetry_tier":    symmetry_tier(lr_ratio),
                "cross_subject_CV": round(cv_L, 3) if cv_L else "",
            }
            rows.append(row)

        label_stats[label] = {
            "atl_L": atl_L, "atl_R": atl_R,
            "vols_L": vols_L,
            "cv_L": cv_L,
        }

    # --CSV yaz ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "quality_report.csv")
    cols = [
        "subject", "label",
        "atlas_vol_L_mm3", "atlas_vol_R_mm3",
        "warped_vol_L_mm3", "warped_vol_R_mm3",
        "VPI_left", "VPI_right", "VPI_tier",
        "LR_ratio", "symmetry_tier",
        "cross_subject_CV",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    # --Özet metin raporu ────────────────────────────────────────────────────
    summary_path = os.path.join(OUT_DIR, "summary.txt")
    iyi   = [r for r in rows if r["VPI_tier"] == "IYI"]
    orta  = [r for r in rows if r["VPI_tier"] == "ORTA"]
    zayif = [r for r in rows if r["VPI_tier"] == "ZAYIF"]
    sim   = [r for r in rows if r["symmetry_tier"] == "SIMETRIK"]
    asim  = [r for r in rows if r["symmetry_tier"] == "ASIMETRIK"]

    # En tutarsiz labellar (en yuksek CV)
    cv_table = [
        (lbl, st["cv_L"])
        for lbl, st in label_stats.items()
        if st["cv_L"] is not None
    ]
    cv_table.sort(key=lambda x: x[1], reverse=True)

    # En kotu VPI'li labellar
    vpi_bad = {}
    for r in rows:
        if r["VPI_left"] != "" and r["VPI_tier"] in ("ZAYIF", "ORTA"):
            vpi_bad[r["label"]] = vpi_bad.get(r["label"], 0) + 1
    vpi_bad_sorted = sorted(vpi_bad.items(), key=lambda x: x[1], reverse=True)

    lines = []
    lines.append("=" * 64)
    lines.append("  THALAMIC NUCLEI WARP KALITE RAPORU")
    lines.append("=" * 64)
    lines.append(f"  Subjects : {len(SUBJECTS)}")
    lines.append(f"  Labels   : {len(labels)}")
    lines.append(f"  Toplam   : {len(rows)} olcum\n")

    lines.append("--Hacim Koruma Indeksi (VPI = warped/atlas) ─────────────")
    lines.append(f"  IYI   (0.70-1.30) : {len(iyi):4d} olcum  ({100*len(iyi)/len(rows):.1f}%)")
    lines.append(f"  ORTA  (0.50-1.50) : {len(orta):4d} olcum  ({100*len(orta)/len(rows):.1f}%)")
    lines.append(f"  ZAYIF (<0.50/>1.50): {len(zayif):4d} olcum  ({100*len(zayif)/len(rows):.1f}%)\n")

    lines.append("--Sol/Sag Simetri ────────────────────────────────────────")
    lines.append(f"  SIMETRIK    (0.75-1.25) : {len(sim):4d}  ({100*len(sim)/len(rows):.1f}%)")
    lines.append(f"  ASIMETRIK   (<0.50/>1.50): {len(asim):4d}  ({100*len(asim)/len(rows):.1f}%)\n")

    lines.append("--Cross-Subject Tutarlilik (CV) — En Yuksek 10 ──────────")
    for lbl, cv in cv_table[:10]:
        bar = "#" * int(cv * 40)
        lines.append(f"  {lbl:<18s} CV={cv:.3f}  {bar}")

    lines.append("")
    lines.append("--Dikkat: En Sorunlu Labels (VPI zayif/orta) ────────────")
    for lbl, cnt in vpi_bad_sorted[:10]:
        lines.append(f"  {lbl:<18s} {cnt}/{len(SUBJECTS)} subjectte zayif/orta")

    lines.append("")
    lines.append(f"  CSV : {csv_path}")
    lines.append("=" * 64)

    report = "\n".join(lines)
    print(report)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print(f"\n  Rapor kaydedildi: {summary_path}\n")


if __name__ == "__main__":
    main()
