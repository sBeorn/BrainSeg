"""
İP-5: FreeSurfer Talamus Alt Çekirdek Segmentasyonu — Pseudo-GT Üretimi
=======================================================================
FreeSurfer 7.x'in `mri_segment_thalamic_nuclei` komutunu kullanarak
her hasta için bağımsız bir pseudo-ground-truth segmentasyonu üretir
ve BrainSeg pipeline çıktısıyla DSC karşılaştırması yapar.

DURUM: Bu script henüz çalıştırılmamıştır.
       FreeSurfer kurulumu ve recon-all (~8 saat/hasta) tamamlandıktan
       sonra çalıştırılacaktır.

Ön koşullar:
  1. FreeSurfer 7.4+ kurulu ve FREESURFER_HOME ayarlı
  2. freesurfer/license.txt mevcut
  3. recon-all tamamlanmış (veya bu script çalıştırılırken yapılır)

Kullanım:
    python scripts/ip5_freesurfer_gt.py
    python scripts/ip5_freesurfer_gt.py --subject IXI002-Guys-0828
    python scripts/ip5_freesurfer_gt.py --skip-recon   # recon-all atla
    python scripts/ip5_freesurfer_gt.py --check        # sadece kurulum kontrol

Çıktılar:
    outputs/<subject>/freesurfer/          — recon-all çıktısı
    outputs/<subject>/freesurfer_gt/       — NIfTI'ya dönüştürülmüş labellar
    outputs/freesurfer_comparison.csv      — BrainSeg vs FreeSurfer DSC tablosu
    outputs/freesurfer_comparison.html     — Karşılaştırma raporu
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import (
    SUBJECTS, PATIENT_DIR, OUTPUT_DIR, EXCLUDE_LABELS, TIER_THRESHOLDS, IS_WINDOWS
)
from scripts.utils.reporter import StepReporter


# ─── WSL yardımcıları (Windows'tan FreeSurfer çağrısı için) ──────────────────

def _wsl_path(windows_path: str) -> str:
    """
    Windows yolunu WSL yoluna çevirir: C:\\foo\\bar → /mnt/c/foo/bar
    Linux/WSL'de çalışırken yolu değiştirmeden döndürür.
    """
    if not IS_WINDOWS:
        return windows_path
    p = windows_path.replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:].lstrip("/")
        return f"/mnt/{drive}/{rest}"
    return p


def _wsl_cmd(cmd: list) -> list:
    """
    Windows'ta subprocess komutunun başına 'wsl' ekler.
    Linux/WSL'de doğrudan çalıştırır.
    """
    if IS_WINDOWS:
        return ["wsl"] + cmd
    return cmd


# ─── FreeSurfer talamus label eşleştirme tablosu ──────────────────────────────
# FreeSurfer 7.x mri_segment_thalamic_nuclei çıktısındaki integer etiket
# kodlarını BrainSeg atlas label isimlerine eşler.
# Kaynak: FreeSurfer ThalamicNuclei.v12.T1.T2.mgz LUT
# https://surfer.nmr.mgh.harvard.edu/fswiki/ThalamicNuclei

FS_LABEL_MAP = {
    # Sol hemisferde (left)
    8103: ("AV",    "left"),
    8104: ("CL",    "left"),
    8105: ("CM",    "left"),   # CeM ile eşleştir
    8106: ("CeM",   "left"),
    8108: ("LD",    "left"),
    8109: ("LGNpc", "left"),   # LGN — pc/mc ayrımı FreeSurfer'da yok; pc kullan
    8110: ("LP",    "left"),
    8112: ("MDmc",  "left"),
    8113: ("MDpc",  "left"),
    8115: ("MGN",   "left"),
    8116: ("MV",    "left"),
    8117: ("Pf",    "left"),
    8119: ("PuA",   "left"),
    8120: ("PuI",   "left"),
    8121: ("PuL",   "left"),
    8122: ("PuM",   "left"),
    8123: ("VApc",  "left"),
    8124: ("VAmc",  "left"),
    8125: ("VLa",   "left"),
    8126: ("VLpd",  "left"),
    8127: ("VM",    "left"),
    8128: ("VPLp",  "left"),
    8129: ("VPI",   "left"),
    8130: ("VPM",   "left"),
    # Sağ hemisferde (right) — aynı kodlar + 1000 offset
    9103: ("AV",    "right"),
    9104: ("CL",    "right"),
    9105: ("CM",    "right"),
    9106: ("CeM",   "right"),
    9108: ("LD",    "right"),
    9109: ("LGNpc", "right"),
    9110: ("LP",    "right"),
    9112: ("MDmc",  "right"),
    9113: ("MDpc",  "right"),
    9115: ("MGN",   "right"),
    9116: ("MV",    "right"),
    9117: ("Pf",    "right"),
    9119: ("PuA",   "right"),
    9120: ("PuI",   "right"),
    9121: ("PuL",   "right"),
    9122: ("PuM",   "right"),
    9123: ("VApc",  "right"),
    9124: ("VAmc",  "right"),
    9125: ("VLa",   "right"),
    9126: ("VLpd",  "right"),
    9127: ("VM",    "right"),
    9128: ("VPLp",  "right"),
    9129: ("VPI",   "right"),
    9130: ("VPM",   "right"),
}

# BrainSeg atlas'ında olup FreeSurfer'da karşılığı olmayan labellar
FS_NOT_AVAILABLE = {
    "AD", "AM", "Hb", "LGNmc", "Li", "mtt", "Po",
    "Pv", "RN", "SG", "sPf", "STh", "VLpv", "VPLa",
}


# ─── FreeSurfer kurulum kontrolü ──────────────────────────────────────────────

def check_freesurfer() -> dict:
    """
    FreeSurfer kurulumunu kontrol et.
    Windows'ta WSL üzerinden, Linux'ta doğrudan kontrol eder.
    Döndürür: {'ok': bool, 'version': str, 'home': str, 'issues': list}
    """
    issues = []
    fs_home = os.environ.get("FREESURFER_HOME", "")

    if IS_WINDOWS:
        # Windows'ta shutil.which WSL komutlarını bulamaz; doğrudan çalıştırarak test et.
        for cmd in ["recon-all", "mri_segment_thalamic_nuclei", "mri_convert"]:
            try:
                r = subprocess.run(
                    ["wsl", cmd, "--version"],
                    capture_output=True, text=True, timeout=10
                )
                if r.returncode not in (0, 1):   # --version bazı araçlarda 1 döner
                    issues.append(f"WSL'de komut bulunamadı: {cmd}")
            except Exception:
                issues.append(f"WSL'de komut bulunamadı: {cmd}")
        # WSL içinde FREESURFER_HOME ve lisans kontrolü
        try:
            r = subprocess.run(
                ["wsl", "bash", "-lc",
                 'echo "HOME=$FREESURFER_HOME"; '
                 'test -f "$FREESURFER_HOME/license.txt" && echo "LIC=OK" || echo "LIC=MISSING"'],
                capture_output=True, text=True, timeout=10
            )
            out = r.stdout
            if "LIC=MISSING" in out:
                issues.append("WSL'de FreeSurfer lisans dosyası bulunamadı.")
        except Exception as e:
            issues.append(f"WSL kontrol hatası: {e}")
    else:
        if not fs_home:
            issues.append("FREESURFER_HOME ortam değişkeni tanımlı değil.")
        elif not os.path.isdir(fs_home):
            issues.append(f"FREESURFER_HOME dizini bulunamadı: {fs_home}")

        for cmd in ["recon-all", "mri_segment_thalamic_nuclei", "mri_convert"]:
            if not shutil.which(cmd):
                issues.append(f"Komut bulunamadı: {cmd}")

        license_paths = [
            os.path.join(fs_home, "license.txt") if fs_home else "",
            os.path.join(fs_home, ".license")    if fs_home else "",
            os.path.expanduser("~/.freesurfer/license.txt"),
        ]
        if not any(os.path.exists(p) for p in license_paths if p):
            issues.append("FreeSurfer lisans dosyası bulunamadı (license.txt).")

    version = "bilinmiyor"
    if not issues:
        try:
            r = subprocess.run(
                _wsl_cmd(["recon-all", "--version"]),
                capture_output=True, text=True, timeout=10
            )
            version = (r.stdout or r.stderr).strip().split("\n")[0]
        except Exception:
            pass

    return {
        "ok":      len(issues) == 0,
        "version": version,
        "home":    fs_home,
        "issues":  issues,
    }


# ─── recon-all ────────────────────────────────────────────────────────────────

def run_recon_all(subject_id: str, skip: bool = False) -> str:
    """
    FreeSurfer recon-all çalıştır.
    Döndürür: subjects_dir içindeki subject dizin yolu.
    """
    t1_path  = os.path.join(PATIENT_DIR, f"{subject_id}-T1.nii.gz")
    subj_dir = os.path.join(OUTPUT_DIR, subject_id, "freesurfer")
    done_flag = os.path.join(subj_dir, "mri", "aparc+aseg.mgz")

    if skip or os.path.exists(done_flag):
        print(f"  [recon-all] Mevcut sonuç kullanılıyor: {subj_dir}")
        return subj_dir

    if not os.path.exists(t1_path):
        raise FileNotFoundError(f"T1 bulunamadı: {t1_path}")

    os.makedirs(os.path.dirname(subj_dir), exist_ok=True)
    print(f"  [recon-all] Başlıyor — tahmini süre: 6–10 saat...")
    t0 = time.time()

    cmd = _wsl_cmd([
        "recon-all",
        "-i",       _wsl_path(t1_path),
        "-s",       subject_id,
        "-sd",      _wsl_path(os.path.dirname(subj_dir)),
        "-all",
        "-parallel",
    ])
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - t0

    if result.returncode != 0:
        raise RuntimeError(
            f"recon-all başarısız (kod {result.returncode}):\n{result.stderr[-2000:]}"
        )

    print(f"  [recon-all] Tamamlandı ({duration/3600:.1f} saat)")
    return subj_dir


# ─── Talamus alt çekirdek segmentasyonu ───────────────────────────────────────

def run_thalamic_segmentation(subject_id: str, subj_dir: str, use_t2: bool = True) -> str:
    """
    mri_segment_thalamic_nuclei çalıştır.
    Döndürür: segmentasyon MGZ dosyasının yolu.
    """
    seg_mgz = os.path.join(subj_dir, "mri", "ThalamicNuclei.v12.T1.T2.mgz")
    if not use_t2:
        seg_mgz = os.path.join(subj_dir, "mri", "ThalamicNuclei.v12.T1.mgz")

    if os.path.exists(seg_mgz):
        print(f"  [thal-seg] Mevcut segmentasyon kullanılıyor: {seg_mgz}")
        return seg_mgz

    t2_path = os.path.join(PATIENT_DIR, f"{subject_id}-T2.nii.gz")
    t2_registered = os.path.join(OUTPUT_DIR, subject_id, "preproc", "T2_to_T1_preproc.nii.gz")

    # T2 tercih sırası: preprocessed > ham
    t2_file = t2_registered if os.path.exists(t2_registered) else (
              t2_path       if os.path.exists(t2_path) else None)

    cmd = _wsl_cmd([
        "mri_segment_thalamic_nuclei",
        "-s", subject_id,
        "-sd", _wsl_path(os.path.dirname(subj_dir)),
    ])
    if t2_file and use_t2:
        cmd += ["--T2", _wsl_path(t2_file)]
        print(f"  [thal-seg] T2 kullanılıyor: {t2_file}")
    else:
        print("  [thal-seg] Yalnızca T1 kullanılıyor (T2 bulunamadı)")

    print(f"  [thal-seg] Çalışıyor...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"  [thal-seg] Tamamlandı ({time.time()-t0:.0f} sn)")

    if result.returncode != 0:
        raise RuntimeError(
            f"mri_segment_thalamic_nuclei başarısız:\n{result.stderr[-2000:]}"
        )
    return seg_mgz


# ─── MGZ → NIfTI dönüştürme ve label ayrıştırma ──────────────────────────────

def extract_labels_from_segmentation(
    seg_mgz: str, out_dir: str
) -> dict:
    """
    FreeSurfer segmentasyon MGZ'yi NIfTI'ya çevir,
    her label için ayrı binary NIfTI dosyası üret.

    Döndürür: {"label_name_side": nifti_path, ...}
    """
    import nibabel as nib

    os.makedirs(out_dir, exist_ok=True)

    # 1. MGZ → NIfTI
    seg_nii = os.path.join(out_dir, "thalamic_seg_fs.nii.gz")
    if not os.path.exists(seg_nii):
        result = subprocess.run(
            _wsl_cmd(["mri_convert", _wsl_path(seg_mgz), _wsl_path(seg_nii)]),
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"mri_convert başarısız:\n{result.stderr}")

    # 2. Parcellation NIfTI'yı yükle
    seg_img  = nib.load(seg_nii)
    seg_data = np.round(seg_img.get_fdata()).astype(int)
    affine   = seg_img.affine

    # 3. Her label için binary maske çıkar
    label_files = {}
    present_codes = np.unique(seg_data)

    for code in present_codes:
        if code not in FS_LABEL_MAP:
            continue
        label_name, side = FS_LABEL_MAP[code]
        key = f"{label_name}_{side}"

        out_path = os.path.join(out_dir, f"{key}.nii.gz")
        mask = (seg_data == code).astype(np.uint8)
        nib.save(nib.Nifti1Image(mask, affine), out_path)
        label_files[key] = out_path

    print(f"  [extract] {len(label_files)} label NIfTI olarak kaydedildi → {out_dir}")
    return label_files


# ─── DSC hesaplama ve karşılaştırma ───────────────────────────────────────────

def compute_dsc(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = (pred > 0).astype(bool)
    gt   = (gt   > 0).astype(bool)
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    return float(2 * inter / denom) if denom > 0 else float("nan")


def compare_subject(subject_id: str, fs_label_files: dict, warp_id: str = "W2") -> list:
    """
    BrainSeg propagation çıktısı vs FreeSurfer segmentasyonu — DSC karşılaştır.
    Döndürür: [{"subject", "label", "side", "brainseg_dsc_loo", "fs_dsc", ...}, ...]
    """
    import nibabel as nib

    prop_left  = os.path.join(OUTPUT_DIR, subject_id, "propagate", warp_id, "left_labels")
    prop_right = os.path.join(OUTPUT_DIR, subject_id, "propagate", warp_id, "right_labels")
    rel_csv    = os.path.join(OUTPUT_DIR, "expert_review", subject_id, "label_reliability.csv")

    # LOO DSC değerlerini yükle (karşılaştırma için)
    loo_dsc_map = {}
    if os.path.exists(rel_csv):
        import pandas as pd
        rel_df = pd.read_csv(rel_csv)
        rel_df["loo_dsc"] = pd.to_numeric(rel_df["loo_dsc"], errors="coerce")
        for _, row in rel_df.iterrows():
            loo_dsc_map[row["label"]] = row["loo_dsc"]

    rows = []
    for key, fs_path in fs_label_files.items():
        label_name, side = key.rsplit("_", 1)
        bs_dir  = prop_left if side == "left" else prop_right
        bs_path = os.path.join(bs_dir, f"{label_name}.nii.gz")

        if not os.path.exists(bs_path):
            continue

        fs_data  = nib.load(fs_path).get_fdata()
        bs_data  = nib.load(bs_path).get_fdata()

        # Boyut uyuşmazlığı — FreeSurfer ve BrainSeg farklı uzayda olabilir
        # Gerekirse basit resample: aynı shape değilse atla (resampling ip5_align'a bırakılır)
        if fs_data.shape != bs_data.shape:
            rows.append({
                "subject":          subject_id,
                "label":            label_name,
                "side":             side,
                "fs_dsc":           None,
                "brainseg_loo_dsc": loo_dsc_map.get(label_name),
                "note":             f"shape mismatch: fs={fs_data.shape} bs={bs_data.shape}",
            })
            continue

        dsc = compute_dsc(bs_data, fs_data)
        rows.append({
            "subject":          subject_id,
            "label":            label_name,
            "side":             side,
            "fs_dsc":           round(dsc, 4) if not np.isnan(dsc) else None,
            "brainseg_loo_dsc": loo_dsc_map.get(label_name),
            "note":             "",
        })

    return rows


# ─── Karşılaştırma raporu ─────────────────────────────────────────────────────

def save_comparison_report(all_rows: list):
    import pandas as pd

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUTPUT_DIR, "freesurfer_comparison.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nKarşılaştırma CSV: {csv_path}")

    # Özet — label bazında ortalama
    numeric = df[df["fs_dsc"].notna()].copy()
    if numeric.empty:
        print("  Uyarı: Karşılaştırılabilir label bulunamadı (shape mismatch?).")
        return

    summary = numeric.groupby("label").agg(
        fs_dsc_mean        =("fs_dsc",           "mean"),
        brainseg_dsc_mean  =("brainseg_loo_dsc",  "mean"),
        n_sides            =("fs_dsc",            "count"),
    ).round(3).sort_values("fs_dsc_mean", ascending=False)

    summary["delta"] = (summary["fs_dsc_mean"] - summary["brainseg_dsc_mean"]).round(3)

    print("\n=== BrainSeg (LOO proxy) vs FreeSurfer (pseudo-GT) ===")
    print(f"{'Label':10s} {'FS DSC':>8} {'BS DSC':>8} {'Delta':>7}")
    print("-" * 38)
    for lbl, row in summary.iterrows():
        delta_str = f"{row['delta']:+.3f}" if not np.isnan(row["delta"]) else "   N/A"
        print(f"  {lbl:10s} {row['fs_dsc_mean']:>8.3f} {row['brainseg_dsc_mean']:>8.3f} {delta_str:>7}")

    print(f"\nOrtalama FS DSC:      {summary['fs_dsc_mean'].mean():.3f}")
    print(f"Ortalama BrainSeg DSC:{summary['brainseg_dsc_mean'].mean():.3f}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="İP-5: FreeSurfer pseudo-GT üretimi ve BrainSeg karşılaştırması"
    )
    parser.add_argument("--subject",    default=None,
                        help="Tek hasta ID (varsayılan: tüm hastalar)")
    parser.add_argument("--skip-recon", action="store_true",
                        help="recon-all atla (zaten tamamlandıysa)")
    parser.add_argument("--no-t2",      action="store_true",
                        help="T2 kullanma (yalnızca T1)")
    parser.add_argument("--warp",       default="W2",
                        help="Karşılaştırılacak BrainSeg warp adayı (varsayılan: W2)")
    parser.add_argument("--check",      action="store_true",
                        help="Yalnızca FreeSurfer kurulum kontrolü yap")
    args = parser.parse_args()

    # Kurulum kontrolü
    fs = check_freesurfer()
    print("\n=== FreeSurfer Kurulum Kontrolü ===")
    print(f"  Durum  : {'OK' if fs['ok'] else 'HATA'}")
    print(f"  Sürüm  : {fs['version']}")
    print(f"  FS_HOME: {fs['home'] or 'tanımlı değil'}")
    if fs["issues"]:
        for issue in fs["issues"]:
            print(f"  ✗ {issue}")

    if args.check:
        return

    if not fs["ok"]:
        print("\nFreeSurfer kurulu değil veya yapılandırılmamış.")
        print("Kurulum için: https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall")
        print("\nBu script daha sonra FreeSurfer kurulduktan sonra çalıştırılacaktır.")
        sys.exit(1)

    subjects = [args.subject] if args.subject else SUBJECTS
    all_rows = []

    for subj in subjects:
        print(f"\n{'='*60}")
        print(f"  Hasta: {subj}")
        print(f"{'='*60}")

        rep = StepReporter(subj, "IP5_FreeSurfer", os.path.join(OUTPUT_DIR, subj, "freesurfer_gt"))
        try:
            subj_dir = run_recon_all(subj, skip=args.skip_recon)
            seg_mgz  = run_thalamic_segmentation(subj, subj_dir, use_t2=not args.no_t2)
            out_dir  = os.path.join(OUTPUT_DIR, subj, "freesurfer_gt")
            fs_labels = extract_labels_from_segmentation(seg_mgz, out_dir)
            rows      = compare_subject(subj, fs_labels, args.warp)
            all_rows.extend(rows)
            rep.log("fs_labels_extracted", len(fs_labels))
            rep.log("comparisons_made", len(rows))
        except Exception as exc:
            rep.record_exception(exc)
            rep.finish("FAILED")
            continue
        rep.finish("SUCCESS")

    if all_rows:
        save_comparison_report(all_rows)


if __name__ == "__main__":
    main()
