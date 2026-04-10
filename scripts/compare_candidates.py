"""
Düzey-3 Raporlama: Tüm Aday Karşılaştırmaları
===============================================
Tüm hastalar ve pipeline adayları için karşılaştırma CSV + HTML raporları üretir.

Çıktılar (outputs/ kökünde):
  comparison_warp_candidates.csv
  comparison_propagate_labels.csv
  comparison_refine_candidates.csv
  pipeline_ranking_final.csv
  QC_report.html

Kullanım:
    python scripts/compare_candidates.py
    python scripts/compare_candidates.py --warp-only
    python scripts/compare_candidates.py --subject IXI002-Guys-0828
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Windows terminal encoding fix
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(_sys.stderr, "reconfigure"):
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")


import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import (
    SUBJECTS, OUTPUT_DIR, WARP_CANDIDATES, REFINE_CANDIDATES, SCORE_WEIGHTS
)


# ──────────────────────────────────────────────────────────────────────────────
# Warp Karşılaştırması
# ──────────────────────────────────────────────────────────────────────────────

def compare_warp_candidates(subjects: list) -> pd.DataFrame:
    """W1/W2/W3 × tüm hastalar → karşılaştırma tablosu."""
    rows = []
    for subj in subjects:
        for cand in WARP_CANDIDATES:
            report_path = os.path.join(
                OUTPUT_DIR, subj, "warp", cand, f"report_IP2_Warp_{cand}.json"
            )
            if not os.path.exists(report_path):
                continue
            with open(report_path) as f:
                r = json.load(f)
            m = r.get("metrics", {})
            qc = r.get("qc_flags", {})
            rows.append({
                "subject_id":       subj,
                "candidate":        cand,
                "status":           r.get("status"),
                "duration_sec":     r.get("duration_sec"),
                "NMI":              m.get("NMI_post_registration"),
                "jacobian_neg_ratio": m.get("jacobian_negative_ratio"),
                "inverse_consistency_mm": m.get("inverse_consistency_mm"),
                "jacobian_fail":    qc.get("jacobian_negative_ratio_fail", False),
                "description":      WARP_CANDIDATES[cand]["description"],
            })

    if not rows:
        print("  ⚠  Warp raporu bulunamadı.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Kazananı belirt: jacobian_fail=False olanlar arasında en yüksek NMI
    def pick_best(group):
        valid = group[~group["jacobian_fail"]]
        if valid.empty:
            return "NONE"
        return valid.loc[valid["NMI"].idxmax(), "candidate"]

    best = df.groupby("subject_id").apply(pick_best).reset_index()
    best.columns = ["subject_id", "best_candidate"]
    df = df.merge(best, on="subject_id")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Propagate / Label Karşılaştırması
# ──────────────────────────────────────────────────────────────────────────────

def compare_propagate_labels(subjects: list) -> pd.DataFrame:
    """Her label × warp adayı × hasta için morfometrik özet."""
    rows = []
    for subj in subjects:
        for cand in WARP_CANDIDATES:
            lq_path = os.path.join(
                OUTPUT_DIR, subj, "propagate", cand, "label_quality_report.json"
            )
            if not os.path.exists(lq_path):
                continue
            with open(lq_path) as f:
                lq = json.load(f)
            for label_name, sides in lq.get("labels", {}).items():
                for side, data in sides.items():
                    rows.append({
                        "subject_id":          subj,
                        "warp_candidate":      cand,
                        "label_name":          label_name,
                        "side":                side,
                        "volume_mm3":          data.get("volume_mm3"),
                        "connected_components":data.get("connected_components"),
                        "compactness":         data.get("compactness"),
                        "midline_distance_mm": data.get("midline_distance_mm"),
                        "LR_volume_ratio":     data.get("LR_volume_ratio"),
                        "in_problem_labels":   (
                            f"{label_name}_{side}" in lq.get("problem_labels", [])
                        ),
                    })

    if not rows:
        print("  ⚠  Propagate raporu bulunamadı.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # En güvenilir warp adayı (en az problem_label + en stabil CC)
    def pick_best(group):
        counts = group.groupby("warp_candidate")["in_problem_labels"].sum()
        return counts.idxmin() if not counts.empty else "UNKNOWN"

    best = df.groupby(["subject_id", "label_name"]).apply(pick_best).reset_index()
    best.columns = ["subject_id", "label_name", "best_candidate"]
    df = df.merge(best, on=["subject_id", "label_name"])

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Refine Karşılaştırması
# ──────────────────────────────────────────────────────────────────────────────

def compare_refine_candidates(subjects: list) -> pd.DataFrame:
    rows = []
    for subj in subjects:
        for warp in WARP_CANDIDATES:
            for refine in REFINE_CANDIDATES:
                report_path = os.path.join(
                    OUTPUT_DIR, subj, "refine", refine,
                    f"report_IP4_Refine_{warp}_{refine}.json"
                )
                if not os.path.exists(report_path):
                    continue
                with open(report_path) as f:
                    r = json.load(f)
                m   = r.get("metrics", {})
                rm  = m.get("refine_metrics", {})

                # Ortalama metrikler (tüm label'lar)
                def _mean_metric(key):
                    vals = [v[key] for v in rm.values() if key in v]
                    return round(float(np.mean(vals)), 4) if vals else None

                rows.append({
                    "subject_id":          subj,
                    "warp_candidate":      warp,
                    "refine_candidate":    refine,
                    "status":              r.get("status"),
                    "duration_sec":        r.get("duration_sec"),
                    "mean_boundary_improvement": _mean_metric("boundary_improvement"),
                    "mean_volume_drift":   _mean_metric("volume_drift"),
                    "mean_compactness_delta": _mean_metric("compactness_delta"),
                    "total_vessel_overlap": sum(
                        v.get("vessel_overlap_voxels", 0) for v in rm.values()
                    ),
                    "description": REFINE_CANDIDATES[refine]["description"],
                })

    if not rows:
        print("  ⚠  Refine raporu bulunamadı.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Kazanan: vessel_overlap=0, en yüksek boundary_improvement, en düşük drift
    def pick_best(group):
        valid = group[group["total_vessel_overlap"] == 0]
        if valid.empty:
            valid = group
        valid = valid.dropna(subset=["mean_boundary_improvement"])
        if valid.empty:
            return "UNKNOWN"
        return valid.loc[valid["mean_boundary_improvement"].idxmax(), "refine_candidate"]

    best = df.groupby(["subject_id", "warp_candidate"]).apply(pick_best).reset_index()
    best.columns = ["subject_id", "warp_candidate", "best_candidate"]
    df = df.merge(best, on=["subject_id", "warp_candidate"])

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Birleşik Pipeline Sıralaması
# ──────────────────────────────────────────────────────────────────────────────

def build_pipeline_ranking(
    df_warp: pd.DataFrame,
    df_prop: pd.DataFrame,
    df_ref:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Bölüm 6.4 — S_total = 0.30*S_warp + 0.25*S_propagate + 0.25*S_refine
                         + 0.10*S_stability + 0.10*S_runtime
    """
    rows = []

    for subj in SUBJECTS:
        for warp in WARP_CANDIDATES:
            for refine in REFINE_CANDIDATES:
                row = {"subject_id": subj, "warp": warp, "refine": refine}

                # S_warp: normalize NMI (0-1), penalty for jacobian_fail
                if not df_warp.empty:
                    w_row = df_warp[
                        (df_warp["subject_id"] == subj) &
                        (df_warp["candidate"]  == warp)
                    ]
                    if not w_row.empty:
                        nmi = w_row["NMI"].values[0] or 0
                        jfail = bool(w_row["jacobian_fail"].values[0])
                        s_warp = 0.0 if jfail else min(float(nmi), 1.0)
                    else:
                        s_warp = 0.0
                else:
                    s_warp = 0.0

                # S_propagate: (1 - problem_ratio)
                if not df_prop.empty:
                    p_rows = df_prop[
                        (df_prop["subject_id"]     == subj) &
                        (df_prop["warp_candidate"] == warp)
                    ]
                    if not p_rows.empty:
                        problem_ratio = p_rows["in_problem_labels"].mean()
                        s_prop = 1.0 - float(problem_ratio)
                    else:
                        s_prop = 0.0
                else:
                    s_prop = 0.0

                # S_refine: normalize boundary_improvement
                if not df_ref.empty:
                    r_row = df_ref[
                        (df_ref["subject_id"]       == subj) &
                        (df_ref["warp_candidate"]   == warp) &
                        (df_ref["refine_candidate"] == refine)
                    ]
                    if not r_row.empty:
                        bi = r_row["mean_boundary_improvement"].values[0] or 0
                        vo = r_row["total_vessel_overlap"].values[0] or 0
                        s_ref = max(0.0, min(float(bi) * 10, 1.0))
                        if vo > 0:
                            s_ref *= 0.5   # Vessel overlap cezası
                    else:
                        s_ref = 0.0
                else:
                    s_ref = 0.0

                # S_stability: inverse consistency (heuristic)
                if not df_warp.empty and not w_row.empty:
                    ic = w_row.get("inverse_consistency_mm")
                    if ic is not None and len(ic) > 0 and ic.values[0] is not None:
                        ic_val = float(ic.values[0])
                        s_stab = max(0.0, 1.0 - ic_val)
                    else:
                        s_stab = 0.5
                else:
                    s_stab = 0.5

                # S_runtime: hızlı = iyi (normalize)
                if not df_warp.empty and not w_row.empty:
                    dur = w_row.get("duration_sec")
                    if dur is not None and len(dur) > 0 and dur.values[0]:
                        d = float(dur.values[0])
                        s_rt = max(0.0, 1.0 - d / 3600)   # 0→1, 3600s=0
                    else:
                        s_rt = 0.5
                else:
                    s_rt = 0.5

                s_total = (
                    SCORE_WEIGHTS["warp"]      * s_warp  +
                    SCORE_WEIGHTS["propagate"] * s_prop  +
                    SCORE_WEIGHTS["refine"]    * s_ref   +
                    SCORE_WEIGHTS["stability"] * s_stab  +
                    SCORE_WEIGHTS["runtime"]   * s_rt
                )

                row.update({
                    "S_warp":      round(s_warp, 4),
                    "S_propagate": round(s_prop, 4),
                    "S_refine":    round(s_ref,  4),
                    "S_stability": round(s_stab, 4),
                    "S_runtime":   round(s_rt,   4),
                    "S_total":     round(s_total, 4),
                })
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("S_total", ascending=False)

    # Kazanan
    best_combo = df.iloc[0]
    df["best_candidate"] = (
        (df["warp"] == best_combo["warp"]) &
        (df["refine"] == best_combo["refine"])
    ).map({True: f"BEST: {best_combo['warp']}+{best_combo['refine']} "
                 f"(S={best_combo['S_total']:.3f})",
           False: ""})

    return df


# ──────────────────────────────────────────────────────────────────────────────
# HTML Raporu
# ──────────────────────────────────────────────────────────────────────────────

def build_html_report(
    df_warp: pd.DataFrame,
    df_prop: pd.DataFrame,
    df_ref:  pd.DataFrame,
    df_rank: pd.DataFrame,
    output_path: str,
):
    qc_images = _collect_qc_images()

    def df_to_html(df, title):
        if df.empty:
            return f"<h3>{title}</h3><p>Veri yok.</p>"
        return (
            f"<h3>{title}</h3>"
            + df.to_html(
                classes="table table-striped table-hover table-sm",
                border=0, index=False, na_rep="—",
                float_format=lambda x: f"{x:.4f}",
            )
        )

    img_html = ""
    for img_path in qc_images[:50]:   # ilk 50 görsel
        rel = os.path.relpath(img_path, os.path.dirname(output_path))
        img_html += (
            f'<div class="qc-img">'
            f'<img src="{rel}" title="{os.path.basename(img_path)}">'
            f'<p>{os.path.basename(img_path)}</p></div>\n'
        )

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
  <meta charset="UTF-8">
  <title>BrainSeg Pipeline QC Raporu</title>
  <link rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
  <style>
    body {{ font-family: sans-serif; padding: 20px; }}
    h2   {{ color: #2c3e50; border-bottom: 2px solid #2c3e50; }}
    h3   {{ color: #34495e; margin-top: 30px; }}
    .table {{ font-size: 12px; }}
    .qc-section {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }}
    .qc-img {{ text-align: center; }}
    .qc-img img {{ max-width: 300px; border: 1px solid #ccc; }}
    .qc-img p {{ font-size: 10px; color: #555; }}
  </style>
</head>
<body>
  <h2>BrainSeg Pipeline — QC Raporu</h2>
  <p><b>Oluşturulma tarihi:</b> {pd.Timestamp.now().isoformat(timespec='seconds')}</p>
  <p><b>Hastalar:</b> {', '.join(SUBJECTS)}</p>

  {df_to_html(df_rank,  "Birleşik Pipeline Sıralaması")}
  {df_to_html(df_warp,  "Warp Adayları Karşılaştırması (W1/W2/W3)")}
  {df_to_html(df_prop,  "Label Morfometrik Özeti")}
  {df_to_html(df_ref,   "Refine Adayları Karşılaştırması (R1/R2/R3)")}

  <h3>QC Görselleri</h3>
  <div class="qc-section">
    {img_html if img_html else "<p>Görsel bulunamadı.</p>"}
  </div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  → HTML rapor kaydedildi: {output_path}")


def _collect_qc_images() -> list:
    """outputs/ altındaki tüm PNG dosyalarını bul."""
    images = []
    for root, _, files in os.walk(OUTPUT_DIR):
        for fname in sorted(files):
            if fname.endswith(".png"):
                images.append(os.path.join(root, fname))
    return images


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Düzey-3 Karşılaştırma Raporları")
    parser.add_argument("--subject",   type=str, default=None)
    parser.add_argument("--warp-only", action="store_true")
    args = parser.parse_args()

    subjects = [args.subject] if args.subject else SUBJECTS

    print(f"\n{'='*60}")
    print(f"  Karşılaştırma Raporları — {len(subjects)} hasta")
    print(f"{'='*60}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_warp = compare_warp_candidates(subjects)
    df_prop = compare_propagate_labels(subjects) if not args.warp_only else pd.DataFrame()
    df_ref  = compare_refine_candidates(subjects) if not args.warp_only else pd.DataFrame()
    df_rank = build_pipeline_ranking(df_warp, df_prop, df_ref)

    # CSV kaydet
    _save_csv(df_warp, os.path.join(OUTPUT_DIR, "comparison_warp_candidates.csv"))
    _save_csv(df_prop, os.path.join(OUTPUT_DIR, "comparison_propagate_labels.csv"))
    _save_csv(df_ref,  os.path.join(OUTPUT_DIR, "comparison_refine_candidates.csv"))
    _save_csv(df_rank, os.path.join(OUTPUT_DIR, "pipeline_ranking_final.csv"))

    # HTML
    build_html_report(
        df_warp, df_prop, df_ref, df_rank,
        os.path.join(OUTPUT_DIR, "QC_report.html"),
    )

    print(f"\n  Çıktılar → {OUTPUT_DIR}")
    if not df_rank.empty:
        best = df_rank.iloc[0]
        print(f"\n  ★  En iyi pipeline: Warp={best['warp']}, Refine={best['refine']}, "
              f"S_total={best['S_total']:.3f}")
    print(f"{'='*60}\n")


def _save_csv(df: pd.DataFrame, path: str):
    if df.empty:
        return
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"  → CSV kaydedildi: {path}")


if __name__ == "__main__":
    main()
