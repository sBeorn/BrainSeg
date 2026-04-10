"""
Morphometrics Özet Raporu
=========================
5 hasta × 41 label pipeline sonuçlarını araştırmacı dostu
HTML + CSV formatında üretir.
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import OUTPUT_DIR, SUBJECTS, EXCLUDE_LABELS

BASE        = OUTPUT_DIR
EXPERT_BASE = os.path.join(BASE, "expert_review")
OUT_DIR     = os.path.join(BASE, "morphometrics_summary")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── 1. Veri yükleme ───────────────────────────────────────────────────────────
dfs = []
for subj in SUBJECTS:
    df = pd.read_csv(os.path.join(EXPERT_BASE, subj, "label_reliability.csv"))
    df["subject"] = subj
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

# glcm kolonları sembol olarak gelmiş — atla
num_cols = [
    "atlas_mm3", "left_mm3", "right_mm3", "retention_%",
    "lr_volume_ratio", "t1_mean_left", "t1_mean_right",
    "mirror_err_mm", "compactness_l", "compactness_r",
    "t1_outlier_l", "t1_outlier_r",
    "gradient_l", "gradient_r",
    "bbox_fill_l", "bbox_fill_r",
    "consist_score", "loo_dsc", "quality_score",
]
for c in num_cols:
    all_df[c] = pd.to_numeric(all_df[c], errors="coerce")

# Thalamus pseudo-label'ları çıkar (config.EXCLUDE_LABELS)
df_labels = all_df[~all_df["label"].isin(EXCLUDE_LABELS)].copy()

# ─── 2. Warp karşılaştırması ────────────────────────────────────────────────────
warp_df = pd.read_csv(os.path.join(BASE, "comparison_warp_candidates.csv"))
warp_summary = warp_df.groupby("candidate")[["NMI", "jacobian_neg_ratio"]].mean().round(4)
warp_winner = "W2"  # pipeline ranking'den

# ─── 3. Per-label agregasyon ───────────────────────────────────────────────────
agg = df_labels.groupby("label").agg(
    atlas_mm3=("atlas_mm3", "first"),
    vol_left_mean=("left_mm3", "mean"),
    vol_right_mean=("right_mm3", "mean"),
    retention_pct=("retention_%", "mean"),
    lr_ratio_mean=("lr_volume_ratio", "mean"),
    lr_ratio_std=("lr_volume_ratio", "std"),
    mirror_err_mean=("mirror_err_mm", "mean"),
    compactness_mean=("compactness_l", "mean"),   # sol referans
    t1_outlier_mean=("t1_outlier_l", "mean"),
    gradient_mean=("gradient_l", "mean"),
    bbox_fill_mean=("bbox_fill_l", "mean"),
    consist_score=("consist_score", "mean"),
    loo_dsc=("loo_dsc", "mean"),
    quality_score=("quality_score", "mean"),
    tier1_n=("tier", lambda x: (x == "TIER-1").sum()),
    tier2_n=("tier", lambda x: (x == "TIER-2").sum()),
    tier3_n=("tier", lambda x: (x == "TIER-3").sum()),
).reset_index()

n_subj = len(SUBJECTS)
agg["tier1_pct"] = (agg["tier1_n"] / n_subj * 100).round(0).astype(int)
agg["dominant_tier"] = agg.apply(
    lambda r: "TIER-1" if r.tier1_n >= 4 else ("TIER-3" if r.tier3_n >= 3 else "TIER-2"),
    axis=1,
)
agg = agg.sort_values("quality_score", ascending=False).reset_index(drop=True)
agg.index += 1  # 1'den başlat

# CSV kaydet
csv_path = os.path.join(OUT_DIR, "label_morphometrics_summary.csv")
agg.to_csv(csv_path)
print(f"CSV kaydedildi: {csv_path}")

# ─── 4. HTML ───────────────────────────────────────────────────────────────────
def tier_badge(tier):
    colors = {"TIER-1": "#2ecc71", "TIER-2": "#f39c12", "TIER-3": "#e74c3c"}
    return f'<span style="background:{colors.get(tier,"#aaa")};color:#fff;padding:2px 8px;border-radius:4px;font-size:0.85em;font-weight:bold">{tier}</span>'

def bar(val, max_val=1.0, color="#3498db"):
    pct = min(max(val / max_val, 0), 1) * 100 if not np.isnan(val) else 0
    return (
        f'<div style="background:#eee;border-radius:3px;height:14px;width:80px;display:inline-block;vertical-align:middle">'
        f'<div style="background:{color};width:{pct:.0f}%;height:100%;border-radius:3px"></div></div>'
        f' <span style="font-size:0.8em">{val:.2f}</span>'
    )

def dsc_color(v):
    if np.isnan(v): return "#999"
    if v >= 0.7: return "#27ae60"
    if v >= 0.5: return "#f39c12"
    return "#e74c3c"

# --- HTML tablo satırı ---
def row_html(r):
    tier = r["dominant_tier"]
    tier_colors = {"TIER-1": "#f0faf4", "TIER-2": "#fffbf0", "TIER-3": "#fdf0f0"}
    bg = tier_colors.get(tier, "#fff")
    dsc_v = r["loo_dsc"]
    dsc_str = f'<span style="color:{dsc_color(dsc_v)};font-weight:bold">{dsc_v:.3f}</span>' if not np.isnan(dsc_v) else "—"
    cons = r["consist_score"]
    cons_str = f'{cons:.3f}' if not np.isnan(cons) else "—"
    return (
        f'<tr style="background:{bg}">'
        f'<td style="font-weight:bold;padding:5px 8px">{r["label"]}</td>'
        f'<td style="text-align:right;padding:5px 8px">{r["atlas_mm3"]:.0f}</td>'
        f'<td style="text-align:right;padding:5px 8px">{r["vol_left_mean"]:.1f}</td>'
        f'<td style="text-align:right;padding:5px 8px">{r["vol_right_mean"]:.1f}</td>'
        f'<td style="text-align:right;padding:5px 8px">{r["retention_pct"]:.0f}%</td>'
        f'<td style="text-align:right;padding:5px 8px">{r["lr_ratio_mean"]:.2f} ± {r["lr_ratio_std"]:.2f}</td>'
        f'<td style="text-align:right;padding:5px 8px">{r["mirror_err_mean"]:.1f}</td>'
        f'<td style="padding:5px 8px">{bar(r["compactness_mean"], 1.0, "#8e44ad")}</td>'
        f'<td style="padding:5px 8px">{bar(r["consist_score"] if not np.isnan(r["consist_score"]) else 0, 1.0, "#2980b9")}</td>'
        f'<td style="text-align:center;padding:5px 8px">{dsc_str}</td>'
        f'<td style="text-align:center;padding:5px 8px">{bar(r["quality_score"], 1.0, "#27ae60")}</td>'
        f'<td style="text-align:center;padding:5px 8px">{tier_badge(tier)}'
        f' <small style="color:#888">{r["tier1_pct"]}% T1</small></td>'
        f'</tr>'
    )

rows_html = "\n".join(row_html(r) for _, r in agg.iterrows())

# Warp tablo
warp_rows = ""
for cand, wr in warp_summary.iterrows():
    flag = " ✓ KAZANAN" if cand == warp_winner else ""
    bold = "font-weight:bold;background:#f0faf4" if cand == warp_winner else ""
    warp_rows += (
        f'<tr style="{bold}"><td style="padding:5px 10px">{cand}{flag}</td>'
        f'<td style="text-align:right;padding:5px 10px">{wr["NMI"]:.4f}</td>'
        f'<td style="text-align:right;padding:5px 10px">{wr["jacobian_neg_ratio"]:.4f}</td></tr>'
    )

# Tier özet
tier_counts = df_labels["tier"].value_counts()
t1 = tier_counts.get("TIER-1", 0)
t2 = tier_counts.get("TIER-2", 0)
t3 = tier_counts.get("TIER-3", 0)
total = t1 + t2 + t3

# Problematik labellar
prob_labels = agg[agg["dominant_tier"] == "TIER-3"].sort_values("quality_score")
prob_rows = ""
for _, r in prob_labels.iterrows():
    flags = df_labels[df_labels["label"] == r["label"]]["morph_flags"].dropna()
    flag_str = flags.iloc[0] if len(flags) > 0 else "—"
    dsc_val = r["loo_dsc"]
    dsc_col = dsc_color(dsc_val)
    dsc_disp = f"{dsc_val:.3f}"
    qs = r["quality_score"]
    lbl = r["label"]
    atl = r["atlas_mm3"]
    prob_rows += (
        f'<tr><td style="font-weight:bold;padding:5px 8px">{lbl}</td>'
        f'<td style="padding:5px 8px">{atl:.0f} mm³</td>'
        f'<td style="text-align:center;padding:5px 8px"><span style="color:{dsc_col};font-weight:bold">{dsc_disp}</span></td>'
        f'<td style="text-align:center;padding:5px 8px">{qs:.3f}</td>'
        f'<td style="padding:5px 8px;color:#c0392b;font-size:0.9em">{flag_str}</td></tr>'
    )

html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<title>BrainSeg Morphometrics Raporu</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px 40px; color: #2c3e50; background: #f8f9fa; }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
  h2 {{ color: #34495e; margin-top: 40px; }}
  h3 {{ color: #7f8c8d; }}
  .card {{ background: #fff; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .stat-grid {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0; }}
  .stat-box {{ background: #fff; border-radius: 8px; padding: 16px 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; min-width: 140px; }}
  .stat-box .num {{ font-size: 2em; font-weight: bold; }}
  .stat-box .lbl {{ color: #7f8c8d; font-size: 0.85em; margin-top: 4px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.88em; }}
  th {{ background: #34495e; color: #fff; padding: 8px 10px; text-align: left; position: sticky; top: 0; }}
  tr:hover {{ filter: brightness(0.97); }}
  .section-note {{ color: #7f8c8d; font-size: 0.9em; margin: -10px 0 10px 0; }}
  .legend {{ display: flex; gap: 16px; margin: 10px 0; font-size: 0.85em; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .dot {{ width: 14px; height: 14px; border-radius: 50%; }}
</style>
</head>
<body>

<h1>BrainSeg — Morphometrics Özet Raporu</h1>
<p style="color:#7f8c8d">5 hasta × 41 anatomik label | Seçilen pipeline: <strong>W2 + ham propagation</strong> | Tarih: 2026-04-10</p>

<!-- ─── BÖLÜM 1: Pipeline seçimi ─── -->
<h2>1. Pipeline Seçimi</h2>
<div class="card">
  <p>3 warp adayı (W1, W2, W3) ve 3 refinement adayı (R1, R2, R3) karşılaştırıldı.
     Toplam 45 segmentasyon üretildi (5 hasta × 3 warp × 3 refine).</p>
  <h3>Warp Adayları — Kayıt Kalitesi</h3>
  <p class="section-note">NMI: Normalized Mutual Information (yüksek = iyi). Jacobian neg. ratio: topolojik bozulma (0 = bozulma yok).</p>
  <table>
    <tr><th>Aday</th><th>NMI (ort.)</th><th>Jacobian Neg. Ratio</th></tr>
    {warp_rows}
  </table>
  <br>
  <h3>Refinement Kararı</h3>
  <p>Ham propagation (R1 = refine uygulanmadı) tüm hastalarda R2/R3'ten üstün çıktı.
     <strong>R2 ve R3 label tutarlılığını bozmaktadır</strong> — TIER-3 oranını artırmaktadır.</p>
  <div style="background:#f0faf4;border-left:4px solid #2ecc71;padding:10px 16px;border-radius:4px;margin-top:10px">
    <strong>Seçilen pipeline:</strong> W2 (en yüksek NMI) + ham propagation (refinement yok)<br>
    <strong>Gerekçe:</strong> En yüksek kayıt kalitesi (NMI=1.1086) + label tutarlılığını koruma
  </div>
</div>

<!-- ─── BÖLÜM 2: Genel istatistikler ─── -->
<h2>2. Segmentasyon Kalitesi — Genel Bakış</h2>
<div class="stat-grid">
  <div class="stat-box">
    <div class="num" style="color:#2ecc71">{t1}</div>
    <div class="lbl">TIER-1 (Güvenilir)<br>{t1/total*100:.0f}%</div>
  </div>
  <div class="stat-box">
    <div class="num" style="color:#f39c12">{t2}</div>
    <div class="lbl">TIER-2 (Sınır)<br>{t2/total*100:.0f}%</div>
  </div>
  <div class="stat-box">
    <div class="num" style="color:#e74c3c">{t3}</div>
    <div class="lbl">TIER-3 (Güvenilmez)<br>{t3/total*100:.0f}%</div>
  </div>
  <div class="stat-box">
    <div class="num" style="color:#3498db">{df_labels['loo_dsc'].mean():.3f}</div>
    <div class="lbl">Ort. LOO DSC<br>(41 label, 5 hasta)</div>
  </div>
  <div class="stat-box">
    <div class="num" style="color:#8e44ad">{df_labels['consist_score'].mean():.3f}</div>
    <div class="lbl">Ort. Cross-subject<br>Tutarlılık (CV bazlı)</div>
  </div>
  <div class="stat-box">
    <div class="num" style="color:#e67e22">{df_labels['quality_score'].mean():.3f}</div>
    <div class="lbl">Ort. Kalite Skoru<br>(kompozit)</div>
  </div>
</div>

<div class="card">
  <h3>LOO DSC Dağılımı</h3>
  <p class="section-note">Leave-One-Out Dice Similarity Coefficient: atlas ≠ hedef veri olduğunda label örtüşme kalitesi.</p>
  <table style="width:auto">
    <tr><th>Percentil</th><th>DSC değeri</th><th>Yorum</th></tr>
    <tr><td>Min</td><td style="color:#e74c3c;font-weight:bold">{df_labels['loo_dsc'].min():.3f}</td><td>En kötü label</td></tr>
    <tr><td>25. percentil</td><td>{df_labels['loo_dsc'].quantile(0.25):.3f}</td><td></td></tr>
    <tr><td>Medyan</td><td>{df_labels['loo_dsc'].median():.3f}</td><td></td></tr>
    <tr><td>75. percentil</td><td>{df_labels['loo_dsc'].quantile(0.75):.3f}</td><td></td></tr>
    <tr><td>Max</td><td style="color:#27ae60;font-weight:bold">{df_labels['loo_dsc'].max():.3f}</td><td>En iyi label</td></tr>
  </table>
</div>

<!-- ─── BÖLÜM 3: Problematik labellar ─── -->
<h2>3. Dikkat Gerektiren Labellar (TIER-3)</h2>
<div class="card">
  <p class="section-note">Bu labellar 5 hastanın en az 3'ünde TIER-3 olarak sınıflandırıldı.
  Araştırmada kullanılmadan önce manuel doğrulama önerilir.</p>
  <table>
    <tr><th>Label</th><th>Atlas hacmi</th><th>LOO DSC</th><th>Kalite skoru</th><th>Sorun</th></tr>
    {prob_rows}
  </table>
</div>

<!-- ─── BÖLÜM 4: Tam morphometrics tablosu ─── -->
<h2>4. Tüm Labellar — Morphometrics Tablosu</h2>
<div class="card">
  <p class="section-note">
    Kalite skoruna göre azalan sırayla. Sol hemisferi baz alan metrikler (compactness, gradient vb.).<br>
    <strong>Sütunlar:</strong>
    Atlas mm³ = atlas referans hacmi |
    Vol L/R = 5 hasta ortalaması (mm³) |
    Retention = atlas hacminin ne kadarı korundu |
    LR Ratio = sol/sağ simetri (1.0 = tam simetrik) |
    Mirror Err = centroid simetri hatası (mm) |
    Compactness = şekil yuvarlaklığı (1=küre) |
    Tutarlılık = hastalar arası CV bazlı skor |
    LOO DSC = segmentasyon doğruluğu |
    Kalite = kompozit skor
  </p>
  <div class="legend">
    <div class="legend-item"><div class="dot" style="background:#f0faf4;border:1px solid #2ecc71"></div> TIER-1 arka planı</div>
    <div class="legend-item"><div class="dot" style="background:#fffbf0;border:1px solid #f39c12"></div> TIER-2 arka planı</div>
    <div class="legend-item"><div class="dot" style="background:#fdf0f0;border:1px solid #e74c3c"></div> TIER-3 arka planı</div>
  </div>
  <div style="overflow-x:auto">
  <table>
    <tr>
      <th>Label</th>
      <th>Atlas mm³</th>
      <th>Vol L (mm³)</th>
      <th>Vol R (mm³)</th>
      <th>Retention</th>
      <th>LR Ratio ± SD</th>
      <th>Mirror Err (mm)</th>
      <th>Compactness</th>
      <th>Tutarlılık</th>
      <th>LOO DSC</th>
      <th>Kalite Skoru</th>
      <th>Sınıf</th>
    </tr>
    {rows_html}
  </table>
  </div>
</div>

<!-- ─── BÖLÜM 5: Metodoloji notu ─── -->
<h2>5. Metodoloji Notu</h2>
<div class="card">
  <h3>Kalite Skoru Nasıl Hesaplandı?</h3>
  <p>Kompozit kalite skoru 3 bileşenden oluşur:</p>
  <ul>
    <li><strong>LOO DSC skoru (ağırlık 0.50):</strong> Atlas-to-subject leave-one-out Dice benzerliği</li>
    <li><strong>Morphometrics penaltı (ağırlık 0.30):</strong> 8 kriter üzerinden ceza puanı:
      <ol>
        <li>L/R hacim asimetrisi (simetrik olmayan yapılar için −0.30)</li>
        <li>T1 arka plan kirliliği (−0.20/hemisferde yüksek T1 ortalama)</li>
        <li>Centroid simetri hatası / mirror error (−0.15)</li>
        <li>Düşük kompaktlık/sphericity (−0.10 veya −0.20)</li>
        <li>Yüksek T1 outlier oranı (−0.08 veya −0.15)</li>
        <li>Zayıf gradient sınır netliği (−0.10/hemisferde)</li>
        <li>Düşük BBox doluluk oranı / parçalı yapı (−0.10)</li>
        <li>Düşük GLCM homojenlik (−0.10)</li>
      </ol>
    </li>
    <li><strong>Cross-subject tutarlılık (ağırlık 0.20):</strong> 5 hasta arası varyasyon katsayısı bazlı skor</li>
  </ul>
  <h3>TIER Sınıflandırması</h3>
  <ul>
    <li><strong>TIER-1:</strong> Kalite skoru yüksek, morfometrik sorun yok → araştırmada kullanılabilir</li>
    <li><strong>TIER-2:</strong> Sınır değerlerde → dikkatli yorumlanmalı</li>
    <li><strong>TIER-3:</strong> Ciddi morfometrik sorun → manuel doğrulama gerektirir</li>
  </ul>
  <h3>Veri Kaynağı</h3>
  <p>IXI veri seti — 5 sağlıklı yetişkin (Guys ve HH tarayıcıları). 1.5T T1-ağırlıklı MRI.
  Atlas: histolojik talamus atlas (Iglesias et al.). Warp: ANTsPy SyN deformable registration.</p>
</div>

</body>
</html>
"""

html_path = os.path.join(OUT_DIR, "morphometrics_report.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)
print(f"HTML rapor kaydedildi: {html_path}")

# ─── 5. Temiz CSV — araştırmacı için ──────────────────────────────────────────
clean_cols = {
    "label": "Label",
    "atlas_mm3": "Atlas_Hacim_mm3",
    "vol_left_mean": "Sol_Hacim_Ort_mm3",
    "vol_right_mean": "Sag_Hacim_Ort_mm3",
    "retention_pct": "Retention_Yuzde",
    "lr_ratio_mean": "LR_Oran_Ort",
    "lr_ratio_std": "LR_Oran_SD",
    "mirror_err_mean": "Mirror_Err_mm",
    "compactness_mean": "Kompaktlik",
    "t1_outlier_mean": "T1_Outlier_Oran",
    "gradient_mean": "Gradient_Netligi",
    "bbox_fill_mean": "BBox_Doluluk",
    "consist_score": "Hastalar_arasi_Tutarlilik",
    "loo_dsc": "LOO_DSC",
    "quality_score": "Kalite_Skoru",
    "dominant_tier": "TIER",
    "tier1_pct": "TIER1_Yuzde",
}
clean_df = agg.rename(columns=clean_cols)[list(clean_cols.values())]
clean_df = clean_df.round(3)
clean_csv = os.path.join(OUT_DIR, "morphometrics_clean.csv")
clean_df.to_csv(clean_csv, index=False, encoding="utf-8-sig")
print(f"Temiz CSV kaydedildi: {clean_csv}")
print(f"\nToplam {len(clean_df)} label, {len(SUBJECTS)} hasta")
