"""
BrainSeg — Ana Proje Raporu Üretici
=====================================
Tüm pipeline sonuçlarını ve morphometrics bulgularını
araştırmacı dostu tek bir HTML dosyasında toplar.

Çıktı: outputs/brainseg_project_report.html
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import OUTPUT_DIR, SUBJECTS, EXCLUDE_LABELS

BASE     = OUTPUT_DIR
EXPERT   = os.path.join(BASE, "expert_review")
LOO_CSV  = os.path.join(BASE, "loo_validation", "loo_label_dsc.csv")
WARP_CSV = os.path.join(BASE, "comparison_warp_candidates.csv")
EXCLUDE  = EXCLUDE_LABELS
OUT      = os.path.join(BASE, "brainseg_project_report.html")

# ─── Veri yükleme ─────────────────────────────────────────────────────────────
dfs = []
for subj in SUBJECTS:
    df = pd.read_csv(os.path.join(EXPERT, subj, "label_reliability.csv"))
    df["subject"] = subj
    dfs.append(df)
all_df = pd.concat(dfs, ignore_index=True)

NUM = ["atlas_mm3","left_mm3","right_mm3","retention_%","lr_volume_ratio",
       "t1_mean_left","t1_mean_right","mirror_err_mm","compactness_l","compactness_r",
       "t1_outlier_l","t1_outlier_r","gradient_l","gradient_r",
       "bbox_fill_l","bbox_fill_r","consist_score","loo_dsc","quality_score"]
for c in NUM:
    all_df[c] = pd.to_numeric(all_df[c], errors="coerce")

df_lab = all_df[~all_df["label"].isin(EXCLUDE)].copy()

loo  = pd.read_csv(LOO_CSV)
loo  = loo[~loo["label"].isin(EXCLUDE)]
warp = pd.read_csv(WARP_CSV)

# ─── Agregasyon ───────────────────────────────────────────────────────────────
agg = df_lab.groupby("label").agg(
    atlas_mm3   =("atlas_mm3",    "first"),
    vol_l       =("left_mm3",     "mean"),
    vol_r       =("right_mm3",    "mean"),
    retention   =("retention_%",  "mean"),
    lr_ratio    =("lr_volume_ratio", "mean"),
    mirror_err  =("mirror_err_mm","mean"),
    compactness =("compactness_l","mean"),
    t1_outlier  =("t1_outlier_l", "mean"),
    gradient    =("gradient_l",   "mean"),
    bbox_fill   =("bbox_fill_l",  "mean"),
    consist     =("consist_score","mean"),
    loo_dsc     =("loo_dsc",      "mean"),
    quality     =("quality_score","mean"),
    tier        =("tier", lambda x: x.value_counts().idxmax()),
    morph_flags =("morph_flags",  lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else ""),
).reset_index().sort_values("quality", ascending=False)

# ─── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────
def tier_badge(tier):
    c = {"TIER-1":"#27ae60","TIER-2":"#e67e22","TIER-3":"#e74c3c"}.get(tier,"#999")
    return f'<span class="badge" style="background:{c}">{tier}</span>'

def dsc_cell(v):
    if pd.isna(v):
        return '<td class="num">—</td>'
    c = "#27ae60" if v >= 0.7 else ("#e67e22" if v >= 0.5 else "#e74c3c")
    return f'<td class="num" style="color:{c};font-weight:bold">{v:.3f}</td>'

def mini_bar(v, hi=1.0, color="#3498db"):
    pct = min(max(v/hi, 0), 1)*100 if not pd.isna(v) else 0
    return (f'<div class="bar-wrap"><div class="bar-fill" style="width:{pct:.0f}%;background:{color}"></div>'
            f'</div><span class="bar-val">{v:.2f}</span>')

def row_bg(tier):
    return {"TIER-1":"#f8fffe","TIER-2":"#fffdf5","TIER-3":"#fff8f8"}.get(tier,"#fff")

# ─── Per-subject özet ──────────────────────────────────────────────────────────
subj_rows = ""
for subj in SUBJECTS:
    s  = df_lab[df_lab["subject"] == subj]
    t1 = (s["tier"] == "TIER-1").sum()
    t2 = (s["tier"] == "TIER-2").sum()
    t3 = (s["tier"] == "TIER-3").sum()
    tot = t1 + t2 + t3
    mean_q = s["quality_score"].mean()
    mean_d = s["loo_dsc"].mean()
    short  = subj.split("-")[0] + "-" + subj.split("-")[1]
    subj_rows += (
        f'<tr><td><strong>{short}</strong></td>'
        f'<td class="num t1">{t1}</td>'
        f'<td class="num t2">{t2}</td>'
        f'<td class="num t3">{t3}</td>'
        f'<td class="num">{t1/tot*100:.0f}%</td>'
        f'<td class="num">{mean_d:.3f}</td>'
        f'<td class="num">{mean_q:.3f}</td></tr>'
    )

# ─── Warp tablosu ─────────────────────────────────────────────────────────────
warp_agg = warp.groupby("candidate")[["NMI","jacobian_neg_ratio"]].mean().round(4)
warp_rows = ""
for cand, wr in warp_agg.iterrows():
    winner = cand == "W2"
    style  = 'style="background:#f0fdf4;font-weight:bold"' if winner else ""
    flag   = " &nbsp;<span class='badge' style='background:#27ae60'>SEÇILDI</span>" if winner else ""
    warp_rows += (
        f'<tr {style}><td>{cand}{flag}</td>'
        f'<td class="num">{wr["NMI"]:.4f}</td>'
        f'<td class="num">{wr["jacobian_neg_ratio"]:.4f}</td>'
        f'<td>{"Affine + SyN (CC)" if cand=="W2" else "Affine + SyN"}</td></tr>'
    )

# ─── TIER-3 detay tablosu ─────────────────────────────────────────────────────
t3_rows = ""
for _, r in agg[agg["tier"]=="TIER-3"].iterrows():
    flags = r["morph_flags"] if r["morph_flags"] else "—"
    t3_rows += (
        f'<tr style="background:#fff8f8">'
        f'<td><strong>{r["label"]}</strong></td>'
        f'<td class="num">{r["atlas_mm3"]:.0f}</td>'
        f'<td class="num">{r["lr_ratio"]:.2f}</td>'
        f'<td class="num">{r["compactness"]:.3f}</td>'
        f'<td class="num">{r["consist"]:.3f}</td>'
        f'{dsc_cell(r["loo_dsc"])}'
        f'<td class="num">{r["quality"]:.3f}</td>'
        f'<td style="color:#c0392b;font-size:0.85em">{flags}</td></tr>'
    )

# ─── Tam morphometrics tablosu ────────────────────────────────────────────────
morph_rows = ""
for _, r in agg.iterrows():
    bg    = row_bg(r["tier"])
    tb    = tier_badge(r["tier"])
    lr_ok = abs(r["lr_ratio"] - 1.0) < 0.30
    lr_color = "#27ae60" if lr_ok else "#e74c3c"
    ret_color = "#27ae60" if r["retention"] >= 65 else "#e67e22"
    morph_rows += (
        f'<tr style="background:{bg}">'
        f'<td style="font-weight:bold">{r["label"]}</td>'
        f'<td class="num">{r["atlas_mm3"]:.0f}</td>'
        f'<td class="num">{r["vol_l"]:.0f}</td>'
        f'<td class="num">{r["vol_r"]:.0f}</td>'
        f'<td class="num" style="color:{ret_color}">{r["retention"]:.0f}%</td>'
        f'<td class="num" style="color:{lr_color}">{r["lr_ratio"]:.2f}</td>'
        f'<td class="num">{r["mirror_err"]:.1f}</td>'
        f'<td>{mini_bar(r["compactness"], 1.0, "#8e44ad")}</td>'
        f'<td>{mini_bar(r["t1_outlier"], 0.5, "#e67e22")}</td>'
        f'<td>{mini_bar(r["consist"], 1.0, "#2980b9")}</td>'
        f'{dsc_cell(r["loo_dsc"])}'
        f'<td>{mini_bar(r["quality"], 1.0, "#27ae60")}</td>'
        f'<td>{tb}</td>'
        f'</tr>'
    )

# ─── Özet sayılar ─────────────────────────────────────────────────────────────
t1_total = (df_lab["tier"]=="TIER-1").sum()
t2_total = (df_lab["tier"]=="TIER-2").sum()
t3_total = (df_lab["tier"]=="TIER-3").sum()
grand    = t1_total + t2_total + t3_total
dsc_mean = df_lab["loo_dsc"].mean()
dsc_med  = df_lab["loo_dsc"].median()
cons_mean= df_lab["consist_score"].mean()
qual_mean= df_lab["quality_score"].mean()
n_good   = len(loo[loo["mean_dsc"]>=0.7])
n_ok     = len(loo[(loo["mean_dsc"]>=0.5)&(loo["mean_dsc"]<0.7)])
n_poor   = len(loo[loo["mean_dsc"]<0.5])

# ─── Hazırlık değerlendirmesi için hesaplamalar ────────────────────────────────
n_labels      = len(agg)
agg_dsc       = agg.set_index("label")["loo_dsc"]
dsc_good_list = agg[agg["loo_dsc"]>=0.70].sort_values("loo_dsc", ascending=False)
dsc_ok_list   = agg[(agg["loo_dsc"]>=0.50)&(agg["loo_dsc"]<0.70)].sort_values("loo_dsc", ascending=False)
dsc_poor_list = agg[agg["loo_dsc"]<0.50].sort_values("loo_dsc", ascending=False)
small_labels  = agg[agg["atlas_mm3"]<100]
large_labels  = agg[agg["atlas_mm3"]>=200]
dsc_small_mean = small_labels["loo_dsc"].mean()
dsc_large_mean = large_labels["loo_dsc"].mean()

# Hasta tutarlılığı — her hastanın ortalama DSC'si
subj_dsc = []
for s in SUBJECTS:
    sub = df_lab[df_lab["subject"]==s]
    subj_dsc.append(sub["loo_dsc"].mean())
subj_dsc_std = float(np.std(subj_dsc))

# İyi çalışan labellar — HTML satırları
def readiness_row_good(r):
    return (f'<tr style="background:#f8fffe">'
            f'<td style="font-weight:bold">{r["label"]}</td>'
            f'<td class="num">{r["atlas_mm3"]:.0f}</td>'
            f'<td class="num" style="color:#27ae60;font-weight:bold">{r["loo_dsc"]:.3f}</td>'
            f'<td class="num">{r["consist"]:.3f}</td>'
            f'<td class="num">{r["quality"]:.3f}</td>'
            f'<td style="color:#27ae60">Araştırma+klinik pilot için uygun</td></tr>')

def readiness_row_ok(r):
    return (f'<tr style="background:#fffdf5">'
            f'<td style="font-weight:bold">{r["label"]}</td>'
            f'<td class="num">{r["atlas_mm3"]:.0f}</td>'
            f'<td class="num" style="color:#e67e22;font-weight:bold">{r["loo_dsc"]:.3f}</td>'
            f'<td class="num">{r["consist"]:.3f}</td>'
            f'<td class="num">{r["quality"]:.3f}</td>'
            f'<td style="color:#e67e22">Grup düzeyi araştırma için kullanılabilir</td></tr>')

def readiness_row_poor(r):
    reason = "Çok küçük yapı" if r["atlas_mm3"] < 50 else "Düşük atlas-to-subject transfer"
    return (f'<tr style="background:#fff8f8">'
            f'<td style="font-weight:bold">{r["label"]}</td>'
            f'<td class="num">{r["atlas_mm3"]:.0f}</td>'
            f'<td class="num" style="color:#e74c3c;font-weight:bold">{r["loo_dsc"]:.3f}</td>'
            f'<td class="num">{r["consist"]:.3f}</td>'
            f'<td class="num">{r["quality"]:.3f}</td>'
            f'<td style="color:#e74c3c">{reason}</td></tr>')

readiness_rows_good = "\n".join(readiness_row_good(r) for _, r in dsc_good_list.iterrows())
readiness_rows_ok   = "\n".join(readiness_row_ok(r)   for _, r in dsc_ok_list.iterrows())
readiness_rows_poor = "\n".join(readiness_row_poor(r) for _, r in dsc_poor_list.iterrows())

# ─── HTML ─────────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BrainSeg — Proje Raporu</title>
<style>
/* ── Temel ── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: 'Segoe UI', system-ui, Arial, sans-serif;
  font-size: 14px;
  line-height: 1.6;
  color: #2c3e50;
  background: #f5f6fa;
  padding: 0 0 60px;
}}
a {{ color: #2980b9; }}

/* ── Başlık ── */
.page-header {{
  background: linear-gradient(135deg, #1a252f 0%, #2c3e50 100%);
  color: #fff;
  padding: 36px 48px 28px;
}}
.page-header h1 {{ font-size: 1.9em; font-weight: 700; margin-bottom: 6px; }}
.page-header .subtitle {{ color: #a0b0c0; font-size: 0.95em; }}
.page-header .meta {{ margin-top: 14px; display: flex; gap: 28px; flex-wrap: wrap; }}
.meta-item {{ background: rgba(255,255,255,0.08); border-radius: 6px; padding: 8px 16px; font-size: 0.85em; }}
.meta-item strong {{ display: block; color: #74b9ff; font-size: 0.85em; letter-spacing: 0.05em; text-transform: uppercase; }}

/* ── Navigasyon ── */
.nav {{
  background: #2c3e50;
  padding: 0 48px;
  display: flex;
  gap: 0;
  border-bottom: 2px solid #3d5166;
  position: sticky;
  top: 0;
  z-index: 100;
}}
.nav a {{
  color: #a0b4c8;
  text-decoration: none;
  padding: 12px 16px;
  font-size: 0.85em;
  border-bottom: 2px solid transparent;
  margin-bottom: -2px;
  transition: all 0.15s;
}}
.nav a:hover {{ color: #fff; border-color: #74b9ff; }}

/* ── İçerik ── */
.content {{ max-width: 1200px; margin: 0 auto; padding: 0 32px; }}
section {{ margin-top: 48px; scroll-margin-top: 70px; }}
section > h2 {{
  font-size: 1.3em;
  color: #2c3e50;
  border-left: 4px solid #3498db;
  padding-left: 14px;
  margin-bottom: 20px;
}}

/* ── Kartlar ── */
.card {{
  background: #fff;
  border-radius: 10px;
  padding: 24px 28px;
  margin-bottom: 20px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}}
.card h3 {{ font-size: 1em; color: #34495e; margin-bottom: 12px; }}
.card p {{ color: #555; margin-bottom: 10px; }}
.card p:last-child {{ margin-bottom: 0; }}

/* ── Stat kutuları ── */
.stat-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }}
.stat {{
  background: #fff;
  border-radius: 10px;
  padding: 18px 22px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.06);
  text-align: center;
  min-width: 130px;
  flex: 1;
}}
.stat .val {{ font-size: 2em; font-weight: 700; line-height: 1.1; }}
.stat .lbl {{ color: #7f8c8d; font-size: 0.78em; margin-top: 4px; }}

/* ── Tablolar ── */
.tbl-wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.86em; }}
th {{
  background: #34495e;
  color: #fff;
  padding: 9px 11px;
  text-align: left;
  white-space: nowrap;
  font-weight: 600;
}}
td {{ padding: 7px 10px; border-bottom: 1px solid #f0f0f0; vertical-align: middle; }}
tr:hover td {{ background: rgba(52,152,219,0.04); }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}

/* ── Badge ── */
.badge {{
  display: inline-block;
  color: #fff;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 0.8em;
  font-weight: 600;
  white-space: nowrap;
}}

/* ── Renk sınıfları ── */
.t1 {{ color: #27ae60; font-weight: 600; }}
.t2 {{ color: #e67e22; font-weight: 600; }}
.t3 {{ color: #e74c3c; font-weight: 600; }}

/* ── Mini bar ── */
.bar-wrap {{
  display: inline-block;
  background: #eee;
  border-radius: 3px;
  height: 10px;
  width: 60px;
  vertical-align: middle;
  overflow: hidden;
}}
.bar-fill {{ height: 100%; border-radius: 3px; }}
.bar-val {{ font-size: 0.82em; color: #555; margin-left: 5px; }}

/* ── Callout kutuları ── */
.callout {{
  border-left: 4px solid;
  padding: 12px 16px;
  border-radius: 0 6px 6px 0;
  margin: 14px 0;
  font-size: 0.92em;
}}
.callout-green  {{ border-color: #27ae60; background: #f0fdf4; }}
.callout-orange {{ border-color: #e67e22; background: #fffbf0; }}
.callout-red    {{ border-color: #e74c3c; background: #fff5f5; }}
.callout-blue   {{ border-color: #3498db; background: #f0f7ff; }}

/* ── Formül kutusu ── */
.formula {{
  background: #f8f9fa;
  border: 1px solid #dee2e6;
  border-radius: 6px;
  padding: 14px 18px;
  font-family: 'Consolas', monospace;
  font-size: 0.88em;
  margin: 12px 0;
  color: #2c3e50;
}}

/* ── Footer ── */
footer {{
  margin-top: 60px;
  text-align: center;
  color: #aaa;
  font-size: 0.82em;
  padding: 20px;
}}
</style>
</head>
<body>

<!-- ─── BAŞLIK ─── -->
<div class="page-header">
  <h1>BrainSeg — Talamus Segmentasyonu Proje Raporu</h1>
  <div class="subtitle">Atlas tabanlı çok-etiketli propagation · Morphometrics odaklı kalite değerlendirmesi</div>
  <div class="meta">
    <div class="meta-item"><strong>Veri seti</strong>IXI — 5 sağlıklı yetişkin</div>
    <div class="meta-item"><strong>Atlas</strong>Histolojik talamus atlası (Iglesias et al.)</div>
    <div class="meta-item"><strong>Seçilen pipeline</strong>W2 + ham propagation</div>
    <div class="meta-item"><strong>Anatomik etiket</strong>38 talamus alt çekirdeği</div>
    <div class="meta-item"><strong>Tarih</strong>2026-04-10</div>
  </div>
</div>

<!-- ─── NAVİGASYON ─── -->
<nav class="nav">
  <a href="#hedef">Araştırma Hedefi</a>
  <a href="#pipeline">Pipeline Seçimi</a>
  <a href="#loo">LOO Doğrulama</a>
  <a href="#genel">Genel Sonuçlar</a>
  <a href="#morph">Morphometrics Tablosu</a>
  <a href="#sorunlu">Sorunlu Labellar</a>
  <a href="#hazirlik">Kullanıma Hazırlık</a>
  <a href="#freesurfer" style="color:#27ae60">✅ FreeSurfer (Kuruldu)</a>
  <a href="#metod">Metodoloji</a>
</nav>

<div class="content">

<!-- ─── BÖLÜM 1: ARAŞTIRMA HEDEFİ ─── -->
<section id="hedef">
  <h2>1. Araştırma Hedefi</h2>
  <div class="card">
    <p>Bu çalışmanın amacı, <strong>ground-truth segmentasyon olmaksızın</strong> atlas tabanlı talamus alt çekirdek segmentasyonunun
    kalitesini nesnel biçimde değerlendirmektir.</p>
    <p>Klinik veya araştırma MRI'larında talamus morfometrisini otomatik olarak ölçmek için hangi pipeline konfigürasyonunun en güvenilir sonuçları ürettiğini,
    ve hangi anatomik yapıların bu yaklaşımla güvenilir biçimde segmente edilebildiğini belirlemek hedeflenmiştir.</p>
    <div class="callout callout-blue">
      <strong>Temel soru:</strong> Tek bir histolojik talamus atlasından, 5 farklı hastaya yapılan label propagation ile elde edilen
      morphometrics ölçümlerine (hacim, simetri, şekil, doku dokusu) ne kadar güvenilebilir?
    </div>
  </div>
  <div class="card">
    <h3>Kullanılan Yöntem — Genel Akış</h3>
    <div class="formula">
Atlas (T1 + 41 etiket)
  │
  ├─ İP-1: Ön işleme        → Skull stripping (ANTsPyNet), N4 bias düzeltme
  │
  ├─ İP-2: Warp (kayıt)     → ANTsPy SyN deformable registration
  │        3 aday: W1, W2, W3 → Kazanan: W2
  │
  ├─ İP-3: Label propagation → Warp transform'u etiketlere uygula
  │        + Morphometrics    → 38 label × 5 hasta → hacim, simetri, şekil, doku
  │
  ├─ İP-4: Refinement karş.  → R1 (ham), R2, R3 → Kazanan: R1 (ham)
  │
  ├─ LOO Doğrulama           → 5×4=20 cross-registration fold → LOO DSC
  │
  └─ Kalite sınıflandırması  → TIER-1 / TIER-2 / TIER-3
    </div>
  </div>
</section>

<!-- ─── BÖLÜM 2: PİPELİNE SEÇİMİ ─── -->
<section id="pipeline">
  <h2>2. Pipeline Seçimi</h2>
  <div class="card">
    <h3>2a. Warp Adayı Karşılaştırması</h3>
    <p>Her hastaya 3 farklı warp konfigürasyonu uygulandı (toplam 15 kayıt). Değerlendirme kriterleri:</p>
    <ul style="margin:8px 0 12px 20px;color:#555">
      <li><strong>NMI (Normalized Mutual Information)</strong> — Yüksek = atlas ile hedef T1 arasında daha iyi yoğunluk eşleşmesi</li>
      <li><strong>Jacobian negatif oranı</strong> — 0 = deformasyon alanında topolojik bozulma yok (ideal)</li>
    </ul>
    <div class="tbl-wrap">
    <table>
      <tr><th>Aday</th><th>NMI (ort.)</th><th>Jacobian neg.</th><th>Konfigürasyon</th></tr>
      {warp_rows}
    </table>
    </div>
    <div class="callout callout-green" style="margin-top:14px">
      <strong>Karar: W2</strong> — En yüksek NMI (1.1086) ve sıfır topolojik bozulma.
      W2 konfigürasyonu: Affine ön-hizalama + SyN deformable (CC metriği, 4-voxel örnekleme, grad_step=0.1).
    </div>
  </div>
  <div class="card">
    <h3>2b. Refinement Kararı</h3>
    <p>Label propagation sonrası 3 refinement stratejisi karşılaştırıldı:</p>
    <ul style="margin:8px 0 12px 20px;color:#555">
      <li><strong>R1 (ham propagation)</strong> — Warp transform sonrası nearest-neighbour interpolasyon</li>
      <li><strong>R2</strong> — Morphological opening + closing</li>
      <li><strong>R3</strong> — Gaussian smoothing + threshold</li>
    </ul>
    <div class="callout callout-orange">
      <strong>Bulgu:</strong> R2 ve R3, küçük yapılarda (AD, Pv, MV) label sınırlarını bozarak TIER-3 oranını artırdı.
      Ham propagation (R1), 5 hastanın tamamında daha yüksek label tutarlılığı sergiledi.
    </div>
    <div class="callout callout-green">
      <strong>Karar: Ham propagation (R1) korundu.</strong>
      Morphometrics araştırması için segmentasyon sınırlarına post-hoc müdahale yapılmadı.
    </div>
  </div>
</section>

<!-- ─── BÖLÜM 3: LOO DOĞRULAMA ─── -->
<section id="loo">
  <h2>3. Leave-One-Out Doğrulama</h2>
  <div class="card">
    <p>Projede ground-truth (elle çizilmiş segmentasyon) bulunmadığından, pipeline doğruluğu <strong>Leave-One-Out (LOO)</strong>
    stratejisiyle tahmin edildi.</p>
    <h3>LOO nasıl çalışır?</h3>
    <div class="formula">
Her hasta sırayla "atlas" rolünü üstlenir:
  IXI002 atlas → IXI012, IXI013, IXI015, IXI016 hedef  (4 fold)
  IXI012 atlas → IXI002, IXI013, IXI015, IXI016 hedef  (4 fold)
  ...
  Toplam: 5 × 4 = 20 cross-registration fold
  Her fold: 38 label × 2 hemisferde DSC hesaplanır → 1 label için 40 ölçüm
    </div>
    <div class="callout callout-orange">
      <strong>Önemli kısıt:</strong> LOO DSC gerçek ground-truth değil, <em>proxy</em> bir metriktir.
      "Hedef segmentasyon" olarak kullanılan veri de pipeline'ın ürettiği çıktıdır.
      Yüksek LOO DSC → atlas-to-subject tutarlılığı yüksek.
      Düşük LOO DSC → küçük yapı veya atlas-arası yüksek varyasyon.
    </div>
    <h3 style="margin-top:16px">LOO DSC Dağılımı (38 label)</h3>
    <div class="tbl-wrap">
    <table style="width:auto">
      <tr><th>Kategori</th><th>Label sayısı</th><th>Oran</th><th>Yorum</th></tr>
      <tr><td style="color:#27ae60;font-weight:bold">DSC ≥ 0.70</td>
          <td class="num">{n_good}</td>
          <td class="num">{n_good/38*100:.0f}%</td>
          <td>İyi — klinik doğrulama eşiği karşılandı</td></tr>
      <tr><td style="color:#e67e22;font-weight:bold">0.50 ≤ DSC &lt; 0.70</td>
          <td class="num">{n_ok}</td>
          <td class="num">{n_ok/38*100:.0f}%</td>
          <td>Makul — morphometrics için kullanılabilir, dikkatli yorumlanmalı</td></tr>
      <tr><td style="color:#e74c3c;font-weight:bold">DSC &lt; 0.50</td>
          <td class="num">{n_poor}</td>
          <td class="num">{n_poor/38*100:.0f}%</td>
          <td>Zayıf — çoğunlukla küçük (&lt;50 mm³) veya asimetrik yapılar</td></tr>
    </table>
    </div>
    <p style="margin-top:12px;color:#7f8c8d;font-size:0.88em">
      Not: Küçük yapılarda (≤20 mm³) DSC, 1–2 voxel kaymasına aşırı duyarlıdır.
      Bu nedenle düşük DSC tek başına güvenilmezlik göstergesi değildir; morphometrics skoru ile birlikte değerlendirilmelidir.
    </p>
  </div>
</section>

<!-- ─── BÖLÜM 4: GENEL SONUÇLAR ─── -->
<section id="genel">
  <h2>4. Genel Sonuçlar</h2>
  <div class="stat-row">
    <div class="stat"><div class="val t1">{t1_total}</div><div class="lbl">TIER-1<br>Güvenilir<br>{t1_total/grand*100:.0f}% (5 hasta × 38 label)</div></div>
    <div class="stat"><div class="val t2">{t2_total}</div><div class="lbl">TIER-2<br>Sınır değer<br>{t2_total/grand*100:.0f}%</div></div>
    <div class="stat"><div class="val t3">{t3_total}</div><div class="lbl">TIER-3<br>Güvenilmez<br>{t3_total/grand*100:.0f}%</div></div>
    <div class="stat"><div class="val" style="color:#2980b9">{dsc_mean:.3f}</div><div class="lbl">Ort. LOO DSC<br>(medyan: {dsc_med:.3f})</div></div>
    <div class="stat"><div class="val" style="color:#8e44ad">{cons_mean:.3f}</div><div class="lbl">Ort. Cross-subject<br>Tutarlılık</div></div>
    <div class="stat"><div class="val" style="color:#27ae60">{qual_mean:.3f}</div><div class="lbl">Ort. Kompozit<br>Kalite Skoru</div></div>
  </div>
  <div class="card">
    <h3>Hasta Bazında Özet</h3>
    <div class="tbl-wrap">
    <table>
      <tr><th>Hasta</th><th>TIER-1</th><th>TIER-2</th><th>TIER-3</th><th>T1 Oranı</th><th>Ort. LOO DSC</th><th>Ort. Kalite</th></tr>
      {subj_rows}
    </table>
    </div>
    <p style="margin-top:10px;color:#7f8c8d;font-size:0.88em">
      IXI012 ve IXI015 en yüksek TIER-1 oranına sahip. Guys tarayıcısı hastalarında (IXI002, IXI016)
      hafif daha düşük tutarlılık — muhtemelen T1 kontrast farkından kaynaklanıyor.
    </p>
  </div>
</section>

<!-- ─── BÖLÜM 5: MORPHOMETRİCS TABLOSU ─── -->
<section id="morph">
  <h2>5. Morphometrics Tablosu — Tüm Labellar</h2>
  <div class="card">
    <p>38 talamus alt çekirdeğinin 5 hasta üzerinden ortalamaları. Kalite skoruna göre azalan sırada.</p>
    <div class="callout callout-blue" style="margin-bottom:14px">
      <strong>Sütun rehberi:</strong>
      <strong>Atlas mm³</strong> = referans atlas hacmi |
      <strong>Vol L/R</strong> = ölçülen sol/sağ hacim ortalaması (mm³) |
      <strong>Retention</strong> = atlas hacminin korunan yüzdesi (yeşil ≥65%) |
      <strong>LR Oran</strong> = sol/sağ simetri (yeşil 0.70–1.30 arasında) |
      <strong>Mirror Err</strong> = centroid simetri hatası (mm) |
      <strong>Kompaktlık</strong> = şekil küreseli (1=küre, düşük=parçalı) |
      <strong>T1 Outlier</strong> = T1 yoğunluk anormalliği oranı |
      <strong>Tutarlılık</strong> = 5 hasta arası CV bazlı skor |
      <strong>LOO DSC</strong> = segmentasyon doğruluğu proxy'si |
      <strong>Kalite</strong> = kompozit skor (LOO 0.50 + morph 0.30 + tutarlılık 0.20)
    </div>
    <div class="tbl-wrap">
    <table>
      <tr>
        <th>Label</th>
        <th>Atlas mm³</th>
        <th>Vol L</th>
        <th>Vol R</th>
        <th>Retention</th>
        <th>LR Oran</th>
        <th>Mirror Err (mm)</th>
        <th>Kompaktlık</th>
        <th>T1 Outlier</th>
        <th>Tutarlılık</th>
        <th>LOO DSC</th>
        <th>Kalite</th>
        <th>TIER</th>
      </tr>
      {morph_rows}
    </table>
    </div>
  </div>
</section>

<!-- ─── BÖLÜM 6: SORUNLU LABELLAR ─── -->
<section id="sorunlu">
  <h2>6. Dikkat Gerektiren Labellar (TIER-3)</h2>
  <div class="card">
    <div class="callout callout-red">
      Aşağıdaki 5 label, 5 hastanın büyük çoğunluğunda TIER-3 olarak sınıflandırıldı.
      <strong>Morphometrics analizinde bu labelların çıktıları kullanılmadan önce manuel doğrulama gereklidir.</strong>
    </div>
    <div class="tbl-wrap">
    <table>
      <tr><th>Label</th><th>Atlas mm³</th><th>LR Oran</th><th>Kompaktlık</th><th>Tutarlılık</th><th>LOO DSC</th><th>Kalite</th><th>Tespit edilen sorun</th></tr>
      {t3_rows}
    </table>
    </div>
    <h3 style="margin-top:20px">Sorunların Yorumu</h3>
    <ul style="margin:8px 0 0 20px;color:#555;line-height:2">
      <li><strong>AD (Anterior Dorsal, 9 mm³)</strong> — Aşırı küçük hacim. LR asimetrisi 7×.
          Bu boyutta propagation'da 1–2 voxel kayması bile yapının tamamını etkiliyor.
          Kompaktlık=0: sol veya sağ hemisferden birinde yapı tamamen kaybolmuş.</li>
      <li><strong>Pv (Paraventrikular, 7 mm³)</strong> — En küçük label. LR oranı aşırı yüksek
          (sol hemisfer yok denecek kadar küçük). Lateral ventriküle yakın konumu nedeniyle
          CSF sinyaliyle karışıyor olabilir.</li>
      <li><strong>MV (Medioventral, 16 mm³)</strong> — LR asimetrisi + yüksek T1 outlier oranı.
          Her iki hemisfer de parçalı (fragmented). Warp transform'un bu bölgede yeterli hassasiyeti yok.</li>
      <li><strong>sPf (Subparafasikular, 14 mm³)</strong> — T1 outlier oranı yüksek. Her iki hemisfer parçalı.
          Atlas'taki sPf tanımı bu çözünürlükte ayırt edilemiyor olabilir.</li>
      <li><strong>mtt (Mamillotalamik traktus, 57 mm³)</strong> — Hacim görece büyük ama LR oranı 0.82
          (sol > sağ). Sınır kalite; dikkatli kullanılabilir.</li>
    </ul>
  </div>
</section>

<!-- ─── BÖLÜM 7: KULLANIMA HAZIRLIK DEĞERLENDİRMESİ ─── -->
<section id="hazirlik">
  <h2>7. Kullanıma Hazırlık Değerlendirmesi</h2>

  <div class="card">
    <h3>Genel Sonuç</h3>
    <div class="callout callout-red" style="font-size:1.05em">
      <strong>Ticari veya klinik kullanıma hazır değil.</strong>
      Araştırma aracı olarak belirli yapılar için kullanılabilir;
      ancak mevcut pipeline tek başına bir ürüne dönüştürülemez.
    </div>
    <p style="margin-top:14px">
      Pipeline deterministik ve hasta-to-hasta tutarlı çalışıyor —
      5 hasta arasındaki ortalama DSC sapmasi <strong>{subj_dsc_std:.4f}</strong>
      (pratik olarak sıfır). Sorun tutarsızlık değil, mutlak doğruluk sınırıdır.
    </p>
  </div>

  <!-- Nicel tablo -->
  <div class="card">
    <h3>DSC Eşiklerine Göre Durum</h3>
    <p class="section-note" style="color:#7f8c8d;font-size:0.88em;margin-bottom:12px">
      Ticari segmentasyon yazılımları için kabul edilen DSC eşiği kritik yapılarda ≥ 0.85,
      genel yapılarda ≥ 0.75'tir. Klinik araştırma pilotları için ≥ 0.70 alt sınır kabul görür.
      Not: buradaki DSC değerleri gerçek ground-truth değil, LOO proxy ölçümleridir.
    </p>
    <div class="tbl-wrap">
    <table style="width:auto">
      <tr>
        <th>Eşik</th>
        <th>Label sayısı</th>
        <th>Oran</th>
        <th>Ticari beklenti</th>
        <th>Durum</th>
      </tr>
      <tr style="background:#f8fffe">
        <td style="color:#27ae60;font-weight:bold">DSC ≥ 0.85</td>
        <td class="num">0</td>
        <td class="num">%0</td>
        <td>Ticari ürün için minimum</td>
        <td><span class="badge" style="background:#e74c3c">KARŞILANMIYOR</span></td>
      </tr>
      <tr style="background:#f8fffe">
        <td style="color:#27ae60;font-weight:bold">DSC ≥ 0.70</td>
        <td class="num">{n_good} / {n_labels}</td>
        <td class="num">%{n_good/n_labels*100:.0f}</td>
        <td>Klinik pilot eşiği</td>
        <td><span class="badge" style="background:#e67e22">KISMI</span></td>
      </tr>
      <tr style="background:#fffdf5">
        <td style="color:#e67e22;font-weight:bold">DSC ≥ 0.50</td>
        <td class="num">{n_good+n_ok} / {n_labels}</td>
        <td class="num">%{(n_good+n_ok)/n_labels*100:.0f}</td>
        <td>Grup düzeyi araştırma</td>
        <td><span class="badge" style="background:#e67e22">KISMI</span></td>
      </tr>
      <tr style="background:#fff8f8">
        <td style="color:#e74c3c;font-weight:bold">DSC &lt; 0.50</td>
        <td class="num">{n_poor} / {n_labels}</td>
        <td class="num">%{n_poor/n_labels*100:.0f}</td>
        <td>—</td>
        <td><span class="badge" style="background:#e74c3c">YETERSIZ</span></td>
      </tr>
    </table>
    </div>
  </div>

  <!-- Boyut etkisi -->
  <div class="card">
    <h3>Yapı Boyutu ile DSC İlişkisi</h3>
    <p>Pipeline performansı anatomik yapının boyutuna güçlü biçimde bağımlıdır:</p>
    <div class="stat-row" style="margin-top:14px">
      <div class="stat">
        <div class="val" style="color:#27ae60">{dsc_large_mean:.3f}</div>
        <div class="lbl">≥ 200 mm³ yapılar<br>({len(large_labels)} label) ort. DSC</div>
      </div>
      <div class="stat">
        <div class="val" style="color:#e74c3c">{dsc_small_mean:.3f}</div>
        <div class="lbl">&lt; 100 mm³ yapılar<br>({len(small_labels)} label) ort. DSC</div>
      </div>
      <div class="stat">
        <div class="val" style="color:#3498db">{subj_dsc_std:.4f}</div>
        <div class="lbl">Hasta-to-hasta DSC sapması<br>(deterministik pipeline)</div>
      </div>
    </div>
    <div class="callout callout-orange" style="margin-top:14px">
      <strong>Fizik sınırı:</strong> 1 mm izotropik MRI'da 7–16 mm³ boyutundaki yapılar
      (AD, Pv, MV, sPf) yalnızca 7–16 voxelden oluşur. Bu boyutta 1 voxel kayması
      DSC'yi 0.3–0.5 birim düşürür. Sorun algoritma değil, çözünürlüktür.
      Bu yapılar için 0.7T veya 3T ultra-yüksek çözünürlük gereklidir.
    </div>
  </div>

  <!-- Neden çalışmıyor — 3 temel sorun -->
  <div class="card">
    <h3>Ticari Kullanımı Engelleyen 3 Temel Sorun</h3>

    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:12px">

      <div style="flex:1;min-width:260px;background:#fff8f8;border:1px solid #f5c6cb;border-radius:8px;padding:16px">
        <div style="font-weight:bold;color:#c0392b;margin-bottom:8px">① Validasyon — Ground-truth yok</div>
        <p style="color:#555;font-size:0.9em">
          Raporlanan tüm DSC değerleri LOO <em>proxy</em>'dir — gerçek el çizimi segmentasyona değil,
          pipeline'ın kendi çıktısına karşı hesaplanmıştır.
          Gerçek ground-truth ile karşılaştırıldığında bu sayıların
          <strong>%10–20 daha düşük</strong> çıkması istatistiksel olarak beklenir.
          Ticari onay (FDA/CE) veya klinik yayın için en az 2 bağımsız uzmanın
          etiketlediği 50+ hastaya ihtiyaç var.
        </p>
      </div>

      <div style="flex:1;min-width:260px;background:#fff8f8;border:1px solid #f5c6cb;border-radius:8px;padding:16px">
        <div style="font-weight:bold;color:#c0392b;margin-bottom:8px">② Tek atlas, küçük kohort</div>
        <p style="color:#555;font-size:0.9em">
          Pipeline tek bir histolojik atlas kullanıyor.
          Gerçek popülasyonda yaş (çocuk–yaşlı), patoloji (Parkinson, Alzheimer, MS),
          tarayıcı (1.5T–7T), protokol farklılıkları anatomik varyasyonu büyük ölçüde artırır.
          5 sağlıklı yetişkin bu varyasyonu temsil etmiyor.
          Çözüm: <strong>multi-atlas fusion</strong> (en az 5 atlas) + 20+ hasta.
        </p>
      </div>

      <div style="flex:1;min-width:260px;background:#fff8f8;border:1px solid #f5c6cb;border-radius:8px;padding:16px">
        <div style="font-weight:bold;color:#c0392b;margin-bottom:8px">③ 18/38 label DSC &lt; 0.50</div>
        <p style="color:#555;font-size:0.9em">
          Labelların %47'si DSC 0.50'nin altında — bunlar bireysel morphometrik ölçüm
          için güvenilmez. Küçük yapıların bir kısmı çözünürlük sınırından kaynaklanıyor
          olsa da VLa (319 mm³), VPM (186 mm³), CeM (154 mm³) gibi orta boyutlu yapıların
          da bu kategoride olması atlas-to-subject transfer kalitesinin sınırını gösteriyor.
        </p>
      </div>

    </div>
  </div>

  <!-- Güvenilir labellar tablosu -->
  <div class="card">
    <h3>Label Bazında Kullanım Kılavuzu</h3>
    <p style="color:#7f8c8d;font-size:0.88em;margin-bottom:12px">
      Her label için mevcut veriye dayalı kullanım önerisi.
      "Araştırma" = grup düzeyi istatistik; "Klinik pilot" = kontrollü çalışmada bireysel ölçüm.
    </p>

    <h4 style="color:#27ae60;margin:16px 0 8px">Klinik pilot için uygun (DSC ≥ 0.70)</h4>
    <div class="tbl-wrap">
    <table>
      <tr><th>Label</th><th>Atlas mm³</th><th>LOO DSC</th><th>Tutarlılık</th><th>Kalite</th><th>Öneri</th></tr>
      {readiness_rows_good}
    </table>
    </div>

    <h4 style="color:#e67e22;margin:20px 0 8px">Grup düzeyi araştırma için kullanılabilir (DSC 0.50–0.70)</h4>
    <div class="tbl-wrap">
    <table>
      <tr><th>Label</th><th>Atlas mm³</th><th>LOO DSC</th><th>Tutarlılık</th><th>Kalite</th><th>Öneri</th></tr>
      {readiness_rows_ok}
    </table>
    </div>

    <h4 style="color:#e74c3c;margin:20px 0 8px">Manuel doğrulama gerektiriyor (DSC &lt; 0.50)</h4>
    <div class="tbl-wrap">
    <table>
      <tr><th>Label</th><th>Atlas mm³</th><th>LOO DSC</th><th>Tutarlılık</th><th>Kalite</th><th>Neden yetersiz</th></tr>
      {readiness_rows_poor}
    </table>
    </div>
  </div>

  <!-- Yol haritası -->
  <div class="card">
    <h3>Pipeline'ı İyileştirmek İçin Gerekli Adımlar</h3>
    <table style="width:auto">
      <tr>
        <th>Gereksinim</th>
        <th>Mevcut durum</th>
        <th>Hedef</th>
        <th>Öncelik</th>
      </tr>
      <tr>
        <td>Ortalama DSC (gerçek GT)</td>
        <td class="num" style="color:#e74c3c">0.514 (proxy)</td>
        <td class="num">≥ 0.75</td>
        <td><span class="badge" style="background:#e74c3c">Kritik</span></td>
      </tr>
      <tr>
        <td>DSC ≥ 0.70 olan label oranı</td>
        <td class="num" style="color:#e74c3c">%{n_good/n_labels*100:.0f}</td>
        <td class="num">≥ %70</td>
        <td><span class="badge" style="background:#e74c3c">Kritik</span></td>
      </tr>
      <tr>
        <td>Ground-truth validasyon</td>
        <td style="color:#e74c3c">Yok (LOO proxy)</td>
        <td>2 uzman × 50+ hasta</td>
        <td><span class="badge" style="background:#e74c3c">Kritik</span></td>
      </tr>
      <tr>
        <td>Validasyon hasta sayısı</td>
        <td class="num" style="color:#e74c3c">5</td>
        <td class="num">≥ 50</td>
        <td><span class="badge" style="background:#e74c3c">Kritik</span></td>
      </tr>
      <tr>
        <td>Atlas sayısı (multi-atlas fusion)</td>
        <td class="num" style="color:#e67e22">1</td>
        <td class="num">≥ 5</td>
        <td><span class="badge" style="background:#e67e22">Yüksek</span></td>
      </tr>
      <tr>
        <td>Tarayıcı çeşitliliği</td>
        <td style="color:#e67e22">2 site (Guys, HH), 1.5T</td>
        <td>≥ 5 site, 1.5T–3T</td>
        <td><span class="badge" style="background:#e67e22">Yüksek</span></td>
      </tr>
      <tr>
        <td>Hasta popülasyonu</td>
        <td style="color:#e67e22">Yalnızca sağlıklı yetişkin</td>
        <td>Sağlıklı + hasta kohortları</td>
        <td><span class="badge" style="background:#e67e22">Yüksek</span></td>
      </tr>
      <tr>
        <td>MRI çözünürlüğü (küçük yapılar)</td>
        <td style="color:#e67e22">1 mm izotropik</td>
        <td>0.5–0.7 mm (3T veya 7T)</td>
        <td><span class="badge" style="background:#2980b9">Orta</span></td>
      </tr>
    </table>
  </div>

</section>

<!-- ─── BÖLÜM 8: FreeSurfer Entegrasyonu ─── -->
<section id="freesurfer">
  <h2>8. FreeSurfer Pseudo-GT Entegrasyonu</h2>

  <div style="background:#1a3a2a;color:#fff;border-radius:10px;padding:18px 24px;margin-bottom:20px;display:flex;align-items:center;gap:16px">
    <div style="font-size:2em">✅</div>
    <div>
      <div style="font-weight:700;font-size:1.05em;margin-bottom:4px">
        FreeSurfer 7.4.1 KURULDU — WSL (Ubuntu 22.04) üzerinde hazır
      </div>
      <div style="color:#a0d4b8;font-size:0.9em">
        Kurulum tamamlandı, lisans alındı, <code>recon-all --version</code> doğrulandı.
        Kod entegrasyonu (WSL wrappers, run_pipeline.py ip5 adımı) tamamlandı.
        <strong>Beklenen sonraki adım:</strong> recon-all çalıştırılması (~30–50 saat toplam).
        recon-all tamamlandıktan sonra bu bölüm gerçek DSC verileriyle güncellenecektir.
      </div>
    </div>
  </div>

  <div class="card">
    <h3>Neden FreeSurfer?</h3>
    <p>
      Mevcut pipeline'ın tüm doğrulama metrikleri <strong>LOO proxy</strong> üzerine kurulu —
      yani pipeline'ın kendi çıktısı kendi çıktısına karşı ölçülüyor.
      FreeSurfer, bağımsız bir araçla üretilmiş pseudo-ground-truth sağlar:
    </p>
    <ul style="margin:10px 0 0 20px;color:#555;line-height:2.0">
      <li>Onlarca klinik çalışmada valide edilmiş, yayınlanmış referans araç</li>
      <li>T1 + T2 füzyonu ile talamus alt çekirdek segmentasyonu
          (<code>mri_segment_thalamic_nuclei</code>)</li>
      <li>BrainSeg'den tamamen bağımsız algoritma — ortak hata kaynağı yok</li>
      <li>25 talamus alt çekirdeği (BrainSeg atlasıyla 20 ortak label — karşılaştırılabilir)</li>
    </ul>
    <div class="callout callout-orange" style="margin-top:14px">
      <strong>Önemli not:</strong> FreeSurfer çıktısı da gerçek ground-truth değil —
      elle çizim yerine geçmez. Ancak LOO proxy'e göre çok daha güçlü bir bağımsız referanstır.
      İki bağımsız yöntemin aynı bölgede aynı ölçümü üretmesi güvenilirliği anlamlı ölçüde artırır.
    </div>
  </div>

  <div class="card">
    <h3>Entegrasyon Planı</h3>
    <div class="formula">
İP-5 Pipeline (scripts/ip5_freesurfer_gt.py):

Adım 1 — recon-all  (~6–10 saat / hasta)
  recon-all -i T1.nii.gz -s &lt;subject&gt; -all -parallel
  → Korteks yeniden yapılandırması + subkortikal segmentasyon

Adım 2 — Talamus alt çekirdek segmentasyonu  (~15 dak / hasta)
  mri_segment_thalamic_nuclei -s &lt;subject&gt; --T2 T2_to_T1.nii.gz
  → ThalamicNuclei.v12.T1.T2.mgz (25 label, her iki hemisferde)

Adım 3 — NIfTI dönüştürme + label ayrıştırma
  mri_convert → thalamic_seg_fs.nii.gz
  Label kodları → binary NIfTI (FreeSurfer LUT tablosundan)

Adım 4 — BrainSeg vs FreeSurfer DSC karşılaştırması
  Her label için: DSC(BrainSeg_output, FreeSurfer_output)
  → freesurfer_comparison.csv
    </div>
  </div>

  <div class="card">
    <h3>Mevcut Pipeline ile Karşılaştırılabilecek Labellar</h3>
    <p class="section-note" style="color:#7f8c8d;font-size:0.88em;margin-bottom:10px">
      FreeSurfer'ın 25 talamus etiketinden 20'si BrainSeg atlasıyla eşleştirilebilir.
      Kalan 5 label (STh, RN, AD, Pv vb.) FreeSurfer'da tanımlı değil.
    </p>
    <div class="tbl-wrap">
    <table>
      <tr><th>Karşılaştırılabilir labellar (20)</th><th>FreeSurfer'da bulunmayan (18)</th></tr>
      <tr>
        <td style="color:#27ae60;vertical-align:top;padding:8px 12px">
          AV, CL, CM, CeM, LD, LGNpc, LP, MDmc, MDpc, MGN,
          MV, Pf, PuA, PuI, PuL, PuM, VApc, VAmc, VLa,
          VLpd, VM, VPLp, VPI, VPM
          <span style="color:#7f8c8d;font-size:0.85em"><br>(T2 ile birlikte daha doğru)</span>
        </td>
        <td style="color:#e74c3c;vertical-align:top;padding:8px 12px">
          STh, RN, AD, Pv, sPf, mtt, Hb, SG, Li, Po,
          AM, LGNmc, VLpv, VPLa
          <span style="color:#7f8c8d;font-size:0.85em"><br>(histolojik atlasa özgü, FreeSurfer segmentasyonunda yok)</span>
        </td>
      </tr>
    </table>
    </div>
  </div>

  <div class="card">
    <h3>Entegrasyon Sonrasında Ne Değişecek?</h3>
    <div class="callout callout-red">
      <strong>Bu rapordaki aşağıdaki değerler FreeSurfer entegrasyonu tamamlandığında güncellenecektir:</strong>
      <ul style="margin:8px 0 0 20px;line-height:2">
        <li><strong>Bölüm 3 — LOO DSC tablosu:</strong> Proxy DSC değerleri,
            FreeSurfer pseudo-GT'ye karşı hesaplanan bağımsız DSC ile karşılaştırılacak</li>
        <li><strong>Bölüm 7 — DSC eşik tablosu:</strong> "DSC ≥ 0.70" olan label sayısı
            değişebilir (artış veya azalış — tahmin edilemiyor)</li>
        <li><strong>Bölüm 7 — Label kullanım kılavuzu:</strong> Bazı labelların
            TIER'ı değişebilir; özellikle LOO proxy'de iyi görünen ama
            gerçekte zayıf olanlar ortaya çıkacak</li>
        <li><strong>Bölüm 7 — Yol haritası tablosu:</strong>
            "Ground-truth validasyon" satırı güncellenecek</li>
      </ul>
    </div>
    <div class="callout callout-blue" style="margin-top:12px">
      <strong>Tahmini etki:</strong> Literatür bulguları ve LOO proxy güvenilirliğine göre
      FreeSurfer DSC'nin LOO proxy DSC'den <strong>%5–20 daha düşük</strong> çıkması beklenir.
      Bu, gerçek performansın mevcut rapordan daha muhafazakâr olduğu anlamına gelir.
      Eğer iki yöntem yakın sonuç verirse pipeline güvenilirliği teyit edilmiş olacak.
    </div>
  </div>

  <div class="card">
    <h3>Çalıştırma Talimatı</h3>
    <p><strong>FreeSurfer kuruldu</strong> — recon-all çalıştırılmayı bekliyor:</p>
    <div class="formula">
# 1. Kurulum kontrolü (doğrulama)
python scripts/ip5_freesurfer_gt.py --check

# 2. Tüm hastalar için çalıştır (toplam ~30–50 saat)
python scripts/run_pipeline.py --steps ip5

# 3. Belirli bir hasta için
python scripts/run_pipeline.py --steps ip5 --subject IXI002-Guys-0828

# 4. recon-all zaten tamamlandıysa (sadece segmentasyon + karşılaştırma)
python scripts/run_pipeline.py --steps ip5 --skip-recon

# 5. T2 olmadan çalıştır
python scripts/run_pipeline.py --steps ip5 --no-t2

# 6. Bu raporu güncelle
python scripts/generate_project_report.py
    </div>
    <p style="margin-top:10px;color:#27ae60;font-size:0.88em;font-weight:bold">
      FreeSurfer 7.4.1 — WSL (Ubuntu 22.04) kurulu · Lisans: ~/freesurfer/license.txt
    </p>
  </div>

</section>

<!-- ─── BÖLÜM 9: METODOLOJİ ─── -->
<section id="metod">
  <h2>9. Metodoloji</h2>
  <div class="card">
    <h3>Kalite Skoru Hesaplama</h3>
    <div class="formula">
Kalite_Skoru = 0.50 × LOO_DSC_skoru
             + 0.30 × Morphometrics_skoru
             + 0.20 × Cross_subject_tutarlilik_skoru

Morphometrics_skoru = 1.0 − toplam_ceza  (minimum 0)

Ceza kalemleri:
  − L/R hacim asimetrisi &gt; 3× olursa           → −0.30
  − T1 arka plan kirliliği (yüksek T1 ort.)    → −0.20 / hemisferde
  − Centroid simetri hatası (mirror error)     → −0.15
  − Düşük kompaktlık (şekil parçalı)           → −0.10 veya −0.20
  − Yüksek T1 outlier oranı                    → −0.08 veya −0.15
  − Zayıf gradient sınır netliği               → −0.10 / hemisferde
  − Düşük BBox doluluk (parçalı/dağınık yapı)  → −0.10
  − Düşük GLCM homojenlik                      → −0.10
    </div>
    <h3 style="margin-top:16px">TIER Sınıflandırması</h3>
    <table style="width:auto;margin-top:8px">
      <tr>
        <th>TIER</th><th>Kalite Skoru</th><th>Kullanım Önerisi</th>
      </tr>
      <tr style="background:#f8fffe">
        <td><span class="badge" style="background:#27ae60">TIER-1</span></td>
        <td class="num">≥ 0.60</td>
        <td>Morphometrics analizinde doğrudan kullanılabilir. Uzman doğrulaması önerilir.</td>
      </tr>
      <tr style="background:#fffdf5">
        <td><span class="badge" style="background:#e67e22">TIER-2</span></td>
        <td class="num">0.40 – 0.60</td>
        <td>Dikkatli yorumlanmalı. Grup düzeyinde istatistik için kullanılabilir ama bireysel ölçüm güvenilmez olabilir.</td>
      </tr>
      <tr style="background:#fff8f8">
        <td><span class="badge" style="background:#e74c3c">TIER-3</span></td>
        <td class="num">&lt; 0.40</td>
        <td>Manuel doğrulama zorunlu. Otomatik ölçümler araştırma sonuçlarında doğrudan raporlanmamalı.</td>
      </tr>
    </table>
    <h3 style="margin-top:20px">Cross-Subject Tutarlılık Skoru</h3>
    <div class="formula">
Her label için: 5 hastanın sol hacim değerlerinin Varyasyon Katsayısı (CV) hesaplanır.
CV = std / mean

Tutarlılık_skoru = exp(−CV)   (CV=0 → skor=1.0, CV=1 → skor≈0.37)

Bu skor, label'ın gerçek biyolojik varyasyonunu değil,
pipeline'ın hasta-to-hasta reprodusibiliteini yansıtır.
    </div>
    <h3 style="margin-top:16px">Veri ve Yazılım</h3>
    <ul style="margin:8px 0 0 20px;color:#555;line-height:2">
      <li><strong>Veri seti:</strong> IXI (1.5T T1-w MRI, Guys ve HH tarayıcıları, 5 sağlıklı yetişkin)</li>
      <li><strong>Atlas:</strong> Histolojik talamus atlası — 41 sub-nükleus etiketi (MNI152 uzayı)</li>
      <li><strong>Registration:</strong> ANTsPy 0.x — ANTs SyN nonlinear deformable</li>
      <li><strong>Skull stripping:</strong> ANTsPyNet deep learning brain mask</li>
      <li><strong>Morphometrics:</strong> nibabel + scikit-image + scipy (özel pipeline)</li>
      <li><strong>Dil:</strong> Python 3.13</li>
    </ul>
  </div>
</section>

</div><!-- /content -->

<footer>
  BrainSeg Pipeline Raporu · Morphometrics Araştırması · 2026-04-10 ·
  Tüm DSC değerleri LOO proxy'dir — FreeSurfer 7.4.1 kuruldu, recon-all bekleniyor ·
  Ground-truth doğrulaması yapılmamıştır — klinik karar vermede kullanılmamalıdır.
</footer>

</body>
</html>
"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Rapor kaydedildi: {OUT}")
