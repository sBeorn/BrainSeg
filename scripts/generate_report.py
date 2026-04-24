"""
Thalamic Nuclei Morphometrics — HTML Rapor Uretici
====================================================
outputs/morphometrics/morphometrics_all.csv ve
outputs/quality/quality_report.csv verilerinden
kapsamli bir HTML raporu uretir.

Kullanim:
    python scripts/generate_report.py

Cikti:
    outputs/report.html
"""

import csv
import os
from collections import defaultdict
from datetime import date

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MORPH_CSV   = os.path.join(PROJECT_DIR, "outputs", "morphometrics", "morphometrics_all.csv")
QUAL_CSV    = os.path.join(PROJECT_DIR, "outputs", "quality", "quality_report.csv")
SSM_DIR     = os.path.join(PROJECT_DIR, "outputs", "ssm")
OUT_HTML    = os.path.join(PROJECT_DIR, "outputs", "report.html")

# ─────────────────────────────────────────────────────────────────────────────
# Veri Yukle
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

def fv(v, decimals=2):
    """Float formatla, bos veya None ise tire don."""
    try:
        return f"{float(v):.{decimals}f}" if v not in ("", "None", None) else "—"
    except (ValueError, TypeError):
        return "—"

def color_vpi(tier):
    return {"IYI": "#d4edda", "ORTA": "#fff3cd", "ZAYIF": "#f8d7da"}.get(tier, "#fff")

def color_sym(tier):
    return {"SIMETRIK": "#d4edda", "HAFIF_ASIM": "#fff3cd", "ASIMETRIK": "#f8d7da"}.get(tier, "#fff")

def color_cv(cv_str):
    try:
        cv = float(cv_str)
        if cv < 0.15: return "#d4edda"
        if cv < 0.25: return "#fff3cd"
        return "#f8d7da"
    except:
        return "#fff"

# ─────────────────────────────────────────────────────────────────────────────
# HTML Sablon
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px;
       background: #f5f7fa; color: #222; }
.wrap { max-width: 1300px; margin: 0 auto; padding: 24px 20px; }
h1 { font-size: 22px; color: #1a237e; margin-bottom: 4px; }
.subtitle { color: #555; font-size: 13px; margin-bottom: 28px; }
h2 { font-size: 16px; color: #1a237e; border-bottom: 2px solid #1a237e;
     padding-bottom: 5px; margin: 28px 0 14px; }
h3 { font-size: 14px; color: #333; margin: 18px 0 8px; }
.card { background: #fff; border-radius: 8px; box-shadow: 0 1px 4px #0001;
        padding: 18px 22px; margin-bottom: 18px; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr));
              gap: 14px; margin-bottom: 8px; }
.stat-box { background: #f0f4ff; border-radius: 6px; padding: 14px 18px; }
.stat-num { font-size: 28px; font-weight: bold; color: #1a237e; }
.stat-lbl { font-size: 11px; color: #666; margin-top: 2px; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th { background: #1a237e; color: #fff; padding: 7px 9px; text-align: left;
     white-space: nowrap; position: sticky; top: 0; }
td { padding: 5px 9px; border-bottom: 1px solid #eee; white-space: nowrap; }
tr:hover td { background: #f0f4ff !important; }
.tbl-wrap { overflow-x: auto; border-radius: 6px; box-shadow: 0 1px 3px #0001; }
.good  { background: #d4edda; }
.warn  { background: #fff3cd; }
.bad   { background: #f8d7da; }
.na    { background: #f5f5f5; color: #aaa; }
.pill  { display: inline-block; padding: 2px 8px; border-radius: 10px;
         font-size: 11px; font-weight: 600; }
.pill-good  { background: #c3e6cb; color: #155724; }
.pill-warn  { background: #ffeeba; color: #856404; }
.pill-bad   { background: #f5c6cb; color: #721c24; }
.pill-na    { background: #e2e3e5; color: #6c757d; }
.note { background: #fff8e1; border-left: 4px solid #ffc107;
        padding: 10px 14px; border-radius: 0 6px 6px 0; margin: 10px 0;
        font-size: 12px; color: #5d4037; }
.bar-wrap { display: flex; align-items: center; gap: 6px; }
.bar { height: 10px; border-radius: 3px; min-width: 2px; }
.section-desc { color: #555; font-size: 12px; margin-bottom: 12px; line-height: 1.6; }
footer { text-align: center; color: #aaa; font-size: 11px; margin-top: 32px; }
"""

def pill(text, kind):
    cls = {"IYI":"good","ORTA":"warn","ZAYIF":"bad",
           "SIMETRIK":"good","HAFIF_ASIM":"warn","ASIMETRIK":"bad"}.get(text, "na")
    return f'<span class="pill pill-{cls}">{text}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Bolumler
# ─────────────────────────────────────────────────────────────────────────────

def section_ozet(mrows, qrows):
    subjects = sorted(set(r["subject"] for r in mrows))
    labels   = sorted(set(r["label"]   for r in mrows))
    total    = len(mrows)
    iyi      = sum(1 for r in qrows if r["VPI_tier"] == "IYI")
    sim      = sum(1 for r in qrows if r["symmetry_tier"] == "SIMETRIK")
    pct_iyi  = 100 * iyi  / len(qrows) if qrows else 0
    pct_sim  = 100 * sim  / len(qrows) if qrows else 0

    # Ortalama hacim
    vols = [float(r["volume_mm3"]) for r in mrows if r["volume_mm3"]]
    avg_vol = sum(vols) / len(vols) if vols else 0

    html = '<div class="stats-grid">'
    for num, lbl in [
        (len(subjects), "Subject"),
        (len(labels),   "Thalamic Nucleus"),
        (total,         "Toplam Olcum"),
        (f"{pct_iyi:.0f}%", "VPI IYI (warp kalitesi)"),
        (f"{pct_sim:.0f}%", "Sol/Sag Simetrik"),
        (f"{avg_vol:.0f} mm³", "Ortalama Label Hacmi"),
    ]:
        html += f'<div class="stat-box"><div class="stat-num">{num}</div><div class="stat-lbl">{lbl}</div></div>'
    html += "</div>"

    html += "<h3>Subjectler</h3><p class='section-desc'>"
    html += " &nbsp;|&nbsp; ".join(subjects)
    html += "</p>"

    return html


def section_morphometrics(mrows):
    subjects = sorted(set(r["subject"] for r in mrows))
    labels   = sorted(set(r["label"]   for r in mrows))

    # Label bazi ortalamalar (sol+sag, tum subjectler)
    by_label = defaultdict(list)
    for r in mrows:
        by_label[r["label"]].append(r)

    html = '<p class="section-desc">Her label icin 5 subject ve sol+sag hemisfer ortalamalari. '
    html += 'Elongation/Flatness &lt; 30 voxel (&lt; ~20 mm³) olan labellar icin hesaplanamaz (—).</p>'
    html += '<div class="tbl-wrap"><table>'
    html += ("<tr><th>Label</th><th>Hacim L (mm³)</th><th>Hacim R (mm³)</th>"
             "<th>Yuzey (mm²)</th><th>Kompaktlik</th><th>Elongation</th>"
             "<th>Flatness</th><th>BBox X</th><th>BBox Y</th><th>BBox Z</th>"
             "<th>Doluluk</th><th>Bilesken</th><th>Iskelet (mm)</th></tr>")

    for label in labels:
        rows_l = [r for r in by_label[label] if r["side"] == "left"]
        rows_r = [r for r in by_label[label] if r["side"] == "right"]

        def avg(lst, key):
            vals = [float(r[key]) for r in lst if r.get(key) not in ("", "None", None)]
            return f"{sum(vals)/len(vals):.2f}" if vals else "—"

        vol_l = avg(rows_l, "volume_mm3")
        vol_r = avg(rows_r, "volume_mm3")
        sa    = avg(rows_l + rows_r, "surface_area_mm2")
        comp  = avg(rows_l + rows_r, "compactness")
        elo   = avg(rows_l + rows_r, "elongation")
        flat  = avg(rows_l + rows_r, "flatness")
        bx    = avg(rows_l + rows_r, "bbox_x_mm")
        by_   = avg(rows_l + rows_r, "bbox_y_mm")
        bz    = avg(rows_l + rows_r, "bbox_z_mm")
        fill  = avg(rows_l + rows_r, "bbox_fill_ratio")
        cc    = avg(rows_l + rows_r, "connected_components")
        skel  = avg(rows_l + rows_r, "skeleton_length_mm")

        # Kucuk label uyarisi
        all_vols = [float(r["volume_mm3"]) for r in rows_l + rows_r if r.get("volume_mm3")]
        small = all_vols and max(all_vols) < 20
        row_cls = ' class="warn"' if small else ""

        html += f"<tr{row_cls}>"
        html += f"<td><b>{label}</b></td>"
        html += f"<td>{vol_l}</td><td>{vol_r}</td><td>{sa}</td>"
        html += f"<td>{comp}</td><td>{elo}</td><td>{flat}</td>"
        html += f"<td>{bx}</td><td>{by_}</td><td>{bz}</td>"
        html += f"<td>{fill}</td><td>{cc}</td><td>{skel}</td>"
        html += "</tr>"

    html += "</table></div>"
    html += '<p class="note">Sari satir: Hacim &lt; 20 mm³ — AD, MV, Pv, sPf. Elongation/Flatness hesaplanamadi.</p>'
    return html


def section_subject_detail(mrows):
    subjects = sorted(set(r["subject"] for r in mrows))
    labels   = sorted(set(r["label"]   for r in mrows))
    by_subj_label = defaultdict(dict)
    for r in mrows:
        by_subj_label[(r["subject"], r["label"], r["side"])] = r

    html = '<p class="section-desc">Her subject icin sol/sag hacimler (mm³). '
    html += 'Subjectler arasi tutarlilik icin karsilastirabilirsiniz.</p>'
    html += '<div class="tbl-wrap"><table>'

    # Header
    html += "<tr><th>Label</th><th>Taraf</th>"
    for s in subjects:
        html += f"<th>{s.split('-')[0]}<br>{s.split('-')[1]}</th>"
    html += "<th>Ort</th><th>CV</th></tr>"

    import statistics
    for label in labels:
        for side in ("left", "right"):
            vals = []
            html += f"<tr><td><b>{label}</b></td><td>{'Sol' if side=='left' else 'Sag'}</td>"
            for subj in subjects:
                r = by_subj_label.get((subj, label, side))
                v = r["volume_mm3"] if r else ""
                try:
                    fval = float(v)
                    vals.append(fval)
                    html += f"<td>{fval:.1f}</td>"
                except:
                    html += "<td>—</td>"
            if len(vals) >= 2:
                mu = sum(vals)/len(vals)
                cv = statistics.stdev(vals)/mu if mu > 0 else 0
                cv_cls = "good" if cv < 0.15 else ("warn" if cv < 0.25 else "bad")
                html += f"<td>{mu:.1f}</td>"
                html += f'<td class="{cv_cls}">{cv:.3f}</td>'
            else:
                html += "<td>—</td><td>—</td>"
            html += "</tr>"

    html += "</table></div>"
    return html


def section_kalite(qrows):
    subjects = sorted(set(r["subject"] for r in qrows))
    labels   = sorted(set(r["label"]   for r in qrows))
    by_lbl   = defaultdict(list)
    for r in qrows:
        by_lbl[r["label"]].append(r)

    # Ozet istatistikler
    total = len(qrows)
    iyi   = sum(1 for r in qrows if r["VPI_tier"] == "IYI")
    orta  = sum(1 for r in qrows if r["VPI_tier"] == "ORTA")
    zayif = sum(1 for r in qrows if r["VPI_tier"] == "ZAYIF")
    sim   = sum(1 for r in qrows if r["symmetry_tier"] == "SIMETRIK")
    asim  = sum(1 for r in qrows if r["symmetry_tier"] == "ASIMETRIK")

    html = '<p class="section-desc">Hacim Koruma Indeksi (VPI = warped hacim / atlas hacmi). '
    html += '1.0 = atlas ile birebir eslesme. IYI: 0.70–1.30 | ORTA: 0.50–1.50 | ZAYIF: disarida.</p>'

    # Ozet kutular
    html += '<div class="stats-grid">'
    for num, lbl, cls in [
        (f"{100*iyi/total:.1f}%",  f"VPI IYI ({iyi}/{total})",  "good"),
        (f"{100*orta/total:.1f}%", f"VPI ORTA ({orta}/{total})", "warn"),
        (f"{100*zayif/total:.1f}%",f"VPI ZAYIF ({zayif}/{total})","bad"),
        (f"{100*sim/total:.1f}%",  f"Simetrik ({sim}/{total})",  "good"),
        (f"{100*asim/total:.1f}%", f"Asimetrik ({asim}/{total})","bad"),
    ]:
        bg = {"good":"#d4edda","warn":"#fff3cd","bad":"#f8d7da"}[cls]
        html += f'<div class="stat-box" style="background:{bg}"><div class="stat-num">{num}</div><div class="stat-lbl">{lbl}</div></div>'
    html += "</div>"

    # Label bazi kalite tablosu
    html += "<h3>Label Bazinda Kalite</h3>"
    html += '<div class="tbl-wrap"><table>'
    html += ("<tr><th>Label</th><th>Atlas L (mm³)</th><th>Atlas R (mm³)</th>"
             "<th>VPI Sol (ort)</th><th>VPI Sag (ort)</th><th>Tier</th>"
             "<th>L/R Oran (ort)</th><th>Simetri</th>"
             "<th>Cross-CV</th><th>Tutarlilik</th></tr>")

    for label in labels:
        rows = by_lbl[label]
        atl_l = rows[0]["atlas_vol_L_mm3"] if rows else ""
        atl_r = rows[0]["atlas_vol_R_mm3"] if rows else ""

        vpi_l_vals = [float(r["VPI_left"])  for r in rows if r["VPI_left"]  not in ("","None")]
        vpi_r_vals = [float(r["VPI_right"]) for r in rows if r["VPI_right"] not in ("","None")]
        lr_vals    = [float(r["LR_ratio"])  for r in rows if r["LR_ratio"]  not in ("","None")]
        cv_vals    = [float(r["cross_subject_CV"]) for r in rows if r["cross_subject_CV"] not in ("","None")]

        vpi_l_avg = sum(vpi_l_vals)/len(vpi_l_vals) if vpi_l_vals else None
        vpi_r_avg = sum(vpi_r_vals)/len(vpi_r_vals) if vpi_r_vals else None
        lr_avg    = sum(lr_vals)/len(lr_vals)        if lr_vals    else None
        cv_avg    = sum(cv_vals)/len(cv_vals)        if cv_vals    else None

        # Tier: cogunluk VPI tier'i al
        tiers = [r["VPI_tier"] for r in rows if r["VPI_tier"]]
        tier  = max(set(tiers), key=tiers.count) if tiers else "N/A"
        syms  = [r["symmetry_tier"] for r in rows if r["symmetry_tier"]]
        sym   = max(set(syms), key=syms.count) if syms else "N/A"

        vpi_l_str = f"{vpi_l_avg:.2f}" if vpi_l_avg else "—"
        vpi_r_str = f"{vpi_r_avg:.2f}" if vpi_r_avg else "—"
        lr_str    = f"{lr_avg:.2f}"    if lr_avg    else "—"
        cv_str    = f"{cv_avg:.3f}"    if cv_avg    else "—"

        cv_cls = "good" if cv_avg and cv_avg < 0.15 else ("warn" if cv_avg and cv_avg < 0.25 else "bad") if cv_avg else ""
        cv_lbl = "Tutarli" if cv_avg and cv_avg < 0.15 else ("Orta" if cv_avg and cv_avg < 0.25 else "Tutarsiz") if cv_avg else "—"

        html += "<tr>"
        html += f"<td><b>{label}</b></td>"
        html += f"<td>{fv(atl_l)}</td><td>{fv(atl_r)}</td>"
        html += f'<td style="background:{color_vpi(tier)}">{vpi_l_str}</td>'
        html += f'<td style="background:{color_vpi(tier)}">{vpi_r_str}</td>'
        html += f'<td style="background:{color_vpi(tier)}">{pill(tier, "vpi")}</td>'
        html += f'<td>{lr_str}</td>'
        html += f'<td style="background:{color_sym(sym)}">{pill(sym, "sym")}</td>'
        html += f'<td class="{cv_cls}">{cv_str}</td>'
        html += f'<td class="{cv_cls}">{cv_lbl}</td>'
        html += "</tr>"

    html += "</table></div>"
    html += '<p class="note">VPI = warped_hacim / atlas_hacmi. 1.0 ideal. Kucuk yapilarda (AD, Pv, MV, sPf) '
    html += 'birkaç voxel farki bile buyuk VPI sapmasi uretir — istatistiksel sinirlilik.</p>'
    return html


def section_ssm():
    """SSM sonuçları bölümü — outputs/ssm/ klasöründen okunur."""
    import sys
    try:
        import numpy as np
    except ImportError:
        return "<p>numpy bulunamadı.</p>"

    if not os.path.isdir(SSM_DIR):
        return "<p class='note'>SSM verisi bulunamadı. Önce: <code>python scripts/compute_ssm.py</code></p>"

    npz_files = sorted(f for f in os.listdir(SSM_DIR) if f.endswith("_ssm.npz"))
    if not npz_files:
        return "<p class='note'>SSM .npz dosyası bulunamadı.</p>"

    rows = []
    for fname in npz_files:
        label = fname.replace("_ssm.npz", "")
        d = np.load(os.path.join(SSM_DIR, fname), allow_pickle=True)
        ev      = d["eigenvalues"]
        n_modes = int(d["n_modes"])
        tot     = float(np.sum(ev))
        pct     = [100.0 * float(ev[i]) / tot if tot > 0 else 0.0 for i in range(n_modes)]
        sigma1  = float(np.sqrt(max(float(ev[0]), 1e-12)))
        rows.append({
            "label":    label,
            "n_subj":   len(d["subjects"]),
            "n_modes":  n_modes,
            "pct":      pct,
            "sigma1":   sigma1,
        })

    # Özet istatistikler
    pc1_vals = [r["pct"][0] for r in rows]
    pc1_avg  = sum(pc1_vals) / len(pc1_vals)
    pc1_min  = min(pc1_vals)
    pc1_max  = max(pc1_vals)
    best_label  = rows[pc1_vals.index(pc1_max)]["label"]
    worst_label = rows[pc1_vals.index(pc1_min)]["label"]

    html = '<p class="section-desc">'
    html += 'Her label için yüzey landmark koordinatlarından (12 nokta × 3 boyut = 36 boyutlu şekil vektörü) '
    html += 'PCA ile şekil modeli oluşturuldu. PC1–PC4 modları şeklin ana varyasyon eksenlerini temsil eder. '
    html += '3D Slicer\'da <code>slicer_ssm_viewer.py</code> ile interaktif incelenebilir (slider → TPS deformasyonu).'
    html += '</p>'

    # Özet kutular
    html += '<div class="stats-grid">'
    for num, lbl in [
        (len(rows),                 "SSM Hesaplanan Label"),
        (f"{pc1_avg:.1f}%",         "PC1 Ort. Varyans"),
        (f"{pc1_max:.1f}%",         f"PC1 Max ({best_label})"),
        (f"{pc1_min:.1f}%",         f"PC1 Min ({worst_label})"),
        ("±3σ",                     "Slider Aralığı (viewer)"),
        ("TPS",                     "Deformasyon Yöntemi"),
    ]:
        html += f'<div class="stat-box"><div class="stat-num">{num}</div><div class="stat-lbl">{lbl}</div></div>'
    html += '</div>'

    # Label bazında SSM tablosu
    html += '<div class="tbl-wrap"><table>'
    html += ('<tr><th>Label</th><th>N Subj</th><th>N Mod</th>'
             '<th>PC1 Varyans</th><th>PC2 Varyans</th>'
             '<th>PC3 Varyans</th><th>PC4 Varyans</th>'
             '<th>σ₁ (mm)</th><th>Yorum</th></tr>')

    for r in rows:
        pct  = r["pct"]
        pc1  = pct[0] if len(pct) > 0 else 0
        pc2  = pct[1] if len(pct) > 1 else 0
        pc3  = pct[2] if len(pct) > 2 else 0
        pc4  = pct[3] if len(pct) > 3 else 0

        # PC1 renk: > 85% koyu yeşil, > 70% açık yeşil, altı sarı
        if pc1 >= 85:
            pc1_cls = "good"
            yorum   = "Basit varyasyon"
        elif pc1 >= 70:
            pc1_cls = ""
            yorum   = "Normal"
        else:
            pc1_cls = "warn"
            yorum   = "Çok boyutlu şekil"

        def bar(v):
            w = int(v * 1.2)
            return f'<div class="bar-wrap"><div class="bar" style="width:{w}px;background:#1a237e"></div>{v:.1f}%</div>'

        html += f'<tr>'
        html += f'<td><b>{r["label"]}</b></td>'
        html += f'<td>{r["n_subj"]}</td>'
        html += f'<td>{r["n_modes"]}</td>'
        html += f'<td class="{pc1_cls}">{bar(pc1)}</td>'
        html += f'<td>{bar(pc2)}</td>'
        html += f'<td>{pc3:.1f}%</td>'
        html += f'<td>{pc4:.1f}%</td>'
        html += f'<td>{r["sigma1"]:.2f}</td>'
        html += f'<td>{yorum}</td>'
        html += '</tr>'

    html += '</table></div>'
    html += '<p class="note">σ₁ = PC1\'in standart sapması (mm cinsinden landmark uzaklığı). '
    html += 'Yüksek σ₁ → subject arası şekil farkı büyük. '
    html += 'PC1 varyansı > %85 → şekil değişimi neredeyse tek bir eksen üzerinde.</p>'
    return html


def section_limitasyonlar():
    items = [
        ("Kucuk Label Sorunu",
         "AD, Pv, MV, sPf labels < 20 mm³ — 1mm izotropik cozunurlukte yeterli voxel yok. "
         "Elongation/Flatness hesaplanamadi. VPI degerleri guvenilmez."),
        ("Warp Hatasi Yayilimi",
         "Morfometrik olcumler warp sonrasi maskelere gore hesaplandi. "
         "Warp kaydi hataliysa metrikler de etkilenir. Kalite raporundaki VPI bu riski olcer."),
        ("Az Subject",
         "5 subject — istatistiksel guc sinirli. Ortalamalar ve CV degerleri referans niteligi tasir."),
        ("Yuzey Alani Hassasiyeti",
         "Marching cubes yontemi voxel cozunurluguyle sinirli. Kucuk yapilar icin yuzey alani "
         "hafif asiri tahmin edilebilir."),
        ("Iskelet Metrikleri",
         "skeleton_length_mm kucuk izole yapilar icin 0 donuyor. "
         "Buyuk yapilar (CL, MDpc, PuM) icin guvenilir."),
        ("Tek Atlas",
         "Tek bir MNI152 atlasi kullanildi. Label fusion uygulanmadi."),
    ]
    html = '<ul style="padding-left:20px; line-height:2;">'
    for title, desc in items:
        html += f"<li><b>{title}:</b> {desc}</li>"
    html += "</ul>"
    return html


def section_centroid_table(mrows):
    subjects = sorted(set(r["subject"] for r in mrows))
    labels   = sorted(set(r["label"]   for r in mrows))
    by_key   = {(r["subject"], r["label"], r["side"]): r for r in mrows}

    html = '<p class="section-desc">Her label icin sol hemisfer centroid koordinatlari (RAS mm, 3D Slicer pin referansi).</p>'
    html += '<div class="tbl-wrap"><table>'
    html += "<tr><th>Label</th><th>cx (mm)</th><th>cy (mm)</th><th>cz (mm)</th>"
    for s in subjects:
        html += f"<th>Vol L — {s.split('-')[0]}</th>"
    html += "</tr>"

    for label in labels:
        # Ortalama centroid (sol, tum subjectler)
        cx = cy = cz = []
        vals = []
        for subj in subjects:
            r = by_key.get((subj, label, "left"))
            if r:
                try:
                    cx.append(float(r["centroid_x_ras_mm"]))
                    cy.append(float(r["centroid_y_ras_mm"]))
                    cz.append(float(r["centroid_z_ras_mm"]))
                    vals.append(float(r["volume_mm3"]))
                except: pass

        cx_avg = f"{sum(cx)/len(cx):.1f}" if cx else "—"
        cy_avg = f"{sum(cy)/len(cy):.1f}" if cy else "—"
        cz_avg = f"{sum(cz)/len(cz):.1f}" if cz else "—"

        html += f"<tr><td><b>{label}</b></td>"
        html += f"<td>{cx_avg}</td><td>{cy_avg}</td><td>{cz_avg}</td>"
        for subj in subjects:
            r = by_key.get((subj, label, "left"))
            v = fv(r["volume_mm3"]) if r else "—"
            html += f"<td>{v}</td>"
        html += "</tr>"

    html += "</table></div>"
    return html


# ─────────────────────────────────────────────────────────────────────────────
# Ana
# ─────────────────────────────────────────────────────────────────────────────

def main():
    mrows = load_csv(MORPH_CSV)
    qrows = load_csv(QUAL_CSV)
    today = date.today().strftime("%d %B %Y")

    body = f"""
    <div class="wrap">
      <h1>Thalamic Nuclei Morphometrics — Rapor</h1>
      <div class="subtitle">Olusturulma: {today} &nbsp;|&nbsp; Veri: Warped_Hakan (IXI dataset, 5 subject)
        &nbsp;|&nbsp; Atlas: MNI152 T1 1mm</div>

      <h2>1. Proje Ozeti</h2>
      <div class="card">{section_ozet(mrows, qrows)}</div>

      <h2>2. Morphometri — Label Bazinda Ortalamalar</h2>
      <div class="card">{section_morphometrics(mrows)}</div>

      <h2>3. Centroid Koordinatlari ve Subject Bazinda Hacimler</h2>
      <div class="card">{section_centroid_table(mrows)}</div>

      <h2>4. Subject Bazinda Detay — Hacimler (mm³)</h2>
      <div class="card">{section_subject_detail(mrows)}</div>

      <h2>5. Warp Kalite ve Dogruluk Analizi</h2>
      <div class="card">{section_kalite(qrows)}</div>

      <h2>6. İstatistiksel Şekil Modeli (SSM) — PCA Sonuçları</h2>
      <div class="card">{section_ssm()}</div>

      <h2>7. Sinirlamalar</h2>
      <div class="card">{section_limitasyonlar()}</div>

      <footer>BrainSeg Morphometrics Pipeline &nbsp;·&nbsp; {today} &nbsp;·&nbsp; Araştırma amaçlıdır.</footer>
    </div>
    """

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Thalamic Nuclei Morphometrics</title>
<style>{CSS}</style>
</head>
<body>{body}</body>
</html>"""

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Rapor: {OUT_HTML}")


if __name__ == "__main__":
    main()
