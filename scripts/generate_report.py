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
.info { background: #e8f4fd; border-left: 4px solid #2196f3;
        padding: 10px 14px; border-radius: 0 6px 6px 0; margin: 10px 0;
        font-size: 12px; color: #1a3a5c; }
.bar-wrap { display: flex; align-items: center; gap: 6px; }
.bar { height: 10px; border-radius: 3px; min-width: 2px; }
.section-desc { color: #555; font-size: 12px; margin-bottom: 12px; line-height: 1.7; }
.intro-box { background: linear-gradient(135deg,#e8eaf6,#f5f7fa); border-radius: 10px;
             border: 1px solid #c5cae9; padding: 20px 24px; margin-bottom: 18px; }
.intro-box h3 { color: #1a237e; margin-bottom: 8px; }
.intro-box p  { font-size: 13px; color: #333; line-height: 1.8; margin-bottom: 10px; }
.intro-box ul { padding-left: 22px; font-size: 13px; color: #333; line-height: 1.9; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; margin: 10px 0 14px; font-size: 12px; }
.legend-item { display: flex; align-items: center; gap: 6px; }
.legend-dot { width: 14px; height: 14px; border-radius: 3px; flex-shrink: 0; }
.sozluk-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px,1fr));
               gap: 12px; }
.sozluk-item { background: #f8f9ff; border-radius: 6px; padding: 12px 16px;
               border-left: 3px solid #1a237e; }
.sozluk-item dt { font-weight: bold; color: #1a237e; margin-bottom: 4px; }
.sozluk-item dd { font-size: 12px; color: #444; line-height: 1.6; }
footer { text-align: center; color: #aaa; font-size: 11px; margin-top: 32px; }
th[title] { cursor: help; border-bottom: 2px dotted #fff; }
"""

def pill(text, kind):
    cls = {"IYI":"good","ORTA":"warn","ZAYIF":"bad",
           "SIMETRIK":"good","HAFIF_ASIM":"warn","ASIMETRIK":"bad"}.get(text, "na")
    return f'<span class="pill pill-{cls}">{text}</span>'


def legend_html():
    return """
    <div class="legend">
      <div class="legend-item"><div class="legend-dot" style="background:#d4edda"></div> İyi / Simetrik</div>
      <div class="legend-item"><div class="legend-dot" style="background:#fff3cd"></div> Orta / Dikkat</div>
      <div class="legend-item"><div class="legend-dot" style="background:#f8d7da"></div> Zayıf / Sorunlu</div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Bolumler
# ─────────────────────────────────────────────────────────────────────────────

def section_giris():
    return """
    <div class="intro-box">
      <h3>Bu Rapor Ne Anlatiyor?</h3>
      <p>
        <b>Talamus</b>, beynin tam ortasinda yer alan ve duyu, hareket, dikkat gibi islevleri
        duzenleyen kritik bir yapıdır. Talamus kendi icinde birbirinden farklı islevlere sahip
        <b>39 alt cekirdekten (nucleus)</b> olusur — her biri beyinde farkli bir bolgeyle baglantili,
        farklı bir gorevi olan kucuk alt bolumler.
      </p>
      <p>
        Bu calismada IXI veri setinden alinan <b>5 saglikli bireyin</b> beyin MRI goruntuleri
        kullanilarak su sorulara yanit aranmistir:
      </p>
      <ul>
        <li>Her thalamic nucleus ne kadar buyuk, hangi sekilde, beynin neresinde?</li>
        <li>Sol ve sag taraflar birbirine ne kadar simetrik?</li>
        <li>Bireyler arasinda sekil/hacim farkliligi ne kadar?</li>
        <li>Bu farkliliklari ozetleyen istatistiksel bir sekil modeli uretebilir miyiz?</li>
      </ul>
      <p style="margin-top:10px; font-size:12px; color:#555;">
        <b>Nasil calisti?</b> &nbsp;
        Hakan atlasindaki hazir etiket maskeleri, her bireyin beyin MRI goruntusuyle
        <i>warp (eslesme/kayit)</i> teknigi kullanilarak hizalanmistir.
        Ardindan her maskenin hacmi, yuzey alani, sekil ozellikleri ve uzaydaki konumu
        otomatik olarak olculmustir. Son adimda tum bireylerin sekil verileri birlestirilip
        istatistiksel bir sekil modeli (SSM) olusturulmustur.
      </p>
    </div>
    <div class="info" style="margin-bottom:10px;">
      <b>Renk kodlari bu raporun tamaminda gecerlidir:</b> &nbsp;
      Yesil = iyi/beklenen aralik &nbsp;|&nbsp; Sari = dikkat/sinir deger &nbsp;|&nbsp; Kirmizi = sorunlu/beklenenden sapma
    </div>
    """


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
        (len(subjects), "Analiz Edilen Birey (Subject)"),
        (len(labels),   "Thalamic Nucleus (Alt Cekirdek)"),
        (total,         "Toplam Olcum Sayisi"),
        (f"{pct_iyi:.0f}%", "Warp Kalitesi Iyi (VPI)"),
        (f"{pct_sim:.0f}%", "Sol/Sag Simetrik"),
        (f"{avg_vol:.0f} mm³", "Ortalama Nucleus Hacmi"),
    ]:
        html += f'<div class="stat-box"><div class="stat-num">{num}</div><div class="stat-lbl">{lbl}</div></div>'
    html += "</div>"

    html += "<h3>Analiz Edilen Bireyler</h3><p class='section-desc'>"
    html += " &nbsp;|&nbsp; ".join(subjects)
    html += "</p>"
    html += """<p class='section-desc'>
      IXI veri seti saglikli yetiskin bireylere ait anonim MRI goruntuleri icermektedir.
      Her birey icin sol ve sag thalamic nucleus ayri ayri olculmustur
      (toplam: 5 birey &times; 39 nucleus &times; 2 taraf = 390 olcum noktasi).
    </p>"""

    return html


def section_morphometrics(mrows):
    subjects = sorted(set(r["subject"] for r in mrows))
    labels   = sorted(set(r["label"]   for r in mrows))

    # Label bazi ortalamalar (sol+sag, tum subjectler)
    by_label = defaultdict(list)
    for r in mrows:
        by_label[r["label"]].append(r)

    html = '<p class="section-desc">'
    html += '<b>Her satirda bir thalamic nucleus (alt cekirdek) var.</b> '
    html += 'Degerler 5 bireyin sol ve sag hemisfer olcumlerinin ortalamasidir. '
    html += 'Sutun aciklamalari icin sutun basliklari uzerine farenizi getirin (tooltip). '
    html += '<i>Elongation ve Flatness</i> degerleri cok kucuk yapilarda (hacmi &lt;20 mm³ olanlar) '
    html += 'hesaplanamaz — bu satirlar sari renkle isaretlenmistir.</p>'
    html += legend_html()
    html += '<div class="tbl-wrap"><table>'
    html += (
        '<tr>'
        '<th title="Thalamic nucleus adi">Nucleus</th>'
        '<th title="Sol hemisfer ortalama hacmi (milimetre kup). Beyin sol yarisindaki yapinin buyuklugu.">Hacim Sol (mm³)</th>'
        '<th title="Sag hemisfer ortalama hacmi (milimetre kup). Beyin sag yarisindaki yapinin buyuklugu.">Hacim Sag (mm³)</th>'
        '<th title="Yapiyi saran dis yuzey alani (milimetre kare). Yuzey ne kadar girintili/cikintiliysa o kadar yuksek.">Yuzey Alani (mm²)</th>'
        '<th title="Sekil ne kadar kureye benziyor? 1.0 = mukemmel kure. Dusuk deger = uzun/yassik sekil.">Kompaktlik (0-1)</th>'
        '<th title="Yapinin en uzun ekseninin en kisa eksenine orani. Yuksek deger = uzun/igne seklinde.">Uzama (Elongation)</th>'
        '<th title="Yapinin ne kadar yassik oldugu. Yuksek deger = disk/pankek seklinde.">Yassiklik (Flatness)</th>'
        '<th title="Yapinin X yonunde (sol-sag) kapligi mesafe (mm).">Genislik X (mm)</th>'
        '<th title="Yapinin Y yonunde (on-arka) kapligi mesafe (mm).">Derinlik Y (mm)</th>'
        '<th title="Yapinin Z yonunde (asagi-yukari) kapligi mesafe (mm).">Yukseklik Z (mm)</th>'
        '<th title="Yapinin kapligi kutunun yuzde kaci dolu? 1.0 = tam dolu kutu, dusuk = seyrek/dagitik sekil.">Kutu Doluluk</th>'
        '<th title="Kac ayri parcadan olusuyor? 1 = tek parcali (ideal). Yuksek = parcali/kopuk yapi.">Parca Sayisi</th>'
        '<th title="Yapinin iskelet uzunlugu (mm). Cok kucuk yapilar icin 0 donebilir.">Iskelet (mm)</th>'
        '</tr>'
    )

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
    html += '<p class="note"><b>Sari satirlar</b> hacmi 20 mm³\'ten kucuk nucleusları gosteriyor (AD, MV, Pv, sPf). '
    html += 'Bu yapilar 1mm cozunurluklu goruntude cok az voxel icerdiginden '
    html += 'Uzama ve Yassiklik metrikleri guvenilir sekilde hesaplanamaz.</p>'
    return html


def section_subject_detail(mrows):
    subjects = sorted(set(r["subject"] for r in mrows))
    labels   = sorted(set(r["label"]   for r in mrows))
    by_subj_label = defaultdict(dict)
    for r in mrows:
        by_subj_label[(r["subject"], r["label"], r["side"])] = r

    html = '<p class="section-desc">'
    html += 'Bu tablo her nucleus icin bireylerin (subject) hacimlerini yan yana gostermektedir. '
    html += 'Boylece kimlerin buyuk kimlerin kucuk yapilara sahip oldugu karsilastirilabildigi gibi '
    html += 'bireylerarasi tutarlilik da gorulur. '
    html += '<b>CV (Varyasyon Katsayisi)</b> sutunu tutarliligi ozetler: '
    html += 'CV &lt; 0.15 = bireyler birbirine cok benziyor (yesil), '
    html += '0.15&ndash;0.25 = orta farklilik (sari), '
    html += '&gt; 0.25 = buyuk bireysel farklilik (kirmizi).</p>'
    html += legend_html()
    html += '<div class="tbl-wrap"><table>'

    # Header
    html += '<tr><th title="Nucleus adi">Nucleus</th><th title="Sol veya sag hemisfer">Taraf</th>'
    for s in subjects:
        html += f'<th title="{s}">{s.split("-")[0]}<br>{s.split("-")[1]}</th>'
    html += '<th title="Tum bireylerin ortalamasi">Ortalama</th>'
    html += '<th title="Varyasyon Katsayisi: standart sapma / ortalama. Dusuk = bireyler arasi tutarlilik iyi.">CV</th></tr>'

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

    html = '<p class="section-desc">'
    html += '<b>Warp nedir?</b> Her bireyin beyni atlasla eslestirilirken maskelerin boyutu '
    html += 'biraz degismis olabilir. Bu bolum o degisimin ne kadar buyuk oldugunu olcer.<br><br>'
    html += '<b>VPI (Hacim Koruma Indeksi)</b> = warp sonrasi olculen hacim / atlastaki beklenen hacim. '
    html += '1.0 = atlas ile birebir. '
    html += '<span style="background:#d4edda;padding:1px 6px;border-radius:3px;">IYI: 0.70&ndash;1.30</span> &nbsp; '
    html += '<span style="background:#fff3cd;padding:1px 6px;border-radius:3px;">ORTA: 0.50&ndash;1.50</span> &nbsp; '
    html += '<span style="background:#f8d7da;padding:1px 6px;border-radius:3px;">ZAYIF: &lt;0.50 veya &gt;1.50</span><br><br>'
    html += '<b>Sol/Sag Simetri</b>: Thalamus simetrik bir yapi oldugu icin sol ve sag taraflar benzer hacimde '
    html += 'olmalidir. L/R orani 1.0\'a ne kadar yakinsa o kadar simetrik.'
    html += '</p>'
    html += legend_html()

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
    html += "<h3>Her Nucleus Icin Warp Kalite Tablosu</h3>"
    html += '<div class="tbl-wrap"><table>'
    html += (
        '<tr>'
        '<th title="Thalamic nucleus adi">Nucleus</th>'
        '<th title="Atlastaki beklenen sol hacim (referans deger)">Atlas Sol (mm³)</th>'
        '<th title="Atlastaki beklenen sag hacim (referans deger)">Atlas Sag (mm³)</th>'
        '<th title="Olculen sol hacim / atlas sol hacmi. 1.0 = ideal. 5 bireyin ortalamasi.">VPI Sol (ort)</th>'
        '<th title="Olculen sag hacim / atlas sag hacmi. 1.0 = ideal. 5 bireyin ortalamasi.">VPI Sag (ort)</th>'
        '<th title="Genel warp kalite sinifi">Kalite</th>'
        '<th title="Sol hacim / sag hacim orani. 1.0 = tam simetrik. 5 bireyin ortalamasi.">Sol/Sag Oran</th>'
        '<th title="Sol ve sag hacimler birbirine ne kadar yakin?">Simetri</th>'
        '<th title="Varyasyon Katsayisi: 5 birey arasinda bu nucleusun ne kadar farkli olculdugunu gosterir. Dusuk = tutarli.">Bireyler Arasi CV</th>'
        '<th title="CV degeri yorumu: Tutarli (&lt;0.15) / Orta (0.15-0.25) / Tutarsiz (&gt;0.25)">Tutarlilik</th>'
        '</tr>'
    )

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
    html += '<p class="note"><b>Kucuk yapilarda VPI neden sapabilir?</b> &nbsp; '
    html += 'AD, Pv, MV, sPf gibi cok kucuk nucleuslar (hacim &lt;20 mm³) 1mm cozunurluklu '
    html += 'goruntude sadece birkaç voxelden olusur. Bu durumlarda 1-2 voxellik warp hatasi '
    html += 'bile VPI\'yi buyuk olcude etkiler. Bu bir olcum sinirlamasi olup warp algoritmasi '
    html += 'hatasini degil, gorunturun cozunurlugunun yetersizligini yansitir.</p>'
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
    html += '<b>Istatistiksel Sekil Modeli (SSM) nedir?</b> &nbsp; '
    html += 'Bes bireyin ayni nucleusuna bakinca sekillerin birbiriyle benzer ama farkli oldugu gorulur. '
    html += 'SSM bu farkliliklari matematiksel olarak ozetler: "Bireyler arasinda sekil en cok hangi yonde degisiyor?" '
    html += 'sorusuna yanit verir. Bunu yapmak icin her nucleus icin 12 yuzey noktasi (landmark) belirlenir '
    html += 've bu 12 noktanin 3D koordinatlari (12 &times; 3 = 36 sayi) bir araya getirilerek '
    html += '<b>PCA (Temel Bilesenler Analizi)</b> uygulanir.<br><br>'
    html += '<b>P1, P2, P3, P4 nedir?</b> &nbsp; '
    html += 'PCA\'nin buldugu ana varyasyon eksenleridir. '
    html += 'P1 bireyler arasindaki sekil farkinin en buyuk kaynagini temsil eder (en fazla varyans), '
    html += 'P2 ikinci buyuk kaynagi, vs. '
    html += '3D Slicer\'da bu modlarin slider\'lari ile ortalama sekilten ne kadar sapilabilecegi '
    html += 'interaktif olarak gorsellestirilir (<code>slicer_ssm_viewer.py</code>).<br><br>'
    html += '<b>sigma (σ)</b>: Her modun "dogal" degisim buyuklugu. '
    html += 'Slider +1σ konumundaysa o moda gore ortalamadan 1 standart sapma uzakta olan sekli gorursunuz.'
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
    html += (
        '<tr>'
        '<th title="Thalamic nucleus adi">Nucleus</th>'
        '<th title="SSM hesaplamasina katilan birey sayisi">N Birey</th>'
        '<th title="Kac PCA modu hesaplandi (max 4)">N Mod</th>'
        '<th title="P1 modunun acikladi sekil varyansinin yuzdesi. Yuksek = sekil degisimi tek bir eksene yuklu.">P1 Varyansi</th>'
        '<th title="P2 modunun acikladi sekil varyansinin yuzdesi.">P2 Varyansi</th>'
        '<th title="P3 modunun acikladi sekil varyansinin yuzdesi.">P3 Varyansi</th>'
        '<th title="P4 modunun acikladi sekil varyansinin yuzdesi.">P4 Varyansi</th>'
        '<th title="P1 modunun standart sapmasi (mm). Bireyler arasindaki tipik sekil farkliligi buyuklugu.">σ₁ (mm)</th>'
        '<th title="P1 varyansi yorumu">Yorum</th>'
        '</tr>'
    )

    for r in rows:
        pct  = r["pct"]
        pc1  = pct[0] if len(pct) > 0 else 0
        pc2  = pct[1] if len(pct) > 1 else 0
        pc3  = pct[2] if len(pct) > 2 else 0
        pc4  = pct[3] if len(pct) > 3 else 0

        if pc1 >= 85:
            pc1_cls = "good"
            yorum   = "Tek eksende degisim"
        elif pc1 >= 70:
            pc1_cls = ""
            yorum   = "Normal dagilim"
        else:
            pc1_cls = "warn"
            yorum   = "Cok boyutlu varyasyon"

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
    html += '<p class="note">'
    html += '<b>σ₁ yuksekse ne anlama gelir?</b> &nbsp; '
    html += 'O nucleusun sekli bireyden bireye buyuk farklilik gosteriyor demektir. '
    html += 'Dusuk σ₁ = bireyler o nucleusu cok benzer sekillerle tasir.<br>'
    html += '<b>P1 varyansi &gt;%85 ise:</b> Tum sekil farkliligi neredeyse tek bir yonde — '
    html += 'buyume/kucusme veya tek yonde uzama gibi basit bir degisim hakimdir. '
    html += '<b>Dusuk P1 varyansi:</b> Sekil farkliligi karmasik, birden fazla boyutta dagitik.</p>'
    return html


def section_limitasyonlar():
    items = [
        ("Kucuk Nucleus Sorunu",
         "AD, Pv, MV, sPf gibi nucleuslar 20 mm³\'ten kucuktur — 1mm cozunurluklu goruntude "
         "sadece birkaç voxele karsilik gelir. Bu yapilarda Uzama/Yassiklik hesaplanamaz, "
         "VPI degerleri guvenilmez. Daha yuksek cozunurluklu MRI ile bu sinir ortadan kalkar."),
        ("Warp Hatasi Yayilimi",
         "Tum metrikler warp sonrasi maskelere dayaniyor. Eger atlas-birey eslemesi "
         "hataliysa tum metrikler de etkilenir. 'Warp Kalite' bolumundeki VPI bu riski olcer "
         "ve zayif VPI goruldugunde o nucleusun olcumlerine dikkatli yaklasılmalidir."),
        ("Az Birey Sayisi",
         "Sadece 5 birey analiz edildi. Bu sayida istatistiksel guc sinirlidir; "
         "ortalama ve CV degerleri kesin istatistik degil referans niceligi tasir. "
         "Daha guclu sonuclar icin en az 20-30 birey onerilir."),
        ("Yuzey Alani Hassasiyeti",
         "Yuzey olusturmak icin kullanilan 'marching cubes' yontemi voxel boyutuna baglidir. "
         "Kucuk yapilarda gercek yuzeyden biraz daha buyuk tahmin uretir."),
        ("Iskelet Metrikleri",
         "Cok kucuk veya kopuk yapilar icin iskelet uzunlugu 0 olarak donmektedir. "
         "Bu deger ancak CL, MDpc, PuM gibi buyuk ve tek parcali yapilarda guvenilirdir."),
        ("Tek Atlas Kullanimi",
         "Analiz tek bir MNI152 T1 atlasi kullanilarak yapildi. "
         "Birden fazla atlas veya label fusion yontemlerinin kullanimi daha saglam sonuclar verir."),
    ]
    html = '<p class="section-desc">Her calismanin teknik sinirlari vardir. '
    html += 'Asagidaki maddeler bu analizin sonuclarini yorumlarken dikkate alinmasi gereken '
    html += 'kisitlamalari aciklamaktadir.</p>'
    html += '<ul style="padding-left:20px; line-height:2.2;">'
    for title, desc in items:
        html += f"<li><b>{title}:</b> {desc}</li>"
    html += "</ul>"
    return html


def section_sozluk():
    kavramlar = [
        ("Talamus (Thalamus)",
         "Beynin merkezinde yer alan ve duyu, hareket, dikkat sinyallerini ilgili korteks bolgelerine "
         "ileten role yapisi. Kendi icinde 39+ alt cekirdege (nucleus) bolunur."),
        ("Thalamic Nucleus (Alt Cekirdek)",
         "Talamusun islevsel ve anatomik alt birimleri. Her birinin baglantili oldugu korteks bolgesi "
         "ve islevi farklidir. Bu analizde Hakan atlasindaki 39 nucleus kullanilmistir."),
        ("Atlas",
         "Standart bir beyin sablonu uzerine isaretlenmis anatomik haritalar. Bu calismada "
         "MNI152 uzayindaki Hakan talamus atlasi kullanilmistir."),
        ("Warp / Kayit (Registration)",
         "Bir bireyin MRI goruntusunu standart atlasla eslestirecek bicimde matematiksel olarak "
         "deforme etme islemi. Boylece atlas maskeleri her bireye uyarlanir."),
        ("Maske",
         "Bir nucleus bolgesini belirten ikili (0/1) 3D goruntu. 1 olan voxellar o nucleusa aittir."),
        ("VPI — Hacim Koruma Indeksi",
         "Warp sonrasi olculen hacmin atlastaki beklenen hacme orani. 1.0 = mukemmel eslesme. "
         "0.70-1.30 = kabul edilebilir. Bu sinirin disinda kalan olcumler warp hatasindan etkilenebilir."),
        ("Kompaktlik",
         "Seklin kureye ne kadar benzedigi. 1.0 = mukemmel kure. Dusuk deger = uzun veya gircik sekil."),
        ("Elongation (Uzama)",
         "En uzun eksenin en kisa eksenine orani. Yuksek = igne/sigara seklinde uzun yapi."),
        ("Flatness (Yassiklik)",
         "Yapinin ne kadar pankek/disk seklinde oldugu. Yuksek = yassik, ince yapi."),
        ("Centroid",
         "Yapinin agirlik merkezi koordinati. 3D Slicer\'da 'pin' olarak gosterilebilir."),
        ("CV — Varyasyon Katsayisi",
         "Standart sapma / ortalama. Bireyler arasi farklilik olcusu. "
         "< 0.15 = tutarli, 0.15-0.25 = orta, > 0.25 = yuksek bireysel farklilik."),
        ("SSM — Istatistiksel Sekil Modeli",
         "Birden fazla bireyin sekil verilerinden PCA ile elde edilen model. "
         "'Bu yapilar tipik olarak nasil gozukur ve bireyler arasinda nasil degisir?' sorusunu yanıtlar."),
        ("PCA — Temel Bilesenler Analizi",
         "Cok boyutlu verideki en buyuk varyasyon yonlerini bulan istatistiksel yontem. "
         "P1 en fazla varyasi, P2 ikinci en fazla varyasi acıklar."),
        ("P1, P2, P3, P4 (SSM Modlari)",
         "PCA\'nin buldugu varyasyon eksenleri. 3D Slicer\'da slider ile her modu ±3σ araliginda "
         "gezdirerek ortalama sekilten sapan bireysel varyasyonlar interaktif goruntulenebilir."),
        ("TPS — Thin Plate Spline",
         "Landmark noktalarini kullanarak 3D meshin elastik sekilde deforme edilmesi. "
         "Slicer viewer\'da mavi modeli gunceller."),
        ("sigma σ",
         "Standart sapma. SSM\'de bir modun sigma degeri, bireyler arasindaki tipik sekil farkinin "
         "o eksende ne buyuk oldugunu mm cinsinden gosterir."),
    ]
    html = '<p class="section-desc">Bu raporda gecen teknik terimlerin sade aciklamalari.</p>'
    html += '<div class="sozluk-grid"><dl>'
    for term, desc in kavramlar:
        html += f'<div class="sozluk-item"><dt>{term}</dt><dd>{desc}</dd></div>'
    html += '</dl></div>'
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
      <h1>Thalamic Nuclei Morphometrics — Analiz Raporu</h1>
      <div class="subtitle">
        Olusturulma: {today} &nbsp;|&nbsp; Veri: IXI Dataset — 5 Saglikli Birey
        &nbsp;|&nbsp; Atlas: Hakan Talamus Atlasi (MNI152 T1 1mm)
        &nbsp;|&nbsp; 39 Nucleus &nbsp;|&nbsp; Arastirma amaclidir
      </div>

      <h2>Bu Calisma Hakkinda</h2>
      <div class="card">{section_giris()}</div>

      <h2>1. Ozet Bulgular</h2>
      <div class="card">{section_ozet(mrows, qrows)}</div>

      <h2>2. Her Nucleus Icin Sekil Olcumleri (Morfometri)</h2>
      <div class="card">{section_morphometrics(mrows)}</div>

      <h2>3. Nucleusların Beyndeki Konumu ve Birey Kiyaslamasi</h2>
      <div class="card">{section_centroid_table(mrows)}</div>

      <h2>4. Bireyler Arasi Hacim Karsilastirmasi</h2>
      <div class="card">{section_subject_detail(mrows)}</div>

      <h2>5. Warp (Atlas Eslesme) Kalitesi</h2>
      <div class="card">{section_kalite(qrows)}</div>

      <h2>6. Istatistiksel Sekil Modeli (SSM) — Sekil Varyasyonu Analizi</h2>
      <div class="card">{section_ssm()}</div>

      <h2>7. Sinirlamalar ve Dikkat Edilmesi Gerekenler</h2>
      <div class="card">{section_limitasyonlar()}</div>

      <h2>8. Terimler Sozlugu</h2>
      <div class="card">{section_sozluk()}</div>

      <footer>BrainSeg Morphometrics Pipeline &nbsp;·&nbsp; {today} &nbsp;·&nbsp;
      IXI Dataset / Hakan Atlas &nbsp;·&nbsp; Arastirma amaclidir.</footer>
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
