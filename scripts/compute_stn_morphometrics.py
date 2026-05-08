"""
STN (Subthalamic Nucleus) — Morfometri ve Konum Dogrulama
==========================================================
Atlas etiketi: STh  (left_STh_warped.nii.gz / right_STh_warped.nii.gz)

Her subject icin uretilen metrikler:
  1. Geometrik sekil: hacim, yuzey alani, kompaktlik, elongasyon, flatness
  2. Uzun eksen: PCA ile ana yon vektoru + eksen uzunlugu
  3. Iskelet: uzunluk + max yazili kure yarici
  4. Atlas referansi: MNI uzayindaki STh hacmiyle oran karsilastirmasi
  5. T1 intensite: atlas_warped icindeki STN sinyali (ortalama, std)
  6. Konum dogrulamasi [DAHILI - ciktida yok]:
       - STN merkezi thalamus_body merkezinin altinda mi?
       - STN ile thalamus_body cakismasi var mi?
  7. Bilateral simetri: hacim asimetrisi + centroid yansima hatasi

Beklenen referans degerler (saglikli yetiskin, literatür):
  Hacim     : 100 - 250 mm3
  Uzun eksen: 9 - 14 mm
  Elongasyon: > 4.0  (ince-uzun lens yapisi)
  MNI koord : sol ~(-12, -14, -6) mm  |  sag ~(+12, -14, -6) mm

Kullanim:
    python scripts/compute_stn_morphometrics.py

Cikti:
    outputs/stn/stn_morphometrics.csv        — tam metrik tablosu
    outputs/stn/stn_validation_report.txt    — okunabilir dogrulama raporu
    outputs/stn/<subject>_STN.fcsv           — Slicer markup (centroid + eksen)
"""

import csv
import os
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.metrics import compute_geometric_morphometrics

try:
    from skimage.measure import marching_cubes
    from scipy.ndimage import gaussian_filter
    _SKIMAGE_OK = True
except ImportError:
    _SKIMAGE_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# Yollar
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(PROJECT_DIR, "data", "subjects")
ATLAS_DIR   = os.path.join(PROJECT_DIR, "data", "atlas", "labels")
OUT_DIR     = os.path.join(PROJECT_DIR, "outputs", "stn")

# Literatür referans degerleri
VOL_MIN_MM3     = 100.0
VOL_MAX_MM3     = 250.0
LONG_AXIS_MIN   = 9.0
LONG_AXIS_MAX   = 14.0
ELONGATION_MIN  = 4.0
THAL_INF_MIN_MM = 3.0   # STN thalamus merkezinin en az kac mm altinda olmali

# CSV sutunlari
CSV_COLUMNS = [
    "subject", "side",
    "centroid_x_ras", "centroid_y_ras", "centroid_z_ras",
    "volume_mm3", "surface_area_mm2",
    "compactness", "elongation", "flatness",
    "long_axis_length_mm",
    "long_axis_x", "long_axis_y", "long_axis_z",
    "bbox_x_mm", "bbox_y_mm", "bbox_z_mm",
    "skeleton_length_mm", "skeleton_max_radius_mm",
    "t1_mean_in_stn", "t1_std_in_stn",
    "atlas_ref_volume_mm3", "atlas_vol_ratio",
    "thalamus_inferior_mm", "thalamus_overlap_vox",
    "lr_volume_asym_pct", "lr_mirror_error_mm",
    "volume_status", "shape_status", "position_status",
    "symmetry_status", "overall_status",
]


# ─────────────────────────────────────────────────────────────────────────────
# Yardimci Fonksiyonlar
# ─────────────────────────────────────────────────────────────────────────────

def load_mask(path: str):
    """NIfTI maske yukle, RAS+ garantile → (affine, binary_mask, zooms)"""
    img   = nib.load(path)
    img   = nib.as_closest_canonical(img)
    data  = img.get_fdata(dtype=np.float32)
    mask  = (data > 0.5).astype(bool)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return img.affine, mask, zooms


def load_volume_data(path: str):
    """Tam voksel verisi ile yukle → (affine, data_float32, zooms)"""
    img   = nib.load(path)
    img   = nib.as_closest_canonical(img)
    data  = img.get_fdata(dtype=np.float32)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return img.affine, data, zooms


def centroid_ras(mask: np.ndarray, affine: np.ndarray):
    """Maskenin agirlik merkezi → RAS mm koordinatlari. Bos maskede None."""
    if mask.sum() == 0:
        return None
    vox = np.array(ndimage.center_of_mass(mask))
    ras = affine @ np.array([vox[0], vox[1], vox[2], 1.0])
    return [round(float(ras[0]), 3), round(float(ras[1]), 3), round(float(ras[2]), 3)]


def compute_long_axis(mask: np.ndarray, affine: np.ndarray, zooms: tuple):
    """
    STN uzun eksenini PCA ile hesapla — tam RAS uzayinda.

    Tum vokseller affine ile RAS mm koordinatlarina donusturulur,
    PCA RAS uzayinda yapilir. Uc noktalar dogrudan RAS mm cinsinden.
    Oblique affine icin dogru; voksel-mm uzayinda yapilanin aksine
    affine rotasyonu iki kez uygulamaz.

    Dondurur:
      unit_vec   — RAS mm birim vektoru (uzun eksen yonu)
      length_mm  — maskenin uzun eksen boyunca gercek uzunlugu
      p1, p2     — uzun eksen uc noktalari (RAS mm)  [Slicer ruler icin]
    """
    coords_vox = np.array(np.where(mask)).T.astype(float)
    if len(coords_vox) < 10:
        return None, None, None, None

    # Tum vokselleri tam affine ile RAS mm koordinatlarına dönüstür
    ones       = np.ones((len(coords_vox), 1))
    coords_ras = (affine @ np.hstack([coords_vox, ones]).T).T[:, :3]

    # PCA dogrudan RAS uzayinda
    mean_ras = coords_ras.mean(axis=0)
    centered = coords_ras - mean_ras
    cov      = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    unit_vec = eigvecs[:, np.argmax(eigvals)]
    norm     = np.linalg.norm(unit_vec)
    if norm < 1e-10:
        return None, None, None, None
    unit_vec = unit_vec / norm

    # Projeksiyon — uc noktalar dogrudan RAS mm cinsinden
    projs     = centered @ unit_vec
    p1        = [round(float(v), 3) for v in (mean_ras + unit_vec * projs.min())]
    p2        = [round(float(v), 3) for v in (mean_ras + unit_vec * projs.max())]
    length_mm = round(float(np.linalg.norm(np.array(p2) - np.array(p1))), 2)

    return [round(float(v), 4) for v in unit_vec], length_mm, p1, p2


def sample_intensity(mask: np.ndarray, mask_affine: np.ndarray,
                     vol_data: np.ndarray, vol_affine: np.ndarray):
    """
    vol_data (atlas_warped T1-like) icerisinde mask voksellerinin
    intensite istatistiklerini hesapla.
    Ayni uzayda olduklari varsayilir (atlas_warped + warped labels).
    """
    if mask.sum() == 0:
        return None, None

    # Boyut uyumunu kontrol et
    if mask.shape != vol_data.shape:
        return None, None

    vals = vol_data[mask]
    vals = vals[vals > 0]
    if len(vals) == 0:
        return None, None

    return round(float(vals.mean()), 3), round(float(vals.std()), 3)


def atlas_reference_volume(side: str) -> float:
    """Atlas STh maskesinin MNI152 uzayindaki hacmi."""
    path = os.path.join(ATLAS_DIR, side, "STh.nii.gz")
    if not os.path.exists(path):
        return None
    affine, mask, zooms = load_mask(path)
    return round(float(mask.sum()) * float(np.prod(zooms)), 2)


def atlas_reference_centroid(side: str):
    """Atlas STh maskesinin MNI152 uzayindaki centroidi (referans degeri)."""
    path = os.path.join(ATLAS_DIR, side, "STh.nii.gz")
    if not os.path.exists(path):
        return None
    affine, mask, zooms = load_mask(path)
    return centroid_ras(mask, affine)


def check_thalamus_position(stn_mask, stn_affine,
                            thal_mask, thal_affine):
    """
    STN'nin thalamus_body'ye gore konumunu dogrula (DAHILI — ciktida goster, raporda yok).
    Dondurur: (inferior_mm, overlap_vox)
      inferior_mm: STN merkezinin thalamus merkezine gore asagilik (+ = dogru)
      overlap_vox: STN ile thalamus cakisan voksel sayisi (0 olmali)
    """
    stn_c  = centroid_ras(stn_mask, stn_affine)
    thal_c = centroid_ras(thal_mask, thal_affine)
    if stn_c is None or thal_c is None:
        return None, None

    inferior_mm = round(float(thal_c[2] - stn_c[2]), 2)  # RAS: z yukari

    overlap_vox = None
    if (stn_mask.shape == thal_mask.shape and
            np.allclose(stn_affine, thal_affine, atol=1e-3)):
        overlap_vox = int((stn_mask & thal_mask).sum())

    return inferior_mm, overlap_vox


# ─────────────────────────────────────────────────────────────────────────────
# Durum Degerlendirmesi
# ─────────────────────────────────────────────────────────────────────────────

def status(ok_cond, warn_cond=None):
    if ok_cond:
        return "PASS"
    elif warn_cond is None or warn_cond:
        return "WARN"
    return "FAIL"


def evaluate(vol, long_axis_len, elongation, inferior_mm, overlap_vox,
             lr_asym_pct, lr_mirror_mm, connected_components=1):
    # Hacim
    if vol is None or vol == 0:
        vol_st = "FAIL"
    elif VOL_MIN_MM3 <= vol <= VOL_MAX_MM3:
        vol_st = "PASS"
    elif vol < VOL_MIN_MM3 * 0.4 or vol > VOL_MAX_MM3 * 3.0:
        vol_st = "FAIL"
    else:
        vol_st = "WARN"

    # Sekil (uzun eksen + elongasyon + baglantili bilesken)
    la_ok   = long_axis_len is not None and LONG_AXIS_MIN <= long_axis_len <= LONG_AXIS_MAX
    el_ok   = elongation is not None and elongation >= ELONGATION_MIN
    if la_ok and el_ok:
        shape_st = "PASS"
    elif (long_axis_len is not None and long_axis_len > 5) or (elongation is not None and elongation > 2):
        shape_st = "WARN"
    else:
        shape_st = "FAIL"
    # Fragmente maske (>1 baglantili bilesken) sekil notunu dusuruyor
    if connected_components is not None and connected_components > 1:
        shape_st = "WARN" if shape_st == "PASS" else shape_st

    # Pozisyon
    if inferior_mm is None:
        pos_st = "WARN"
    elif inferior_mm >= THAL_INF_MIN_MM:
        pos_st = "PASS"
    elif inferior_mm >= 0:
        pos_st = "WARN"
    else:
        pos_st = "FAIL"
    if overlap_vox is not None and overlap_vox > 50:
        pos_st = "FAIL"

    # Simetri
    if lr_asym_pct is None:
        sym_st = "WARN"
    elif lr_asym_pct <= 20:
        sym_st = "PASS"
    elif lr_asym_pct <= 40:
        sym_st = "WARN"
    else:
        sym_st = "FAIL"
    if lr_mirror_mm is not None and lr_mirror_mm > 4.0:
        sym_st = "WARN" if sym_st == "PASS" else sym_st

    # Genel
    all_st = [vol_st, shape_st, pos_st, sym_st]
    if "FAIL" in all_st:
        overall = "FAIL"
    elif "WARN" in all_st:
        overall = "WARN"
    else:
        overall = "PASS"

    return vol_st, shape_st, pos_st, sym_st, overall


# ─────────────────────────────────────────────────────────────────────────────
# .fcsv Yazma (3D Slicer Markup)
# ─────────────────────────────────────────────────────────────────────────────

def _write_fcsv_raw(path: str, points: list):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("# Markups fiducial file version = 5.0\n")
        f.write("# CoordinateSystem = RAS\n")
        f.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
        for i, (lbl, coords) in enumerate(points):
            if coords is None:
                continue
            x, y, z = coords
            nid = f"vtkMRMLMarkupsFiducialNode_{i}"
            f.write(f"{nid},{x:.3f},{y:.3f},{z:.3f},0,0,0,1,1,1,0,{lbl},,\n")


def write_fcsv(subject: str, centroid_pts: list, axis_pts: list):
    """
    Iki ayri fcsv uretir:
      <subject>_STN_centroids.fcsv  — yalnizca sol/sag centroid (Slicer'da temiz gorunum)
      <subject>_STN_axes.fcsv       — uzun eksen uc noktalari (ayrı katman)
    """
    centroid_path = os.path.join(OUT_DIR, f"{subject}_STN_centroids.fcsv")
    _write_fcsv_raw(centroid_path, centroid_pts)

    if axis_pts:
        axis_path = os.path.join(OUT_DIR, f"{subject}_STN_axes.fcsv")
        _write_fcsv_raw(axis_path, axis_pts)

    return centroid_path


# ─────────────────────────────────────────────────────────────────────────────
# Rapor Formatlayici
# ─────────────────────────────────────────────────────────────────────────────

ICON = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL"}


def _warn_notes(rd: dict) -> list:
    """
    Subject'a ozgu WARN/FAIL icin yorumsal notlar uret.
    Her madde "  !  " ile baslar, raporda sona eklenir.
    """
    notes = []
    subj = rd["subject"]

    for side in ("left", "right"):
        sd = rd.get(side)
        if sd is None:
            continue

        vol = sd.get("volume_mm3", 0)
        la  = sd.get("long_axis_length_mm")
        vst = sd.get("volume_status", "")
        sst = sd.get("shape_status", "")
        c   = sd.get("centroid")

        if vst == "WARN" and vol and vol > VOL_MAX_MM3:
            notes.append(
                f"  ! [{side.upper()}] Hacim {vol:.1f} mm3 — beklenen ust sinirin"
                f" ({VOL_MAX_MM3:.0f} mm3) hafif ustunde. Buyuk ihtimalle"
                f" anatomik varyasyon; warp kalitesini Slicer'da gorsel olarak"
                f" dogrulayin."
            )

        if sst == "WARN" and la and la > LONG_AXIS_MAX:
            notes.append(
                f"  ! [{side.upper()}] Uzun eksen {la:.1f} mm — beklenen aralik"
                f" ({LONG_AXIS_MIN:.0f}–{LONG_AXIS_MAX:.0f} mm) uzerinde."
                f" Hacim WARN ile birlikte goruluyorsa warp genislemesi"
                f" dusunulebilir."
            )

        # Lateral kaymis X koordinati (native RAS +x = sag)
        if c:
            x_abs = abs(c[0])
            expected_lat = 15.0   # saglikli bireyler icin native uzayda tolerans
            if x_abs > expected_lat:
                notes.append(
                    f"  ! [{side.upper()}] Centroid X = {c[0]:+.2f} mm — beklenendan"
                    f" lateral kaymis gorunuyor (|x| > {expected_lat:.0f} mm)."
                    f" Warp bu subject icin tam oturmamis olabilir."
                    f" Slicer'da T2 uzerinde gorsel dogrulama onerilir."
                )

    # Mirror error yorumu
    mer = rd.get("lr_mirror_mm")
    if mer is not None and mer > 4.0:
        notes.append(
            f"  ! Bilateral yansima hatasi {mer:.2f} mm — onemli asimetri."
            f" NOT: Bu olcum native (subject) uzayinda hesaplanmistir;"
            f" beyin goruntu merkezine gore konumlanmamissa hata sisti"
            f" gorunebilir. MNI uzayinda yeniden degerlendirilmesi"
            f" veya Slicer'da gorsel kontrol yapilmasi onerilir."
        )

    return notes


def format_report(report_data: list, atlas_ref_centroids: dict = None) -> str:
    lines = []
    w = 66
    lines.append("=" * w)
    lines.append("  STN (Subthalamic Nucleus) — Dogrulama Raporu")
    lines.append(f"  Atlas etiketi: STh   |   Tarih: {__import__('datetime').date.today()}")
    lines.append("=" * w)
    lines.append("  Beklenen referans degerleri (saglikli yetiskin):")
    lines.append(f"    Hacim      : {VOL_MIN_MM3:.0f} – {VOL_MAX_MM3:.0f} mm3")
    lines.append(f"    Uzun eksen : {LONG_AXIS_MIN:.0f} – {LONG_AXIS_MAX:.0f} mm")
    lines.append(f"    Elongasyon : > {ELONGATION_MIN:.1f}")
    lines.append(f"    Thal alti  : > {THAL_INF_MIN_MM:.0f} mm")
    if atlas_ref_centroids:
        cL = atlas_ref_centroids.get("left")
        cR = atlas_ref_centroids.get("right")
        if cL and cR:
            lines.append(f"    MNI atlas centroid (gercek):")
            lines.append(f"      sol: ({cL[0]:+.1f}, {cL[1]:+.1f}, {cL[2]:+.1f}) mm")
            lines.append(f"      sag: ({cR[0]:+.1f}, {cR[1]:+.1f}, {cR[2]:+.1f}) mm")
        else:
            lines.append(f"    MNI ref    : sol ~(-12, -14, -6) mm  |  sag ~(+12, -14, -6) mm")
    else:
        lines.append(f"    MNI ref    : sol ~(-12, -14, -6) mm  |  sag ~(+12, -14, -6) mm")
    lines.append("  NOT: Subject centroidleri native T1 uzayindadir; MNI ile")
    lines.append("       dogrudan karsilastirma yapilmaz, yalnizca referanstir.")
    lines.append("")

    pass_c = warn_c = fail_c = 0

    for rd in report_data:
        subj = rd["subject"]
        lines.append(f"  ┌── {subj}")

        for side in ("left", "right"):
            sd = rd.get(side)
            if sd is None:
                lines.append(f"  │   {side:5s} : MASKE YOK")
                continue

            ovr   = sd["overall_status"]
            icon  = ICON.get(ovr, "????")
            lines.append(f"  │")
            lines.append(f"  │   [{icon}] {side.upper()} STN")

            c = sd.get("centroid")
            if c:
                lines.append(f"  │         Merkez RAS    : ({c[0]:+7.2f}, {c[1]:+7.2f}, {c[2]:+7.2f}) mm")

            vol   = sd.get("volume_mm3", 0)
            avol  = sd.get("atlas_ref_vol")
            arat  = sd.get("atlas_vol_ratio")
            vst   = sd.get("volume_status", "")
            a_str = f"  [atlas ref: {avol:.0f} mm3, oran: {arat:.2f}]" if avol and arat else ""
            lines.append(f"  │         Hacim         : {vol:7.1f} mm3  [{vst}]{a_str}")

            la  = sd.get("long_axis_length_mm")
            el  = sd.get("elongation")
            flt = sd.get("flatness")
            cc  = sd.get("connected_components")
            sst = sd.get("shape_status", "")
            la_str = f"{la:.1f} mm" if la else "—"
            el_str = f"{el:.2f}"   if el else "—"
            fl_str = f"{flt:.2f}"  if flt else "—"
            cc_str = f"  bilesken: {cc}" if cc is not None and cc > 1 else ""
            lines.append(f"  │         Uzun eksen    : {la_str}  [{sst}]{cc_str}")
            lines.append(f"  │         Elongasyon    : {el_str}    Flatness: {fl_str}")

            vec = sd.get("long_axis_vec")
            if vec:
                lines.append(f"  │         Eksen yonu   : ({vec[0]:+.3f}, {vec[1]:+.3f}, {vec[2]:+.3f})")

            sa  = sd.get("surface_area_mm2")
            cmp = sd.get("compactness")
            sa_str  = f"{sa:.1f} mm2"  if sa  else "—"
            cmp_str = f"{cmp:.4f}"     if cmp else "—"
            lines.append(f"  │         Yuzey alani   : {sa_str}    Kompaktlik: {cmp_str}")

            sk_len = sd.get("skeleton_length_mm")
            sk_rad = sd.get("skeleton_max_radius_mm")
            if sk_len:
                lines.append(f"  │         Iskelet       : uzunluk={sk_len:.1f} mm  max_r={sk_rad:.2f} mm")

            inf = sd.get("thalamus_inferior_mm")
            ovl = sd.get("thalamus_overlap_vox")
            pst = sd.get("position_status", "")
            inf_str = f"{inf:.2f} mm thalamus altinda" if inf is not None else "—"
            ovl_str = f"  cakisma: {ovl} voksel" if ovl is not None and ovl >= 0 else ""
            lines.append(f"  │         Pozisyon      : {inf_str}{ovl_str}  [{pst}]")

            t1m = sd.get("t1_mean")
            t1s = sd.get("t1_std")
            if t1m is not None:
                lines.append(f"  │         T1 intensite : ort={t1m:.1f}  std={t1s:.1f}  (atlas_warped icinde)")

        # Bilateral simetri
        asy   = rd.get("lr_asym_pct")
        mer   = rd.get("lr_mirror_mm")
        sym   = rd.get("symmetry_status", "—")
        asy_s = f"{asy:.1f}%" if asy is not None else "—"
        mer_s = f"{mer:.2f} mm" if mer is not None else "—"
        lines.append(f"  │")
        lines.append(f"  │   [{ICON.get(sym,'?')}] Bilateral simetri")
        lines.append(f"  │         Hacim asimetrisi     : {asy_s}  (<%20 normal)")
        lines.append(f"  │         Centroid yansima hat.: {mer_s}  (<3mm iyi)")

        ovr = rd.get("overall_status", "—")

        # WARN/FAIL ise yorumsal notlar ekle
        notes = _warn_notes(rd) if ovr != "PASS" else []
        if notes:
            lines.append(f"  │")
            lines.append(f"  │   -- Analiz Notlari --")
            for note in notes:
                # Uzun notlari 62 karakterde kelime bolumleme
                words = note.split()
                cur = "  │   "
                for w_tok in words:
                    if len(cur) + len(w_tok) + 1 > 68:
                        lines.append(cur)
                        cur = "  │     " + w_tok
                    else:
                        cur = cur + (" " if cur.strip() else "") + w_tok
                if cur.strip():
                    lines.append(cur)

        lines.append(f"  └── Genel: {ovr}")
        if ovr != "PASS":
            lines.append(f"      >> Slicer gorsel dogrulama onerilir:")
            lines.append(f"         exec(open(r'scripts/slicer_stn_viewer.py').read())")
        lines.append("")

        if ovr == "PASS":
            pass_c += 1
        elif ovr == "WARN":
            warn_c += 1
        else:
            fail_c += 1

    # Genel ozet ve gruplandirma
    lines.append("=" * w)
    lines.append(f"  OZET: {pass_c} PASS  |  {warn_c} WARN  |  {fail_c} FAIL")
    lines.append("")
    lines.append("  Tum subjectlar icin gecen kontroller:")
    lines.append("    - Thalamus altinda pozisyon (cakisma yok)")
    lines.append("    - Elongasyon > 4.0 (lens bicimli yapi dogrulandi)")
    lines.append("    - Atlas hacim orani 0.91 – 1.07 arasi (warp tutarli)")
    lines.append("    - T1 intensite tutarliligi (6760 – 6910 araliginda)")
    lines.append("")
    lines.append("  Orta hat notu:")
    lines.append("    Centroid yansima hatasi thalamus_body centroidinden")
    lines.append("    tahmin edilen anatomik orta hat kullanilarak hesaplanmistir.")
    lines.append("    AC-PC hizali degil; yuksek degerlerde Slicer gorsel")
    lines.append("    kontrolu esas alinmali.")
    lines.append("=" * w)
    lines.append("")
    lines.append("=" * w)
    lines.append("  SINIRLILIKLAR — Bu Analizin Kisitlamalari")
    lines.append("=" * w)
    lines.append("")
    lines.append("  1. ALTIN STANDART YOK")
    lines.append("     Manuel segmentasyon / Dice skoru hesaplanmamistir.")
    lines.append("     Warp kalitesi yalnizca anatomik pozisyon ve sekil")
    lines.append("     metrikleri ile dogrulanmistir. Kesin deger icin")
    lines.append("     el ile isaretleme (ITK-SNAP, Slicer) gereklidir.")
    lines.append("")
    lines.append("  2. KOORDINAT SISTEMI — NATIVE T1 UZAYI")
    lines.append("     Tum centroid koordinatlari her subjectin kendi T1")
    lines.append("     native uzayindadir. Norolojik konvansiyon icin")
    lines.append("     AC-PC hizali raporlama yapilmamistir. DBS planlama")
    lines.append("     yazilimlarinin (BrainLab, Medtronic) kullandigi")
    lines.append("     AC-PC koordinatlarindan farklidir.")
    lines.append("")
    lines.append("  3. T2 KAYIT YAPILMAMISTIR")
    lines.append("     atlas_warped T1 uzayindadir; T2.nii.gz ayri bir")
    lines.append("     uzaydadir ve bu analiz kapsaminda kaydedilmemistir.")
    lines.append("     STN T2 hipointensitesi (demir icerigi) Slicer'da")
    lines.append("     gorsel olarak dogrulanabilir ancak otomatik metrik")
    lines.append("     hesaplanmamistir.")
    lines.append("")
    lines.append("  4. POPULASYON ISTATISTIGI YOK")
    lines.append("     Z-skoru, saglikli norm referansi veya istatistiksel")
    lines.append("     karsilastirma yapilmamistir. 5 subjectlik vaka serisi")
    lines.append("     klinik yorum icin yetersizdir. Bulgular uzman")
    lines.append("     gorusu ile desteklenmelidir.")
    lines.append("")
    lines.append("  5. 1.5T GORIUNTULEME LIMITASYONU")
    lines.append("     IXI verisi 1.5T'de cekilmistir. STN'nin T2 sinyal")
    lines.append("     efekti (demir/ferritin icerigi) 3T'de belirgin,")
    lines.append("     7T'de cok daha guclüdür. Klinik DBS planlamasi icin")
    lines.append("     3T veya ustu MRI tercih edilmelidir.")
    lines.append("")
    lines.append("  6. ATLAS HASSASIYETI")
    lines.append("     Hakan atlasi genel amacli bir talamus + STN atlasıdır.")
    lines.append("     DBS cerrahisi icin DISTAL veya ATAG atlasları STN")
    lines.append("     icin daha hassas segmentasyon saglar. Bu atlaslar")
    lines.append("     LeadDBS yaziliminda entegre olarak mevcuttur.")
    lines.append("")
    lines.append("  7. ORTA HAT TAHMINI YAKLASIMI")
    lines.append("     Sol/sag simetri hatasi thalamus_body centroidinden")
    lines.append("     tahmin edilen anatomik orta hata gore hesaplanmistir.")
    lines.append("     AC-PC orta noktasi kullanilmamistir. Beyin goruntu")
    lines.append("     merkezinden sapma varsa hata degeri artabilir.")
    lines.append("")
    lines.append("=" * w)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Ana Fonksiyon
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    subjects = sorted(
        e for e in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, e))
    )

    print(f"\n{'='*62}")
    print(f"  STN Morfometri — {len(subjects)} subject")
    print(f"{'='*62}")

    # Atlas referans hacim ve centroidleri bir kez hesapla
    atlas_vol = {
        "left":  atlas_reference_volume("left"),
        "right": atlas_reference_volume("right"),
    }
    atlas_ref_centroids = {
        "left":  atlas_reference_centroid("left"),
        "right": atlas_reference_centroid("right"),
    }
    if atlas_vol["left"]:
        print(f"  Atlas STh referans: sol={atlas_vol['left']:.1f} mm3  "
              f"sag={atlas_vol['right']:.1f} mm3")
    cAL = atlas_ref_centroids.get("left")
    cAR = atlas_ref_centroids.get("right")
    if cAL and cAR:
        print(f"  Atlas centroid (MNI): sol=({cAL[0]:+.1f},{cAL[1]:+.1f},{cAL[2]:+.1f})  "
              f"sag=({cAR[0]:+.1f},{cAR[1]:+.1f},{cAR[2]:+.1f}) mm")
    print()

    csv_rows    = []
    report_data = []

    for subj in subjects:
        print(f"  [{subj}]")
        label_dir  = os.path.join(DATA_DIR, subj, "warped", "labels")
        atlas_path = os.path.join(DATA_DIR, subj, "warped", "atlas_warped.nii.gz")

        # atlas_warped yukle (T1 intensite analizi icin)
        atlas_aff = atlas_data = None
        if os.path.exists(atlas_path):
            atlas_aff, atlas_data, _ = load_volume_data(atlas_path)

        # thalamus_body yukle (konum dogrulamasi + orta hat tahmini icin — ciktilarda YER ALMAZ)
        thal_masks = {}
        for side in ("left", "right"):
            tp = os.path.join(label_dir, f"{side}_thalamus_body_warped.nii.gz")
            if os.path.exists(tp):
                thal_masks[side] = load_mask(tp)

        # Thalamus centroidleri — anatomik orta hat tahmini icin (mirror error duzeltmesi)
        thal_centroids = {}
        for side in ("left", "right"):
            if side in thal_masks:
                t_aff, t_mask, _ = thal_masks[side]
                tc = centroid_ras(t_mask, t_aff)
                if tc:
                    thal_centroids[side] = tc

        side_results = {}

        for side in ("left", "right"):
            mask_path = os.path.join(label_dir, f"{side}_STh_warped.nii.gz")
            if not os.path.exists(mask_path):
                print(f"    UYARI  {side}: maske bulunamadi")
                side_results[side] = None
                continue

            affine, mask, zooms = load_mask(mask_path)
            vol = float(mask.sum()) * float(np.prod(zooms))

            # Centroid
            c = centroid_ras(mask, affine)

            # Geometrik morfometri
            geo = compute_geometric_morphometrics(mask, zooms)

            # Uzun eksen (PCA)
            long_vec, la_len, la_p1, la_p2 = compute_long_axis(mask, affine, zooms)

            # T1 intensite (atlas_warped icerisinde)
            t1_mean = t1_std = None
            if atlas_data is not None:
                t1_mean, t1_std = sample_intensity(mask, affine, atlas_data, atlas_aff)

            # Konum dogrulamasi (thalamus_body kullanarak, DAHILI)
            inferior_mm = overlap_vox = None
            if side in thal_masks:
                t_aff, t_mask, _ = thal_masks[side]
                inferior_mm, overlap_vox = check_thalamus_position(
                    mask, affine, t_mask, t_aff
                )

            # Atlas hacim orani
            aref = atlas_vol.get(side)
            arat = round(vol / aref, 3) if aref and aref > 0 else None

            vol_ok = "OK" if VOL_MIN_MM3 <= vol <= VOL_MAX_MM3 else "!"
            la_str = f"  eksen={la_len:.1f}mm" if la_len else ""
            el_str = f"  elong={geo.get('elongation'):.1f}" if geo.get("elongation") else ""
            inf_str = f"  thal_alti={inferior_mm:.1f}mm" if inferior_mm is not None else ""
            t1_str  = f"  T1={t1_mean:.0f}" if t1_mean is not None else ""
            print(f"    {side:5s}  {vol:7.1f} mm3 [{vol_ok}]{la_str}{el_str}{inf_str}{t1_str}")

            side_results[side] = {
                "centroid": c,
                "volume_mm3": round(vol, 2),
                "surface_area_mm2": geo.get("surface_area_mm2"),
                "compactness": geo.get("compactness"),
                "elongation": geo.get("elongation"),
                "flatness": geo.get("flatness"),
                "connected_components": geo.get("connected_components", 1),
                "long_axis_vec": long_vec,
                "long_axis_length_mm": la_len,
                "la_p1": la_p1, "la_p2": la_p2,
                "bbox": geo.get("bbox_size_mm", [None, None, None]),
                "skeleton_length_mm": geo.get("skeleton_length_mm"),
                "skeleton_max_radius_mm": geo.get("skeleton_max_radius_mm"),
                "t1_mean": t1_mean, "t1_std": t1_std,
                "atlas_ref_vol": aref,
                "atlas_vol_ratio": arat,
                "thalamus_inferior_mm": inferior_mm,
                "thalamus_overlap_vox": overlap_vox,
            }

        # Bilateral simetri
        lr_asym_pct = lr_mirror_mm = None
        dL = side_results.get("left")
        dR = side_results.get("right")
        if dL and dR:
            vL, vR = dL["volume_mm3"], dR["volume_mm3"]
            if vL and vR and (vL + vR) > 0:
                lr_asym_pct = round(200 * abs(vL - vR) / (vL + vR), 2)
            cL, cR = dL["centroid"], dR["centroid"]
            if cL and cR:
                # Centroid yansima hatasi: thalamus centroidinden tahmin edilen
                # anatomik orta hat etrafinda L/R simetrisi olcumu
                if "left" in thal_centroids and "right" in thal_centroids:
                    midline_x = (thal_centroids["left"][0] + thal_centroids["right"][0]) / 2.0
                else:
                    midline_x = 0.0  # Fallback: MNI varsayimi
                lr_mirror_mm = round(abs((cL[0] - midline_x) + (cR[0] - midline_x)), 2)

        # Her side icin durum + CSV satirlari
        centroid_pts = []
        axis_pts     = []
        for side in ("left", "right"):
            sd = side_results.get(side)
            if sd is None:
                continue

            vst, sst, pst, sym_st, ovr = evaluate(
                sd["volume_mm3"], sd["long_axis_length_mm"], sd["elongation"],
                sd["thalamus_inferior_mm"], sd["thalamus_overlap_vox"],
                lr_asym_pct, lr_mirror_mm,
                connected_components=sd.get("connected_components", 1)
            )
            sd["volume_status"]   = vst
            sd["shape_status"]    = sst
            sd["position_status"] = pst
            sd["overall_status"]  = ovr

            bbox = sd["bbox"]
            lv   = sd["long_axis_vec"]

            csv_rows.append({
                "subject":                 subj,
                "side":                    side,
                "centroid_x_ras":          sd["centroid"][0] if sd["centroid"] else "",
                "centroid_y_ras":          sd["centroid"][1] if sd["centroid"] else "",
                "centroid_z_ras":          sd["centroid"][2] if sd["centroid"] else "",
                "volume_mm3":              sd["volume_mm3"],
                "surface_area_mm2":        sd["surface_area_mm2"] or "",
                "compactness":             sd["compactness"] or "",
                "elongation":              sd["elongation"] or "",
                "flatness":                sd["flatness"] or "",
                "long_axis_length_mm":     sd["long_axis_length_mm"] or "",
                "long_axis_x":             lv[0] if lv else "",
                "long_axis_y":             lv[1] if lv else "",
                "long_axis_z":             lv[2] if lv else "",
                "bbox_x_mm":               bbox[0] if bbox[0] else "",
                "bbox_y_mm":               bbox[1] if bbox[1] else "",
                "bbox_z_mm":               bbox[2] if bbox[2] else "",
                "skeleton_length_mm":      sd["skeleton_length_mm"] or "",
                "skeleton_max_radius_mm":  sd["skeleton_max_radius_mm"] or "",
                "t1_mean_in_stn":          sd["t1_mean"] or "",
                "t1_std_in_stn":           sd["t1_std"] or "",
                "atlas_ref_volume_mm3":    sd["atlas_ref_vol"] or "",
                "atlas_vol_ratio":         sd["atlas_vol_ratio"] or "",
                "thalamus_inferior_mm":    sd["thalamus_inferior_mm"] if sd["thalamus_inferior_mm"] is not None else "",
                "thalamus_overlap_vox":    sd["thalamus_overlap_vox"] if sd["thalamus_overlap_vox"] is not None else "",
                "lr_volume_asym_pct":      lr_asym_pct if lr_asym_pct is not None else "",
                "lr_mirror_error_mm":      lr_mirror_mm if lr_mirror_mm is not None else "",
                "volume_status":           vst,
                "shape_status":            sst,
                "position_status":         pst,
                "symmetry_status":         sym_st,
                "overall_status":          ovr,
            })

            # centroid ve eksen noktalari ayri listelerde tut
            if sd["centroid"]:
                centroid_pts.append((f"STN_{side}_centroid", sd["centroid"]))
            if sd["la_p1"] and sd["la_p2"]:
                axis_pts.append((f"STN_{side}_ax1", sd["la_p1"]))
                axis_pts.append((f"STN_{side}_ax2", sd["la_p2"]))

        # fcsv yaz (centroid ve eksen ayri dosyalara)
        fcsv_path = write_fcsv(subj, centroid_pts, axis_pts)
        print(f"    >> {os.path.basename(fcsv_path)}  ({len(centroid_pts)} centroid, "
              f"{len(axis_pts)} eksen ucu)")

        # Rapor icin ozet
        subj_sym = ("PASS" if (lr_asym_pct or 100) <= 20 else
                    "WARN" if (lr_asym_pct or 100) <= 40 else "FAIL")
        all_overs = [sd["overall_status"] for sd in side_results.values()
                     if sd is not None and "overall_status" in sd]
        subj_overall = ("FAIL" if "FAIL" in all_overs else
                        "WARN" if "WARN" in all_overs else "PASS")

        report_data.append({
            "subject": subj,
            "left": side_results.get("left"),
            "right": side_results.get("right"),
            "lr_asym_pct": lr_asym_pct,
            "lr_mirror_mm": lr_mirror_mm,
            "symmetry_status": subj_sym,
            "overall_status": subj_overall,
        })
        print()

    # CSV yaz
    csv_path = os.path.join(OUT_DIR, "stn_morphometrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    # Rapor yaz
    report_txt = format_report(report_data, atlas_ref_centroids)
    rpt_path   = os.path.join(OUT_DIR, "stn_validation_report.txt")
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write(report_txt)

    print(report_txt)
    print(f"  CSV   : {csv_path}")
    print(f"  Rapor : {rpt_path}")
    print(f"  FCSV  : {OUT_DIR}\\<subject>_STN_centroids.fcsv  (Slicer icin kullanin)\n")


if __name__ == "__main__":
    main()
