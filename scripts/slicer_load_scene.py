"""
3D Slicer Scene Loader — Thalamic Nuclei Morphometrics
=======================================================
Bu script 3D Slicer'in Python konsolunda calistirilir.

Kullanim (Slicer Python konsolunda):
    exec(open(r"C:/Users/ahmet/Desktop/BrainSeg-20260327T093216Z-1-001/scripts/slicer_load_scene.py").read())

Ne yapar:
  1. Atlas_warped.nii.gz → Volume Rendering ile yari seffaf 3D beyin
  2. Her thalamic nucleus → renkli 3D VTK mesh
  3. Her label icin ayri renkli pin noktasi (sol + sag centroid)

SUBJECT degiskenini degistirerek farkli subjectleri yukleyebilirsin.
"""

import os
import slicer
import vtk
try:
    import qt
except ImportError:
    qt = None

# ─────────────────────────────────────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────────────────────────────────────

SUBJECT = "IXI002-Guys-0828"
# Diger: IXI012-HH-1211 / IXI013-HH-1212 / IXI015-HH-1258 / IXI016-Guys-0697

BASE_DIR   = r"C:/Users/ahmet/Desktop/BrainSeg-20260327T093216Z-1-001"
SHAPES_DIR = os.path.join(BASE_DIR, "outputs", "shapes", SUBJECT)
FCSV_PATH      = os.path.join(BASE_DIR, "outputs", "morphometrics", f"{SUBJECT}.fcsv")
SURFACE_FCSV   = os.path.join(BASE_DIR, "outputs", "morphometrics", f"{SUBJECT}_surface.fcsv")
ATLAS_PATH = os.path.join(BASE_DIR, "data", "subjects", SUBJECT, "warped", "atlas_warped.nii.gz")

MESH_OPACITY   = 0.75   # nucleus mesh seffafligi
BRAIN_OPACITY  = 0.25   # beyin yuzeyinin seffafligi — dusuk tutulur ki nuclei iceriden gorunsun
PIN_SCALE        = 3.0    # centroid nokta boyutu (mesh icinde)
PIN_TEXT_SCALE   = 0.0    # centroid etiketi gizli (kalabalik onlenir)
SURF_PIN_SCALE   = 2.5    # yuzey nokta boyutu (mesh uzerinde)
SURF_TEXT_SCALE  = 2.5    # yuzey nokta etiketi: "VApc_left_Ust" gibi gorunur

# ─────────────────────────────────────────────────────────────────────────────
# Her label icin renk paleti (39 nucleus)
# ─────────────────────────────────────────────────────────────────────────────
LABEL_COLORS = {
    "AD":           (0.902, 0.224, 0.224),
    "AM":           (0.949, 0.514, 0.196),
    "AV":           (0.965, 0.765, 0.137),
    "CeM":          (0.706, 0.863, 0.173),
    "CL":           (0.400, 0.800, 0.400),
    "CM":           (0.196, 0.784, 0.549),
    "Hb":           (0.118, 0.745, 0.745),
    "LD":           (0.137, 0.659, 0.886),
    "LGNmc":        (0.231, 0.455, 0.875),
    "LGNpc":        (0.420, 0.275, 0.820),
    "Li":           (0.608, 0.251, 0.769),
    "LP":           (0.820, 0.208, 0.620),
    "MDmc":         (0.941, 0.200, 0.420),
    "MDpc":         (0.820, 0.157, 0.157),
    "MGN":          (0.710, 0.420, 0.200),
    "MV":           (0.600, 0.600, 0.100),
    "Pf":           (0.350, 0.700, 0.250),
    "Po":           (0.100, 0.650, 0.500),
    "PuA":          (0.200, 0.800, 0.700),
    "PuI":          (0.050, 0.600, 0.900),
    "PuL":          (0.200, 0.400, 0.800),
    "PuM":          (0.500, 0.200, 0.700),
    "Pv":           (0.750, 0.100, 0.550),
    "RN":           (1.000, 0.300, 0.100),
    "SG":           (0.900, 0.700, 0.100),
    "sPf":          (0.500, 0.850, 0.200),
    "STh":          (0.100, 0.750, 0.350),
    "thalamus_body":(0.700, 0.700, 0.700),
    "VAmc":         (0.300, 0.600, 0.900),
    "VApc":         (0.200, 0.300, 0.800),
    "VLa":          (0.600, 0.200, 0.600),
    "VLpd":         (0.800, 0.200, 0.400),
    "VLpv":         (0.900, 0.500, 0.200),
    "VM":           (0.950, 0.800, 0.300),
    "VPI":          (0.450, 0.800, 0.450),
    "VPLa":         (0.200, 0.700, 0.500),
    "VPLp":         (0.100, 0.500, 0.700),
    "VPM":          (0.500, 0.100, 0.700),
    "mtt":          (0.800, 0.400, 0.100),
}

def _lighten(rgb, factor=0.45):
    return tuple(min(1.0, c + (1.0 - c) * factor) for c in rgb)


# ─────────────────────────────────────────────────────────────────────────────
# .fcsv Okuma — label bazli grupla
# ─────────────────────────────────────────────────────────────────────────────
def read_fcsv_by_label(path):
    """
    Centroid .fcsv: her label icin {label: {side: [x,y,z]}} donduruR.
    """
    groups = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split(",")
            if len(parts) < 12:
                continue
            x, y, z   = float(parts[1]), float(parts[2]), float(parts[3])
            lm_label   = parts[11]

            if lm_label.endswith("_left"):
                side  = "left"
                label = lm_label[:-5]
            elif lm_label.endswith("_right"):
                side  = "right"
                label = lm_label[:-6]
            else:
                continue

            if label not in groups:
                groups[label] = {}
            groups[label][side] = [x, y, z]
    return groups


def read_surface_fcsv(path):
    """
    Surface .fcsv: {label: {side: [(direction_label, [x,y,z]), ...]}} donduruR.
    Ornek label: "VApc_left_Ust" -> label=VApc, side=left, dir=Ust
    """
    groups = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split(",")
            if len(parts) < 12:
                continue
            x, y, z  = float(parts[1]), float(parts[2]), float(parts[3])
            lm_label = parts[11]   # "VApc_left_Ust"

            # side ve label'i ayir — son segment yon, oncesi side, oncesi label
            tokens = lm_label.split("_")
            # side: "left" veya "right" aramaliyiz
            side_idx = None
            for i, t in enumerate(tokens):
                if t in ("left", "right"):
                    side_idx = i
                    break
            if side_idx is None:
                continue
            label = "_".join(tokens[:side_idx])
            side  = tokens[side_idx]
            direction = "_".join(tokens[side_idx+1:]) if side_idx+1 < len(tokens) else ""

            if label not in groups:
                groups[label] = {}
            if side not in groups[label]:
                groups[label][side] = []
            groups[label][side].append((direction, [x, y, z]))
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Sahneyi Kur
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n  Sahne temizleniyor...")
slicer.mrmlScene.Clear(0)


# ── 1. Atlas Volume ──────────────────────────────────────────────────────────
volumeNode = None
if os.path.exists(ATLAS_PATH):
    print(f"  Volume yukleniyor: {os.path.basename(ATLAS_PATH)}")
    volumeNode = slicer.util.loadVolume(ATLAS_PATH)
    slicer.util.setSliceViewerLayers(background=volumeNode)
else:
    print(f"  UYARI: atlas bulunamadi: {ATLAS_PATH}")


# ── 2. Beyin Yuzey Mesh'i — Yari Seffaf Kabuk ───────────────────────────────
brain_vtk = os.path.join(SHAPES_DIR, "brain_surface.vtk")
if os.path.exists(brain_vtk):
    print("  Beyin yuzey mesh'i yukleniyor (yari seffaf)...")
    try:
        brainNode = slicer.util.loadModel(brain_vtk)
        brainNode.SetName("brain_surface")
        dn = brainNode.GetDisplayNode()
        dn.SetColor(0.85, 0.80, 0.75)     # ten rengi
        dn.SetOpacity(BRAIN_OPACITY)
        dn.SetVisibility3D(True)
        dn.SetVisibility2D(False)
        dn.SetBackfaceCulling(False)
        dn.SetAmbient(0.3)
        dn.SetDiffuse(0.7)
        print(f"  Beyin yuzey mesh'i hazir (opacity: {BRAIN_OPACITY}).")
    except Exception as e:
        print(f"  Beyin yuzeyi yuklenemedi: {e}")
else:
    print(f"  UYARI: brain_surface.vtk bulunamadi.")
    print(f"  Once 'python scripts/export_3d_shapes.py' calistirin.")


# ── 3. VTK Mesh'ler — Nucleus 3D Sekilleri ───────────────────────────────────
if not os.path.isdir(SHAPES_DIR):
    print(f"  HATA: shapes klasoru bulunamadi. Once export_3d_shapes.py calistirin.")
else:
    vtk_files = sorted(
        f for f in os.listdir(SHAPES_DIR)
        if f.endswith(".vtk") and f != "brain_surface.vtk"   # beyin yuzeyini atla
    )
    print(f"\n  {len(vtk_files)} nucleus mesh yukleniyor...")
    loaded = 0
    for fname in vtk_files:
        base  = fname.replace(".vtk", "")
        side  = "left" if base.endswith("_left") else "right"
        label = base[:-(5 if side == "left" else 6)]
        rgb   = LABEL_COLORS.get(label, (0.7, 0.7, 0.7))
        if side == "right":
            rgb = _lighten(rgb)

        try:
            node = slicer.util.loadModel(os.path.join(SHAPES_DIR, fname))
        except Exception as e:
            continue

        node.SetName(f"{label}_{side}")
        dn = node.GetDisplayNode()
        dn.SetColor(*rgb)
        dn.SetOpacity(MESH_OPACITY)
        dn.SetVisibility3D(True)
        dn.SetVisibility2D(False)
        dn.SetBackfaceCulling(False)
        loaded += 1

    print(f"  {loaded} mesh yuklendi.")


# ── 4. Pin Noktalari — Label Basina Ayri Renkli Markup Node ──────────────────
if not os.path.exists(FCSV_PATH):
    print(f"  UYARI: .fcsv bulunamadi. Once compute_morphometrics.py calistirin.")
else:
    print(f"\n  Renkli pin noktalari olusturuluyor...")
    label_groups = read_fcsv_by_label(FCSV_PATH)
    pin_count = 0

    for label in sorted(label_groups.keys()):
        sides  = label_groups[label]
        rgb    = LABEL_COLORS.get(label, (0.7, 0.7, 0.7))

        # Bu label icin tek bir markup node — sol + sag noktasi
        mNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        mNode.SetName(label)

        for side in ("left", "right"):
            if side not in sides:
                continue
            x, y, z = sides[side]
            idx = mNode.AddControlPoint([x, y, z])
            mNode.SetNthControlPointLabel(idx, f"{label}_{side}")
            pin_count += 1

        # Renk ata — mesh ile AYNI renk, etiket metni KAPALI
        # (etiket karmasasi onlenmis olur; isim sol paneldeki listeden okunur)
        dn = mNode.GetDisplayNode()
        dn.SetSelectedColor(*rgb)
        dn.SetColor(*rgb)
        dn.SetGlyphScale(PIN_SCALE)
        dn.SetTextScale(PIN_TEXT_SCALE)   # label adi pinin yaninda gorunur
        dn.SetGlyphType(13)               # 13 = Sphere3D — dolu kure
        dn.SetVisibility(True)

    print(f"  {pin_count} pin noktasi, {len(label_groups)} label, her biri ayri renkte.")
    print(f"  NOT: PIN_SCALE={PIN_SCALE}  PIN_TEXT_SCALE={PIN_TEXT_SCALE}")


# ── 5. Yuzey Landmark Noktalari — Mesh Uzerinde 6 Anatomik Uc Nokta ──────────
if not os.path.exists(SURFACE_FCSV):
    print(f"\n  UYARI: surface.fcsv bulunamadi.")
    print(f"  Once 'python scripts/compute_morphometrics.py' calistirin.")
else:
    print(f"\n  Yuzey landmark noktalari yukleniyor...")
    surf_groups = read_surface_fcsv(SURFACE_FCSV)
    surf_count  = 0

    for label in sorted(surf_groups.keys()):
        rgb = LABEL_COLORS.get(label, (0.7, 0.7, 0.7))

        # Bu label icin tek bir markup node — tum yon noktalari
        mNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        mNode.SetName(f"{label}_surface")

        for side in ("left", "right"):
            if side not in surf_groups[label]:
                continue
            side_rgb = rgb if side == "left" else _lighten(rgb)
            for direction, coords in surf_groups[label][side]:
                x, y, z = coords
                idx = mNode.AddControlPoint([x, y, z])
                mNode.SetNthControlPointLabel(idx, f"{label}_{side}_{direction}")
                surf_count += 1

        dn = mNode.GetDisplayNode()
        dn.SetSelectedColor(1.0, 1.0, 0.0)   # seciliyken sari
        dn.SetColor(1.0, 0.9, 0.0)            # sari — her renkteki mesh uzerinde gorulur
        dn.SetGlyphScale(SURF_PIN_SCALE)
        dn.SetTextScale(SURF_TEXT_SCALE)
        dn.SetGlyphType(13)          # Sphere3D
        dn.SetVisibility(True)

    print(f"  {surf_count} yuzey nokta yuklendi ({len(surf_groups)} label, her biri 6 yon x 2 taraf).")
    print(f"  Etiketler: <label>_<taraf>_<yon>  ornek: VApc_left_Ust")


# ── 6. Gorunum Ayarlari ───────────────────────────────────────────────────────
layoutManager = slicer.app.layoutManager()
layoutManager.setLayout(
    slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView
)

threeDWidget = layoutManager.threeDWidget(0)
threeDWidget.threeDView().resetFocalPoint()
threeDWidget.threeDView().rotateToViewAxis(3)

# 3D pencere arka plan rengini siyah yap
try:
    if qt:
        threeDWidget.threeDView().setBackgroundColor(
            qt.QColor(10, 10, 10), qt.QColor(30, 30, 30)
        )
except Exception:
    pass

print(f"\n  {'='*52}")
print(f"  HAZIR -- {SUBJECT}")
print(f"  {'='*52}")
print(f"  Sag ust: yari seffaf beyin + icinde renkli nucleus sekilleri + pinler")
print(f"  Beyin seffafligi: %{int(BRAIN_OPACITY*100)}  (brain_surface.vtk)")
print(f"  Daha seffaf: BRAIN_OPACITY = 0.05 | Daha opak: BRAIN_OPACITY = 0.30")
print(f"  Nuclei gorunmuyor mu? Modeli cevirerek bakmayi dene.\n")
