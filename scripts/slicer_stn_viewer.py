"""
STN 3D Slicer Viewer  —  En iyi görselleştirme
================================================
Duzen:
  Sol   — 2D Dilim  (yon degistirme butonu ayri pencerede)
  Orta  — 3D Sahne  (sol STN=kirmizi, sag STN=mavi, thalamus=gri)
  Sag   — Centroid  (coronal, her iki STN centroidi gozukur)

Ekstra katmanlar:
  - Thalamus body: yari saydam gri (anatomik referans)
  - Bilateral cetvel: L/R centroid arasi mesafe (sari cizgi)
  - Dogrulama durumu konsola yazilir

Arkaplan: atlas_warped.nii.gz  (label koordinatlariyla garantili ayni uzay)

Slicer Python konsolundan calistirin:
    SUBJECT = "IXI002-Guys-0828"
    exec(open(r'C:\\Users\\ahmet\\Desktop\\BrainSeg-20260327T093216Z-1-001\\scripts\\slicer_stn_viewer.py').read())
"""

# ─────────────────────────────────────────────────────────────────────────────
# AYARLAR — burayi degistirin
# ─────────────────────────────────────────────────────────────────────────────
if "SUBJECT" not in dir():
    SUBJECT = "IXI002-Guys-0828"

# ─────────────────────────────────────────────────────────────────────────────
import os
import re
import slicer
import qt

try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False

PROJECT_DIR = r"C:\Users\ahmet\Desktop\BrainSeg-20260327T093216Z-1-001"
DATA_DIR    = os.path.join(PROJECT_DIR, "data",    "subjects", SUBJECT)
SHAPES_DIR  = os.path.join(PROJECT_DIR, "outputs", "shapes",   SUBJECT)
STN_DIR     = os.path.join(PROJECT_DIR, "outputs", "stn")

ATLAS_PATH  = os.path.join(DATA_DIR,   "warped", "atlas_warped.nii.gz")
STH_L_VTK   = os.path.join(SHAPES_DIR, "STh_left.vtk")
STH_R_VTK   = os.path.join(SHAPES_DIR, "STh_right.vtk")
THAL_L_VTK  = os.path.join(SHAPES_DIR, "thalamus_body_left.vtk")
THAL_R_VTK  = os.path.join(SHAPES_DIR, "thalamus_body_right.vtk")
FCSV_PATH   = os.path.join(STN_DIR,    f"{SUBJECT}_STN_centroids.fcsv")
RPT_PATH    = os.path.join(STN_DIR,    "stn_validation_report.txt")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Sahneyi temizle
# ─────────────────────────────────────────────────────────────────────────────
slicer.mrmlScene.Clear(0)
lm = slicer.app.layoutManager()
print(f"\n[STN Viewer] Subject: {SUBJECT}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Ozel 3 panelli duzen
#    Sol (Red=Axial) | Orta (3D) | Sag (Yellow=Coronal/Centroid)
# ─────────────────────────────────────────────────────────────────────────────
LAYOUT_ID = 5052
layout_xml = (
    '<layout type="horizontal" split="true">'
    '  <item splitSize="380">'
    '    <view class="vtkMRMLSliceNode" singletontag="Red">'
    '      <property name="orientation" action="default">Axial</property>'
    '      <property name="viewlabel" action="default">2D</property>'
    '      <property name="viewcolor" action="default">#E84040</property>'
    '    </view>'
    '  </item>'
    '  <item splitSize="500">'
    '    <view class="vtkMRMLViewNode" singletontag="1">'
    '      <property name="viewlabel" action="default">3D</property>'
    '    </view>'
    '  </item>'
    '  <item splitSize="360">'
    '    <view class="vtkMRMLSliceNode" singletontag="Yellow">'
    '      <property name="orientation" action="default">Coronal</property>'
    '      <property name="viewlabel" action="default">Centroid</property>'
    '      <property name="viewcolor" action="default">#D4A017</property>'
    '    </view>'
    '  </item>'
    '</layout>'
)
ln = lm.layoutLogic().GetLayoutNode()
if not ln.IsLayoutDescription(LAYOUT_ID):
    ln.AddLayoutDescription(LAYOUT_ID, layout_xml)
lm.setLayout(LAYOUT_ID)

# ─────────────────────────────────────────────────────────────────────────────
# 3. atlas_warped yukle  (label koordinatlariyla ayni uzay — guvenli)
# ─────────────────────────────────────────────────────────────────────────────
bg_node = None
if os.path.exists(ATLAS_PATH):
    bg_node = slicer.util.loadVolume(ATLAS_PATH)
    bg_node.SetName(f"AtlasWarped_{SUBJECT}")

    try:
        arr = slicer.util.arrayFromVolume(bg_node)
        nz  = arr[arr > arr.max() * 0.05]
        d   = bg_node.GetDisplayNode()
        d.SetAutoWindowLevel(False)
        d.SetWindow(float(nz.max() - nz.min()) * 0.6)
        d.SetLevel(float(nz.mean()))
    except Exception:
        bg_node.GetDisplayNode().SetAutoWindowLevel(True)

    for color in ("Red", "Yellow"):
        comp = lm.sliceWidget(color).sliceLogic().GetSliceCompositeNode()
        comp.SetBackgroundVolumeID(bg_node.GetID())
        comp.SetForegroundOpacity(0)
    print("  Arkaplan: atlas_warped.nii.gz")
else:
    print(f"  UYARI: atlas_warped bulunamadi: {ATLAS_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# 4a. STN mesh'leri yukle
# ─────────────────────────────────────────────────────────────────────────────
def load_mesh(path, name, rgb, opacity=0.88, slice_intersect=True):
    if not os.path.exists(path):
        print(f"  EKSIK: {os.path.basename(path)}")
        return None
    node = slicer.util.loadModel(path)
    node.SetName(name)
    d = node.GetDisplayNode()
    if d:
        d.SetColor(*rgb)
        d.SetOpacity(opacity)
        d.SetRepresentation(2)
        d.SetAmbient(0.1)
        d.SetDiffuse(0.9)
        d.SetSpecular(0.2)
        d.SetPower(25)
        d.SetBackfaceCulling(False)
        d.SetVisibility(True)
        d.SetSliceIntersectionVisibility(slice_intersect)
        if slice_intersect:
            d.SetSliceIntersectionThickness(2)
    return node

stn_l = load_mesh(STH_L_VTK, "STN_sol", (0.90, 0.10, 0.10))
stn_r = load_mesh(STH_R_VTK, "STN_sag", (0.10, 0.35, 0.90))
loaded_stn = sum(n is not None for n in [stn_l, stn_r])
print(f"  STN mesh: {loaded_stn}/2  (sol=kirmizi, sag=mavi)")

# ─────────────────────────────────────────────────────────────────────────────
# 4b. Thalamus body — anatomik referans (yari saydam gri)
# ─────────────────────────────────────────────────────────────────────────────
thal_l = load_mesh(THAL_L_VTK, "Thalamus_sol", (0.75, 0.75, 0.75),
                   opacity=0.13, slice_intersect=False)
thal_r = load_mesh(THAL_R_VTK, "Thalamus_sag", (0.75, 0.75, 0.75),
                   opacity=0.13, slice_intersect=False)
loaded_thal = sum(n is not None for n in [thal_l, thal_r])
if loaded_thal > 0:
    print(f"  Thalamus referans: {loaded_thal}/2  (yari saydam gri — anatomik baglam)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Centroid noktalari yukle
# ─────────────────────────────────────────────────────────────────────────────
markup_node     = None
centroid_coords = {}

if os.path.exists(FCSV_PATH):
    markup_node = slicer.util.loadMarkups(FCSV_PATH)
    markup_node.SetName(f"STN_centroid_{SUBJECT}")
    d = markup_node.GetDisplayNode()
    if d:
        d.SetGlyphType(13)
        d.SetGlyphScale(4.0)
        d.SetTextScale(4.5)
        d.SetColor(1.0, 0.80, 0.0)
        d.SetSelectedColor(1.0, 1.0, 0.3)
        d.SetVisibility(True)
        d.SetSliceIntersectionVisibility(True)
        d.SetSliceIntersectionThickness(3)

    for i in range(markup_node.GetNumberOfControlPoints()):
        lbl = markup_node.GetNthControlPointLabel(i)
        pos = [0.0, 0.0, 0.0]
        markup_node.GetNthControlPointPosition(i, pos)
        if "left"  in lbl:
            centroid_coords["left"]  = list(pos)
        if "right" in lbl:
            centroid_coords["right"] = list(pos)

    print(f"  Centroid: {len(centroid_coords)} nokta  "
          f"(sol={centroid_coords.get('left','?')}, sag={centroid_coords.get('right','?')})")
else:
    print(f"  UYARI: {os.path.basename(FCSV_PATH)} bulunamadi")
    print("         Once: python scripts/compute_stn_morphometrics.py")

# ─────────────────────────────────────────────────────────────────────────────
# 5b. Bilateral cetvel — L/R centroid arasi mesafe
# ─────────────────────────────────────────────────────────────────────────────
bilateral_dist_mm = None
if "left" in centroid_coords and "right" in centroid_coords:
    try:
        ruler = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
        ruler.SetName(f"STN_bilateral_{SUBJECT}")
        cL_pos = centroid_coords["left"]
        cR_pos = centroid_coords["right"]
        ruler.AddControlPoint(cL_pos)
        ruler.AddControlPoint(cR_pos)
        rd = ruler.GetDisplayNode()
        if rd:
            rd.SetColor(1.0, 0.95, 0.2)
            rd.SetSelectedColor(1.0, 1.0, 0.5)
            rd.SetTextScale(3.5)
            rd.SetGlyphScale(2.5)
            rd.SetLineThickness(0.4)
            rd.SetVisibility(True)
        if _NP:
            bilateral_dist_mm = round(
                float(np.linalg.norm(np.array(cL_pos) - np.array(cR_pos))), 2)
            print(f"  Bilateral cetvel: {bilateral_dist_mm} mm  (L-R centroid mesafesi)")
    except Exception as e:
        print(f"  Cetvel olusturulamadi: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Sol panel (Red/2D): sol STN centroidine konumlan
# ─────────────────────────────────────────────────────────────────────────────
target = centroid_coords.get("left") or centroid_coords.get("right")
if target is None and stn_l:
    b = [0.0]*6
    stn_l.GetRASBounds(b)
    target = [(b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2]

if target:
    sn_red = lm.sliceWidget("Red").mrmlSliceNode()
    sn_red.JumpSlice(*target)
    print(f"  Sol panel (Axial) → STN merkezi: "
          f"({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}) mm")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Sag panel (Yellow/Centroid): coronal, her iki STN gorunsun
# ─────────────────────────────────────────────────────────────────────────────
sn_yellow = lm.sliceWidget("Yellow").mrmlSliceNode()
sn_yellow.SetOrientationToCoronal()

if "left" in centroid_coords and "right" in centroid_coords:
    cL = centroid_coords["left"]
    cR = centroid_coords["right"]
    mid = [(cL[i] + cR[i]) / 2 for i in range(3)]
    sn_yellow.JumpSlice(*mid)
    lat_span = abs(cL[0] - cR[0]) + 30.0
    fov = max(lat_span, 60.0)
    sn_yellow.SetFieldOfView(fov, fov, 1)
elif target:
    sn_yellow.JumpSlice(*target)
    sn_yellow.SetFieldOfView(70, 70, 1)

sn_yellow.UpdateMatrices()

# ─────────────────────────────────────────────────────────────────────────────
# 8. 3D gorunum: koyu arkaplan, anterior kamera
# ─────────────────────────────────────────────────────────────────────────────
try:
    view3d = lm.threeDWidget(0).threeDView()
    ren    = view3d.renderWindow().GetRenderers().GetFirstRenderer()
    if ren:
        ren.SetBackground(0.12, 0.12, 0.16)
        ren.SetBackground2(0.22, 0.22, 0.30)
        ren.SetGradientBackground(True)
        ren.ResetCamera()
    view3d.rotateToViewAxis(1)
    view3d.resetFocalPoint()
except Exception as e:
    print(f"  3D gorunum ayari atlandi: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Dogrulama durumu — rapor dosyasindan oku
# ─────────────────────────────────────────────────────────────────────────────
validation_status = None
if os.path.exists(RPT_PATH):
    try:
        with open(RPT_PATH, encoding="utf-8") as f:
            rpt = f.read()
        m = re.search(
            r'──\s*' + re.escape(SUBJECT) + r'.*?└──\s*Genel:\s*(\w+)',
            rpt, re.DOTALL
        )
        if m:
            validation_status = m.group(1)
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# 10. Yon Degistirme Butonu (sol 2D panel icin)
# ─────────────────────────────────────────────────────────────────────────────
def set_slice_orientation(orient):
    sn = lm.sliceWidget("Red").mrmlSliceNode()
    {"Axial":    sn.SetOrientationToAxial,
     "Sagittal": sn.SetOrientationToSagittal,
     "Coronal":  sn.SetOrientationToCoronal}[orient]()
    if target:
        sn.JumpSlice(*target)

btn_panel = qt.QWidget()
btn_panel.setWindowTitle(f"2D Dilim Yonu — {SUBJECT}")
btn_panel.setWindowFlags(qt.Qt.Tool | qt.Qt.WindowStaysOnTopHint)
btn_panel.setAttribute(qt.Qt.WA_DeleteOnClose, False)

outer = qt.QVBoxLayout(btn_panel)
outer.setContentsMargins(10, 10, 10, 10)
outer.setSpacing(8)

title = qt.QLabel("Sol panel yonunu sec:")
title.setStyleSheet("font-weight:bold; font-size:13px; color:#ddd;")
btn_panel.setStyleSheet("background:#2b2b2b;")
outer.addWidget(title)

row = qt.QHBoxLayout()
row.setSpacing(6)
for label, orient in [("Ust\n(Axial)", "Axial"),
                       ("Yan\n(Sagittal)", "Sagittal"),
                       ("On\n(Coronal)", "Coronal")]:
    btn = qt.QPushButton(label)
    btn.setFixedSize(82, 48)
    btn.setStyleSheet(
        "QPushButton{"
        "  background:#3a6ea8; color:white; border-radius:6px;"
        "  font-size:12px; font-weight:bold;}"
        "QPushButton:hover{background:#4d87c7;}"
        "QPushButton:pressed{background:#2a5080;}")
    btn.clicked.connect(lambda _, o=orient: set_slice_orientation(o))
    row.addWidget(btn)

outer.addLayout(row)

hint = qt.QLabel("Sag panel daima Coronal kalir.")
hint.setStyleSheet("font-size:10px; color:#888;")
outer.addWidget(hint)

btn_panel.adjustSize()

mw = slicer.util.mainWindow()
if mw:
    geo  = mw.geometry
    pw   = btn_panel.width
    ph   = btn_panel.height
    btn_panel.move(geo.x() + geo.width() - pw - 20,
                   geo.y() + geo.height() - ph - 60)
btn_panel.show()

# ─────────────────────────────────────────────────────────────────────────────
# 11. Ozet
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 56)
print(f"  STN VIEWER HAZIR — {SUBJECT}")
print("=" * 56)
vs = validation_status or "—"
vs_mark = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}.get(vs, f"[{vs}]")
print(f"  Dogrulama durumu : {vs_mark}")
print(f"  STN mesh  : {loaded_stn}/2   (sol=KIRMIZI  sag=MAVI)")
if loaded_thal > 0:
    print(f"  Thalamus  : {loaded_thal}/2   (yari saydam GRI — anatomik baglam)")
print(f"  Centroid  : ALTIN SARI noktalar")
if bilateral_dist_mm is not None:
    print(f"  L-R mesafe: {bilateral_dist_mm} mm  (sari cetvel)")
print()
print("  Panel: Sol=2D (Axial)  |  Orta=3D (Anterior)  |  Sag=Coronal")
print()
if centroid_coords:
    print("  Centroid koordinatlari (atlas_warped RAS mm):")
    for side in ("left", "right"):
        c = centroid_coords.get(side)
        if c:
            print(f"    {side:5s}: ({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f}) mm")
print()
print("  Kontrol edilecekler:")
print("    [ ] 3D'de sol/sag STN ayni yukseklikte mi?")
print("    [ ] Thalamus gri mesh STN'nin hemen ustunde mi?")
print("    [ ] Sari cetvel saglikli aralikta mi? (beklenen ~20-30 mm)")
print("    [ ] Coronal panelde altin noktalar mesh icinde mi?")
if validation_status and validation_status != "PASS":
    print(f"\n  >> {validation_status} — Slicer gorsel dogrulamasi yapın!")
    print(f"     Detay: {RPT_PATH}")
print()
print("  Baska subject icin: SUBJECT = '...' duzelt ve tekrar calistir.")
print("=" * 56 + "\n")
