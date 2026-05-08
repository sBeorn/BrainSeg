"""
STN 3D Slicer Viewer  —  Subject Secici + Ozvektör Kontrolü
=============================================================
Kontrol paneli (sol menüde - Slicer Modül Alanı):
  - Subject dropdown + Yükle butonu
  - Sol panel yön seçici (Axial/Sagittal/Coronal)
  - Uzun eksen (özvektör) sol/sag goster/gizle + vektör degerler

Ekranda:
  Sol   — 2D Dilim
  Orta  — 3D Sahne  (STN kirmizi/mavi, thalamus gri, eksenler cyan/yesil)
  Sag   — Coronal centroid paneli

Kullanim:
    SUBJECT = "IXI002-Guys-0828"
    exec(open(r'C:\\Users\\ahmet\\Desktop\\BrainSeg-20260327T093216Z-1-001\\scripts\\slicer_stn_viewer.py').read())
"""

# ─────────────────────────────────────────────────────────
# BASLANGIÇ SUBJECT (konsola önceden yazilmissa korunur)
# ─────────────────────────────────────────────────────────
if "SUBJECT" not in dir():
    SUBJECT = "IXI002-Guys-0828"

# ─────────────────────────────────────────────────────────
import os, re
import slicer
import qt

try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False

PROJECT_DIR = r"C:\Users\ahmet\Desktop\BrainSeg-20260327T093216Z-1-001"
DATA_BASE   = os.path.join(PROJECT_DIR, "data",    "subjects")
SHAPES_BASE = os.path.join(PROJECT_DIR, "outputs", "shapes")
STN_DIR     = os.path.join(PROJECT_DIR, "outputs", "stn")
RPT_PATH    = os.path.join(STN_DIR,    "stn_validation_report.txt")

SUBJECTS = sorted(
    e for e in os.listdir(DATA_BASE)
    if os.path.isdir(os.path.join(DATA_BASE, e))
)

# ─────────────────────────────────────────────────────────
# Global sahne durumu  (_g)  +  UI widget referansları (_ui)
# ─────────────────────────────────────────────────────────
_g = {
    "subject":         SUBJECT,
    "centroid_coords": {},
    "target":          None,
    "axis_nodes":      {},   # {side: vtkMRMLMarkupsLineNode}
    "axis_vecs":       {},   # {side: [x,y,z]}  — normalize birim vektör
    "axis_len":        {},   # {side: float mm}
    "axis_visible":    {"left": False, "right": False},
}
_ui = {}   # Qt widget referanslari

# ─────────────────────────────────────────────────────────
# Layout  (bir kez tanimla; her Clear sonrasi yeniden set)
# ─────────────────────────────────────────────────────────
LAYOUT_ID  = 5052
LAYOUT_XML = (
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

def _setup_layout():
    lm = slicer.app.layoutManager()
    ln = lm.layoutLogic().GetLayoutNode()
    if not ln.IsLayoutDescription(LAYOUT_ID):
        ln.AddLayoutDescription(LAYOUT_ID, LAYOUT_XML)
    lm.setLayout(LAYOUT_ID)

# ─────────────────────────────────────────────────────────
# Yardimci fonksiyonlar
# ─────────────────────────────────────────────────────────

def _parse_fcsv(path):
    """FCSV dosyasini okuyup {label: [x,y,z]} sozlugu dondur."""
    pts = {}
    if not os.path.exists(path):
        return pts
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split(",")
            if len(parts) < 12:
                continue
            lbl = parts[11]
            try:
                pts[lbl] = [float(parts[1]), float(parts[2]), float(parts[3])]
            except ValueError:
                pass
    return pts


def _load_mesh(path, name, rgb, opacity=0.88, slice_intersect=True):
    if not os.path.exists(path):
        return None
    node = slicer.util.loadModel(path)
    node.SetName(name)
    d = node.GetDisplayNode()
    if d:
        d.SetColor(*rgb)
        d.SetOpacity(opacity)
        d.SetRepresentation(2)
        d.SetAmbient(0.1); d.SetDiffuse(0.9)
        d.SetSpecular(0.2); d.SetPower(25)
        d.SetBackfaceCulling(False)
        d.SetVisibility(True)
        if hasattr(d, "SetVisibility2D"):
            d.SetVisibility2D(slice_intersect)
        else:
            d.SetSliceIntersectionVisibility(slice_intersect)
        if slice_intersect:
            d.SetSliceIntersectionThickness(2)
    return node


def _validation_status(subject):
    if not os.path.exists(RPT_PATH):
        return None
    try:
        with open(RPT_PATH, encoding="utf-8") as f:
            rpt = f.read()
        # Subject adini bul, ondan sonra gelen ilk "Genel:" satirini oku
        idx = rpt.find(subject)
        if idx < 0:
            return None
        m = re.search(r'Genel:\s*(\w+)', rpt[idx:])
        return m.group(1) if m else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────
# Sahne yükleyici  — her subject degisiminde çagrilir
# ─────────────────────────────────────────────────────────

def _load_scene(subject):
    lm = slicer.app.layoutManager()
    slicer.mrmlScene.Clear(0)
    _setup_layout()

    _g["subject"]         = subject
    _g["centroid_coords"] = {}
    _g["target"]          = None
    _g["axis_nodes"]      = {}
    _g["axis_vecs"]       = {}
    _g["axis_len"]        = {}
    # Görünürlük ayarlarini koru (kullanici daha önce açmissa açik kalsin)

    shapes_dir = os.path.join(SHAPES_BASE, subject)
    atlas_path = os.path.join(DATA_BASE, subject, "warped", "atlas_warped.nii.gz")
    fcsv_path  = os.path.join(STN_DIR, f"{subject}_STN_centroids.fcsv")
    axes_path  = os.path.join(STN_DIR, f"{subject}_STN_axes.fcsv")

    print(f"\n[STN Viewer] Yukleniyor: {subject}")

    # 1. Arkaplan: atlas_warped
    if os.path.exists(atlas_path):
        bg = slicer.util.loadVolume(atlas_path)
        bg.SetName(f"AtlasWarped_{subject}")
        try:
            arr = slicer.util.arrayFromVolume(bg)
            nz  = arr[arr > arr.max() * 0.05]
            d   = bg.GetDisplayNode()
            d.SetAutoWindowLevel(False)
            d.SetWindow(float(nz.max() - nz.min()) * 0.6)
            d.SetLevel(float(nz.mean()))
        except Exception:
            bg.GetDisplayNode().SetAutoWindowLevel(True)
        for color in ("Red", "Yellow"):
            comp = lm.sliceWidget(color).sliceLogic().GetSliceCompositeNode()
            comp.SetBackgroundVolumeID(bg.GetID())
            comp.SetForegroundOpacity(0)
        print("  Arkaplan: atlas_warped.nii.gz")
    else:
        print(f"  UYARI: atlas_warped bulunamadi")

    # 2. STN meshleri
    stn_l = _load_mesh(os.path.join(shapes_dir, "STh_left.vtk"),
                       "STN_sol", (0.90, 0.10, 0.10))
    stn_r = _load_mesh(os.path.join(shapes_dir, "STh_right.vtk"),
                       "STN_sag", (0.10, 0.35, 0.90))
    print(f"  STN mesh: {sum(x is not None for x in [stn_l,stn_r])}/2  (sol=kirmizi sag=mavi)")

    # 3. Thalamus body — yari saydam gri
    tl = _load_mesh(os.path.join(shapes_dir, "thalamus_body_left.vtk"),
                    "Thal_sol", (0.78, 0.78, 0.78), opacity=0.22, slice_intersect=False)
    tr = _load_mesh(os.path.join(shapes_dir, "thalamus_body_right.vtk"),
                    "Thal_sag", (0.78, 0.78, 0.78), opacity=0.22, slice_intersect=False)
    n_thal = sum(x is not None for x in [tl, tr])
    if n_thal:
        print(f"  Thalamus: {n_thal}/2  (gri, anatomik referans)")

    # 4. Centroid noktalari
    c_pts = _parse_fcsv(fcsv_path)
    cc = {}
    for lbl, pos in c_pts.items():
        if "left"  in lbl: cc["left"]  = pos
        if "right" in lbl: cc["right"] = pos

    if c_pts:
        mn = slicer.util.loadMarkups(fcsv_path)
        mn.SetName(f"STN_centroid_{subject}")
        d = mn.GetDisplayNode()
        if d:
            d.SetGlyphType(13); d.SetGlyphScale(3.5); d.SetTextScale(0)
            d.SetColor(1.0, 0.80, 0.0); d.SetSelectedColor(1.0, 1.0, 0.3)
            d.SetVisibility(True)
            if hasattr(d, "SetVisibility2D"):
                d.SetVisibility2D(True)
            else:
                d.SetSliceIntersectionVisibility(True)
            try:
                d.SetSliceIntersectionThickness(3)
            except Exception:
                pass
        print(f"  Centroid: {len(cc)} nokta  (altin sari)")
    else:
        print(f"  UYARI: {os.path.basename(fcsv_path)} bulunamadi")
    _g["centroid_coords"] = cc

    # 5. Bilateral cetvel (sari cizgi)
    bilateral_dist = None
    if "left" in cc and "right" in cc:
        try:
            r = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
            r.SetName(f"STN_bilateral_{subject}")
            r.AddControlPoint(*cc["left"])
            r.AddControlPoint(*cc["right"])
            rd = r.GetDisplayNode()
            if rd:
                rd.SetColor(1.0, 0.95, 0.2); rd.SetTextScale(2.0)
                rd.SetGlyphScale(2.0); rd.SetLineThickness(0.4)
                rd.SetVisibility(True)
            if _NP:
                bilateral_dist = round(float(
                    np.linalg.norm(np.array(cc["left"]) - np.array(cc["right"]))), 2)
                print(f"  L-R cetvel: {bilateral_dist} mm")
        except Exception as e:
            print(f"  Cetvel hatasi: {e}")

    # 6. Uzun eksen çizgileri (özvektör görselleştirme)
    ax_pts = _parse_fcsv(axes_path)
    for side, ak1, ak2, col in [
        ("left",  "STN_left_ax1",  "STN_left_ax2",  (0.15, 0.85, 0.85)),  # cyan
        ("right", "STN_right_ax1", "STN_right_ax2", (0.15, 0.85, 0.35)),  # yesil
    ]:
        p1 = ax_pts.get(ak1)
        p2 = ax_pts.get(ak2)
        if not (p1 and p2):
            continue
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
        node.SetName(f"STN_axis_{side}_{subject}")
        node.AddControlPoint(*p1)
        node.AddControlPoint(*p2)
        d = node.GetDisplayNode()
        if d:
            d.SetColor(*col); d.SetTextScale(2.0)
            d.SetGlyphScale(2.0); d.SetLineThickness(0.6)
            d.SetVisibility(_g["axis_visible"].get(side, False))
        _g["axis_nodes"][side] = node
        # Normalize birim vektör
        v = [p2[i] - p1[i] for i in range(3)]
        length = (sum(vi**2 for vi in v)) ** 0.5
        if length > 1e-6:
            _g["axis_vecs"][side] = [round(vi / length, 3) for vi in v]
            _g["axis_len"][side]  = round(length, 2)
            vis_str = "ACIK" if _g["axis_visible"].get(side, False) else "gizli"
            v2 = _g["axis_vecs"][side]
            print(f"  Eksen {side:5s}: {length:.1f} mm  "
                  f"({v2[0]:+.3f},{v2[1]:+.3f},{v2[2]:+.3f})  [{vis_str}]")

    # 7. 2D dilim konumlandirma
    target = cc.get("left") or cc.get("right")
    if target is None and stn_l:
        b = [0.0]*6; stn_l.GetRASBounds(b)
        target = [(b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2]
    _g["target"] = target
    if target:
        lm.sliceWidget("Red").mrmlSliceNode().JumpSlice(*target)

    # 8. Coronal panel FOV
    sn_y = lm.sliceWidget("Yellow").mrmlSliceNode()
    sn_y.SetOrientationToCoronal()
    if "left" in cc and "right" in cc:
        mid = [(cc["left"][i] + cc["right"][i]) / 2 for i in range(3)]
        sn_y.JumpSlice(*mid)
        fov = max(abs(cc["left"][0] - cc["right"][0]) + 30.0, 60.0)
        sn_y.SetFieldOfView(fov, fov, 1)
    elif target:
        sn_y.JumpSlice(*target); sn_y.SetFieldOfView(70, 70, 1)
    sn_y.UpdateMatrices()

    # 9. 3D görünüm
    try:
        v3d = lm.threeDWidget(0).threeDView()
        ren  = v3d.renderWindow().GetRenderers().GetFirstRenderer()
        if ren:
            ren.SetBackground(0.12, 0.12, 0.16)
            ren.SetBackground2(0.22, 0.22, 0.30)
            ren.SetGradientBackground(True)
            ren.ResetCamera()
        v3d.rotateToViewAxis(1)
        v3d.resetFocalPoint()
    except Exception as e:
        print(f"  3D: {e}")

    # 10. Dogrulama durumu
    vs = _validation_status(subject)
    vs_str = {"PASS":"[PASS]","WARN":"[WARN]","FAIL":"[FAIL]"}.get(vs or "", f"[{vs or '—'}]")
    print(f"  Dogrulama: {vs_str}")
    if vs and vs != "PASS":
        print(f"  >> {vs} — gorsel dogrulama onerilir!")

    # 11. Kontrol panelini güncelle
    _refresh_ui(bilateral_dist, vs)
    print(f"  Hazir.\n")


# ─────────────────────────────────────────────────────────
# Eksen görünürlük kontrolleri
# ─────────────────────────────────────────────────────────

def _set_axis_visible(side, visible):
    _g["axis_visible"][side] = visible
    node = _g["axis_nodes"].get(side)
    if node:
        d = node.GetDisplayNode()
        if d:
            d.SetVisibility(visible)
    _refresh_axis_btns()


def _toggle_all_axes():
    any_off = not all(_g["axis_visible"].get(s, False) for s in ("left", "right"))
    for side in ("left", "right"):
        _set_axis_visible(side, any_off)


def _refresh_axis_btns():
    for side in ("left", "right"):
        btn = _ui.get(f"axis_btn_{side}")
        if not btn:
            continue
        vis = _g["axis_visible"].get(side, False)
        btn.setText("Gizle " if vis else "Göster")
        btn.setStyleSheet(
            "QPushButton{background:%s;color:white;border-radius:5px;"
            "font-size:11px;font-weight:bold;padding:2px 8px;}"
            "QPushButton:hover{background:#5aaa5a;}" % ("#3a8a3a" if vis else "#555")
        )
    all_btn = _ui.get("axis_btn_all")
    if all_btn:
        all_on = all(_g["axis_visible"].get(s, False) for s in ("left","right"))
        all_btn.setText("Tümünü Gizle" if all_on else "Tümünü Göster")


def _refresh_ui(bilateral_dist, validation_status):
    # Panel baslik
    panel = _ui.get("panel")
    if panel:
        panel.setWindowTitle(f"STN Viewer — {_g['subject']}")
    # Dogrulama durumu
    lbl = _ui.get("status_lbl")
    if lbl:
        vs  = validation_status or "—"
        col = {"PASS":"#4caf50","WARN":"#ff9800","FAIL":"#f44336"}.get(vs, "#888")
        lbl.setText(f'<span style="color:{col};font-weight:bold;">{vs}</span>')
    # L-R mesafe
    dl = _ui.get("dist_lbl")
    if dl:
        dl.setText(f"{bilateral_dist} mm" if bilateral_dist is not None else "—")
    # Özvektör degerler
    for side in ("left", "right"):
        v  = _g["axis_vecs"].get(side)
        l  = _g["axis_len"].get(side)
        vl = _ui.get(f"vec_lbl_{side}")
        ll = _ui.get(f"len_lbl_{side}")
        if vl:
            vl.setText(f"({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f})" if v else "—")
        if ll:
            ll.setText(f"{l:.1f} mm" if l else "—")
    _refresh_axis_btns()


# ─────────────────────────────────────────────────────────
# Yon seçici
# ─────────────────────────────────────────────────────────

def _set_orient(orient):
    lm = slicer.app.layoutManager()
    sn = lm.sliceWidget("Red").mrmlSliceNode()
    sn.SetOrientation(orient)
    t = _g.get("target")
    if t:
        sn.JumpSlice(*t)


# ─────────────────────────────────────────────────────────
# Kontrol paneli  (bir kez olusturulur, subject degisimde güncellenir)
# ─────────────────────────────────────────────────────────

if hasattr(slicer, "_stn_viewer_dock") and slicer._stn_viewer_dock is not None:
    try:
        slicer._stn_viewer_dock.close()
        slicer._stn_viewer_dock.deleteLater()
    except:
        pass

mw = slicer.util.mainWindow()
dock = qt.QDockWidget("STN Analiz ve Kontrol Paneli", mw)
dock.setObjectName("STNViewerDockWidget")
dock.setAllowedAreas(qt.Qt.LeftDockWidgetArea | qt.Qt.RightDockWidgetArea)
dock.setAttribute(qt.Qt.WA_DeleteOnClose, False)
slicer._stn_viewer_dock = dock

def _hsep(layout, m=4):
    f = qt.QFrame(); f.setFrameShape(qt.QFrame.HLine)
    f.setFrameShadow(qt.QFrame.Sunken)
    layout.addSpacing(m); layout.addWidget(f); layout.addSpacing(m)

panel = qt.QWidget()
dock.setWidget(panel)

vbox = qt.QVBoxLayout(panel)
vbox.setContentsMargins(12, 12, 12, 12)
vbox.setSpacing(8)

# ── Subject seçici ──────────────────────────────────────
lbl_s = qt.QLabel("Subject:")
lbl_s.setStyleSheet("font-weight:bold; font-size:13px;")
vbox.addWidget(lbl_s)

row_s = qt.QHBoxLayout()
combo = qt.QComboBox()
combo.setStyleSheet(
    "QComboBox{background:#2c2c2c;color:#eee;border:1px solid #555;"
    "border-radius:4px;font-size:12px;padding:3px 6px;}"
    "QComboBox::drop-down{border:none;width:20px;}"
    "QComboBox QAbstractItemView{background:#2c2c2c;color:#eee;"
    "selection-background-color:#4a6fa0;}"
)
for s in SUBJECTS:
    combo.addItem(s)
combo.setCurrentText(SUBJECT)
row_s.addWidget(combo, 1)

load_btn = qt.QPushButton("↺  Yükle")
load_btn.setStyleSheet(
    "QPushButton{background:#4caf50;color:white;border-radius:4px;font-size:12px;font-weight:bold;padding:6px 14px;} QPushButton:hover{background:#5cb860;}"
)
load_btn.setFixedHeight(32)
load_btn.clicked.connect(lambda: _load_scene(combo.currentText))
row_s.addWidget(load_btn)
vbox.addLayout(row_s)

row_info = qt.QHBoxLayout()
row_info.addWidget(qt.QLabel("Durum: "))
status_lbl = qt.QLabel("—")
row_info.addWidget(status_lbl)
row_info.addStretch()
row_info.addWidget(qt.QLabel("L-R: "))
dist_lbl = qt.QLabel("—")
dist_lbl.setStyleSheet("font-size:12px; color:#aaa;")
row_info.addWidget(dist_lbl)
vbox.addLayout(row_info)

_ui["panel"]      = dock
_ui["status_lbl"] = status_lbl
_ui["dist_lbl"]   = dist_lbl

# ── Yon seçici ───────────────────────────────────────────
_hsep(vbox)
vbox.addWidget(qt.QLabel("Sol panel yönü:"))

row_o = qt.QHBoxLayout(); row_o.setSpacing(5)
for lbl_txt, orient in [("Üst\n(Axial)","Axial"),
                          ("Yan\n(Sagittal)","Sagittal"),
                          ("Ön\n(Coronal)","Coronal")]:
    b = qt.QPushButton(lbl_txt)
    b.setFixedSize(88, 46)
    b.setStyleSheet("QPushButton{background:#0078d7;color:white;border-radius:4px;font-size:12px;font-weight:bold;} QPushButton:hover{background:#1084e3;}")
    b.clicked.connect(lambda _, o=orient: _set_orient(o))
    row_o.addWidget(b)
vbox.addLayout(row_o)

# ── Uzun eksen (özvektör) ─────────────────────────────────
_hsep(vbox)
ax_hdr = qt.QLabel("Uzun Eksen  (Özvektör):")
ax_hdr.setStyleSheet("font-weight:bold; font-size:12px;")
vbox.addWidget(ax_hdr)

AX_COLORS = {"left": "#008B8B", "right": "#008000"} # Aydinlik ve karanlik temalara uygun

for side, side_tr in [("left","Sol"), ("right","Sağ")]:
    col = AX_COLORS[side]

    row_ax = qt.QHBoxLayout(); row_ax.setSpacing(6)

    dot = qt.QLabel("●")
    dot.setStyleSheet(f"color:{col}; font-size:18px;")
    dot.setFixedWidth(20)
    row_ax.addWidget(dot)

    info = qt.QVBoxLayout(); info.setSpacing(1)
    t_lbl = qt.QLabel(f"{side_tr} STN:")
    t_lbl.setStyleSheet("font-size:11px; color:#999;")
    info.addWidget(t_lbl)

    len_lbl = qt.QLabel("—")
    len_lbl.setStyleSheet("font-size:11px; color:#ccc;")
    info.addWidget(len_lbl)

    vec_lbl = qt.QLabel("—")
    vec_lbl.setStyleSheet(
        f"font-size:11px; color:{col}; font-family:Courier,monospace;")
    info.addWidget(vec_lbl)

    row_ax.addLayout(info)
    row_ax.addStretch()

    ax_btn = qt.QPushButton("Göster")
    ax_btn.setFixedSize(70, 30)
    ax_btn.setStyleSheet(
        "QPushButton{background:#555;color:white;border-radius:5px;"
        "font-size:11px;font-weight:bold;padding:2px 8px;}"
        "QPushButton:hover{background:#5aaa5a;}")
    _ui[f"axis_btn_{side}"] = ax_btn
    ax_btn.clicked.connect(
        lambda _, s=side: _set_axis_visible(s, not _g["axis_visible"].get(s, False)))
    row_ax.addWidget(ax_btn)

    vbox.addLayout(row_ax)
    _ui[f"vec_lbl_{side}"] = vec_lbl
    _ui[f"len_lbl_{side}"] = len_lbl

all_ax_btn = qt.QPushButton("Tümünü Göster")
all_ax_btn.setStyleSheet(
    "QPushButton{background:#555;color:white;border-radius:5px;font-size:11px;font-weight:bold;padding:4px 12px;} QPushButton:hover{background:#3a8a5a;}")
all_ax_btn.clicked.connect(_toggle_all_axes)
_ui["axis_btn_all"] = all_ax_btn
vbox.addWidget(all_ax_btn)

# ── Footer ───────────────────────────────────────────────
_hsep(vbox, 2)
footer = qt.QLabel(
    "Sag panel: Coronal  ·  thalamus=gri  ·  cetvel=sari\n"
    "Eksen: sol=cyan  sag=yesil")
footer.setStyleSheet("font-size:10px; color:#888;")
vbox.addWidget(footer)

vbox.addStretch()

# ── Pencere konumlandirma ─────────────────────────────────
if mw:
    mw.addDockWidget(qt.Qt.LeftDockWidgetArea, dock)
dock.show()

# ─────────────────────────────────────────────────────────
# ilk yükleme
# ─────────────────────────────────────────────────────────
_setup_layout()
_load_scene(SUBJECT)
