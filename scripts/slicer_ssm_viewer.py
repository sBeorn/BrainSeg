"""
SSM Viewer — Thalamic Nuclei Statistical Shape Model
======================================================
3D Slicer Python konsolunda çalıştırılır.

Önce compute_ssm.py çalıştırılmış olmalı:
    python scripts/compute_ssm.py

Kullanım (Slicer Python konsolu):
    exec(open(r"C:/Users/ahmet/Desktop/BrainSeg-20260327T093216Z-1-001/scripts/slicer_ssm_viewer.py").read())

Arayüz:
    Sol panel               — label seçimi + P1–P4 mod slider'ları
    Sol 3D görünüm (GRİ)   — referans şekil (ortalamaya en yakın subject)
    Sağ 3D görünüm (MAVİ)  — PCA model çıktısı (slider'larla şekil değişir)
    Alt sol                 — PCA Scatter Plot
    Alt sağ                 — SSM skor tablosu
"""

import os
import numpy as np
import slicer
import vtk

try:
    import qt
except ImportError:
    raise ImportError("Bu script 3D Slicer Python konsolunda çalışmalıdır.")

# ─────────────────────────────────────────────────────────────────────────────
# Yollar & Sabitler
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR   = r"C:/Users/ahmet/Desktop/BrainSeg-20260327T093216Z-1-001"
SSM_DIR    = os.path.join(BASE_DIR, "outputs", "ssm")
SHAPES_DIR = os.path.join(BASE_DIR, "outputs", "shapes")
MORPH_DIR  = os.path.join(BASE_DIR, "outputs", "morphometrics")

DIRECTIONS  = ["On", "Arka", "Ust", "Alt", "Lat", "Med"]
SIDES       = ["left", "right"]
N_MODES     = 4
SIGMA_RANGE = 3.0
SLIDER_RES  = 200

CUSTOM_LAYOUT_ID = 501


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı Fonksiyonlar
# ─────────────────────────────────────────────────────────────────────────────

def get_available_labels():
    if not os.path.isdir(SSM_DIR):
        return []
    return sorted(
        f.replace("_ssm.npz", "")
        for f in os.listdir(SSM_DIR)
        if f.endswith("_ssm.npz")
    )


def load_ssm(label):
    path = os.path.join(SSM_DIR, f"{label}_ssm.npz")
    if not os.path.exists(path):
        return None
    return dict(np.load(path, allow_pickle=True))


def load_vtk_polydata(vtk_path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_path)
    reader.Update()
    pd = vtk.vtkPolyData()
    pd.DeepCopy(reader.GetOutput())
    return pd


def load_surface_fcsv(path):
    """Surface FCSV → {label: {side: {direction: [x,y,z]}}}"""
    data = {}
    if not os.path.exists(path):
        return data
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split(",")
            if len(parts) < 12:
                continue
            x, y, z  = float(parts[1]), float(parts[2]), float(parts[3])
            lm_label = parts[11]
            tokens   = lm_label.split("_")
            side_idx = None
            for i, t in enumerate(tokens):
                if t in ("left", "right"):
                    side_idx = i
                    break
            if side_idx is None:
                continue
            label     = "_".join(tokens[:side_idx])
            side      = tokens[side_idx]
            direction = "_".join(tokens[side_idx + 1:]) if side_idx + 1 < len(tokens) else ""
            data.setdefault(label, {}).setdefault(side, {})[direction] = [x, y, z]
    return data


def extract_label_landmarks(fcsv_data, label):
    """Bir label için (12, 3) landmark dizisi. Eksikse None döner."""
    pts = []
    for side in SIDES:
        for d in DIRECTIONS:
            try:
                pt = fcsv_data[label][side][d]
            except KeyError:
                return None
            pts.append(pt)
    return np.array(pts, dtype=np.float64)


def find_closest_to_mean(ssm):
    scores = ssm["scores"]
    dists  = np.sum(scores ** 2, axis=1)
    idx    = int(np.argmin(dists))
    return str(ssm["subjects"][idx]), idx


def compute_deformed_landmarks(ssm, b_sigma):
    """mean + Σ b_sigma[i]*sqrt(λ_i)*mode_i  →  (12, 3)"""
    mean    = ssm["mean_shape"]
    eigvecs = ssm["eigenvectors"]
    eigvals = ssm["eigenvalues"]
    n_modes = int(eigvecs.shape[1])
    b = np.zeros(n_modes)
    for i in range(n_modes):
        sigma = float(np.sqrt(max(float(eigvals[i]), 1e-12)))
        b[i]  = float(b_sigma[i]) * sigma
    return (mean + eigvecs @ b).reshape(-1, 3)


def deform_mesh_tps(base_pd, source_lm, target_lm):
    """TPS deformasyonu: source_lm → target_lm, base_pd üzerine uygula."""
    src_pts = vtk.vtkPoints()
    tgt_pts = vtk.vtkPoints()
    for pt in source_lm:
        src_pts.InsertNextPoint(float(pt[0]), float(pt[1]), float(pt[2]))
    for pt in target_lm:
        tgt_pts.InsertNextPoint(float(pt[0]), float(pt[1]), float(pt[2]))

    tps = vtk.vtkThinPlateSplineTransform()
    tps.SetSourceLandmarks(src_pts)
    tps.SetTargetLandmarks(tgt_pts)
    tps.SetBasisToR2LogR()
    tps.Update()

    filt = vtk.vtkTransformPolyDataFilter()
    filt.SetInputData(base_pd)
    filt.SetTransform(tps)
    filt.Update()

    result = vtk.vtkPolyData()
    result.DeepCopy(filt.GetOutput())
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────

def setup_custom_layout():
    custom_xml = """
<layout type="vertical" split="true">
  <item splitSize="560">
    <layout type="horizontal">
      <item>
        <view class="vtkMRMLViewNode" singletontag="1">
          <property name="viewlabel" action="default">Ortalama</property>
          <property name="viewcolor" action="default">#404040</property>
        </view>
      </item>
      <item>
        <view class="vtkMRMLViewNode" singletontag="2">
          <property name="viewlabel" action="default">Model</property>
          <property name="viewcolor" action="default">#003366</property>
        </view>
      </item>
    </layout>
  </item>
  <item splitSize="340">
    <layout type="horizontal">
      <item>
        <view class="vtkMRMLPlotViewNode" singletontag="PlotView1">
          <property name="viewlabel" action="default">PCA Scatter Plot</property>
        </view>
      </item>
      <item>
        <view class="vtkMRMLTableViewNode" singletontag="TableView1">
          <property name="viewlabel" action="default">SSM Skorlar</property>
        </view>
      </item>
    </layout>
  </item>
</layout>
"""
    lm         = slicer.app.layoutManager()
    layoutNode = lm.layoutLogic().GetLayoutNode()
    layoutNode.AddLayoutDescription(CUSTOM_LAYOUT_ID, custom_xml)
    lm.setLayout(CUSTOM_LAYOUT_ID)
    # Layout'ın view node'larını oluşturması için event loop'u ilerlet
    qt.QApplication.processEvents()


def get_3d_view_node_ids():
    """Sahnedeki mevcut 3D view node ID'lerini sıralı döndürür."""
    ids = []
    col = slicer.mrmlScene.GetNodesByClass("vtkMRMLViewNode")
    col.InitTraversal()
    node = col.GetNextItemAsObject()
    while node:
        ids.append(node.GetID())
        node = col.GetNextItemAsObject()
    return sorted(ids)


# ─────────────────────────────────────────────────────────────────────────────
# SSM Viewer
# ─────────────────────────────────────────────────────────────────────────────

class SSMViewer:
    def __init__(self):
        self.labels = get_available_labels()
        if not self.labels:
            raise RuntimeError(
                f"SSM verisi bulunamadı: {SSM_DIR}\n"
                "Önce: python scripts/compute_ssm.py"
            )
        print(f"  {len(self.labels)} label SSM verisi bulundu.")

        self.current_label   = self.labels[0]
        self.current_ssm     = None
        self.b_sigma         = [0.0] * N_MODES
        self.base_subject    = None
        self.base_polydata   = {}
        self.base_landmarks  = None

        # Slicer node referansları
        self.gray_nodes             = {}
        self.blue_nodes             = {}
        self.mean_landmark_node     = None
        self.deformed_landmark_node = None
        self.plot_chart_node        = None
        self.scatter_table_node     = None
        self.ssm_table_node         = None

        # Qt widget referansları
        self.dock          = None
        self.label_combo   = None
        self.subject_combo = None
        self.mode_widgets  = []
        self.info_lbl      = None

        setup_custom_layout()
        self._build_ui()
        self._load_label(self.current_label)

    # ── Node Temizleme ──────────────────────────────────────────────────────
    # mrmlScene.Clear() KULLANMA — layout'ın view node'larını siler!
    # Sadece kendi oluşturduğumuz node'ları sil.

    def _cleanup_nodes(self):
        to_remove = []

        for side_dict in [self.gray_nodes, self.blue_nodes]:
            for n in side_dict.values():
                if n and n.GetScene():
                    to_remove.append(n)

        for n in [self.mean_landmark_node, self.deformed_landmark_node]:
            if n and n.GetScene():
                to_remove.append(n)

        if self.plot_chart_node and self.plot_chart_node.GetScene():
            for i in range(self.plot_chart_node.GetNumberOfPlotSeriesNodes()):
                sn = self.plot_chart_node.GetNthPlotSeriesNode(i)
                if sn:
                    to_remove.append(sn)
            to_remove.append(self.plot_chart_node)

        for n in [self.scatter_table_node, self.ssm_table_node]:
            if n and n.GetScene():
                to_remove.append(n)

        for n in to_remove:
            try:
                slicer.mrmlScene.RemoveNode(n)
            except Exception:
                pass

        self.gray_nodes             = {}
        self.blue_nodes             = {}
        self.mean_landmark_node     = None
        self.deformed_landmark_node = None
        self.plot_chart_node        = None
        self.scatter_table_node     = None
        self.ssm_table_node         = None

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        mw = slicer.util.mainWindow()

        self.dock = qt.QDockWidget("SSM: Şekil Analizi", mw)
        self.dock.setAllowedAreas(
            qt.Qt.LeftDockWidgetArea | qt.Qt.RightDockWidgetArea
        )
        self.dock.setMinimumWidth(290)

        root = qt.QWidget()
        vbox = qt.QVBoxLayout(root)
        vbox.setSpacing(6)
        vbox.setContentsMargins(8, 8, 8, 4)

        title = qt.QLabel("<b>İstatistiksel Şekil Modeli (SSM)</b>")
        title.setAlignment(qt.Qt.AlignCenter)
        title.setStyleSheet("font-size:13px; padding:4px;")
        vbox.addWidget(title)

        sep = qt.QFrame()
        sep.setFrameShape(qt.QFrame.HLine)
        vbox.addWidget(sep)

        # Label seçimi
        g1 = qt.QGroupBox("Yapı (Label)")
        l1 = qt.QVBoxLayout(g1)
        self.label_combo = qt.QComboBox()
        for lbl in self.labels:
            self.label_combo.addItem(lbl)
        l1.addWidget(self.label_combo)
        self.label_combo.connect("currentIndexChanged(int)", self._on_label_changed)
        vbox.addWidget(g1)

        # Subject seçimi
        g2 = qt.QGroupBox("Sol Görünüm — Referans Mesh")
        l2 = qt.QVBoxLayout(g2)
        self.subject_combo = qt.QComboBox()
        self.subject_combo.addItem("⟨ Ortalamaya en yakın ⟩")
        l2.addWidget(self.subject_combo)
        self.subject_combo.connect("currentIndexChanged(int)", self._on_subject_changed)
        vbox.addWidget(g2)

        # Mod slider'ları
        g3 = qt.QGroupBox("Model Parametreleri  b  (σ birimi)")
        l3 = qt.QVBoxLayout(g3)
        self.mode_widgets = []

        for i in range(N_MODES):
            row_w = qt.QWidget()
            row   = qt.QHBoxLayout(row_w)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)

            p_lbl = qt.QLabel(f"P{i+1}")
            p_lbl.setFixedWidth(22)
            p_lbl.setStyleSheet("font-weight:bold;")

            slider = qt.QSlider(qt.Qt.Horizontal)
            slider.setMinimum(-SLIDER_RES)
            slider.setMaximum( SLIDER_RES)
            slider.setValue(0)
            slider.setFixedHeight(18)

            val_lbl = qt.QLabel("0.00σ")
            val_lbl.setFixedWidth(48)
            val_lbl.setAlignment(qt.Qt.AlignRight | qt.Qt.AlignVCenter)

            sig_lbl = qt.QLabel("—")
            sig_lbl.setFixedWidth(80)
            sig_lbl.setStyleSheet("color:#888; font-size:10px;")

            row.addWidget(p_lbl)
            row.addWidget(slider)
            row.addWidget(val_lbl)
            row.addWidget(sig_lbl)
            l3.addWidget(row_w)

            def _make_cb(idx):
                def _cb(v):
                    self._on_slider(idx, v)
                return _cb
            slider.connect("valueChanged(int)", _make_cb(i))

            self.mode_widgets.append(
                {"slider": slider, "val_lbl": val_lbl, "sig_lbl": sig_lbl}
            )

        vbox.addWidget(g3)

        self.info_lbl = qt.QLabel("Yükleniyor...")
        self.info_lbl.setWordWrap(True)
        self.info_lbl.setStyleSheet("color:#555; font-size:10px; padding:2px;")
        vbox.addWidget(self.info_lbl)

        btn_row = qt.QWidget()
        btn_h   = qt.QHBoxLayout(btn_row)
        btn_h.setContentsMargins(0, 0, 0, 0)

        btn_reset = qt.QPushButton("↺  Ortalamaya Dön")
        btn_reset.connect("clicked()", self._on_reset)

        btn_snap = qt.QPushButton("▶  Subject Göster")
        btn_snap.setToolTip("Seçili subject'ın PCA konumunu mavi görünüme yükle")
        btn_snap.connect("clicked()", self._on_show_subject)

        btn_h.addWidget(btn_reset)
        btn_h.addWidget(btn_snap)
        vbox.addWidget(btn_row)
        vbox.addStretch()

        self.dock.setWidget(root)
        mw.addDockWidget(qt.Qt.LeftDockWidgetArea, self.dock)
        self.dock.show()

    # ── Label Yükleme ───────────────────────────────────────────────────────

    def _load_label(self, label):
        print(f"\n  Label yükleniyor: {label}")
        self.current_label = label
        ssm = load_ssm(label)
        if ssm is None:
            print(f"  HATA: SSM verisi bulunamadı — {label}")
            return
        self.current_ssm = ssm
        self.b_sigma = [0.0] * N_MODES

        # Subject combo
        self.subject_combo.blockSignals(True)
        while self.subject_combo.count > 1:
            self.subject_combo.removeItem(1)
        for s in ssm["subjects"]:
            self.subject_combo.addItem(str(s))
        self.subject_combo.setCurrentIndex(0)
        self.subject_combo.blockSignals(False)

        # Slider güncelle
        eigvals = ssm["eigenvalues"]
        tot_var = float(np.sum(eigvals))
        for i, mw in enumerate(self.mode_widgets):
            mw["slider"].blockSignals(True)
            mw["slider"].setValue(0)
            mw["val_lbl"].setText("0.00σ")
            mw["slider"].blockSignals(False)
            if i < int(ssm["n_modes"]):
                sigma = float(np.sqrt(max(float(eigvals[i]), 1e-12)))
                pct   = 100.0 * float(eigvals[i]) / tot_var if tot_var > 0 else 0.0
                mw["sig_lbl"].setText(f"σ={sigma:.2f} ({pct:.1f}%)")
                mw["slider"].setEnabled(True)
            else:
                mw["sig_lbl"].setText("—")
                mw["slider"].setEnabled(False)

        pct_strs = [
            f"P{i+1}:{100*float(eigvals[i])/tot_var:.1f}%"
            for i in range(min(N_MODES, int(ssm["n_modes"])))
            if tot_var > 0
        ]
        self.info_lbl.setText("Açıklanan varyans: " + "  ".join(pct_strs))

        # Base subject & mesh
        self.base_subject, _ = find_closest_to_mean(ssm)
        print(f"  Referans subject: {self.base_subject}")
        self._reload_base_meshes(self.base_subject)

        # 3D sahne
        self._setup_scene()

        # Plot & tablo
        self._update_scatter_plot()
        self._update_ssm_table()
        print(f"  '{label}' yüklendi.")

    def _reload_base_meshes(self, subject):
        self.base_polydata  = {}
        self.base_landmarks = None

        for side in SIDES:
            vtk_path = os.path.join(
                SHAPES_DIR, subject, f"{self.current_label}_{side}.vtk"
            )
            if os.path.exists(vtk_path):
                self.base_polydata[side] = load_vtk_polydata(vtk_path)
                print(f"  Mesh yüklendi: {os.path.basename(vtk_path)}")
            else:
                print(f"  UYARI: mesh bulunamadı: {vtk_path}")

        fcsv_path = os.path.join(MORPH_DIR, f"{subject}_surface.fcsv")
        fcsv_data = load_surface_fcsv(fcsv_path)
        lm = extract_label_landmarks(fcsv_data, self.current_label)
        if lm is not None:
            self.base_landmarks = lm
        else:
            self.base_landmarks = self.current_ssm["mean_shape"].reshape(-1, 3)
            print("  UYARI: FCSV landmark bulunamadı, SSM ortalaması kullanıldı")

    # ── 3D Sahne ────────────────────────────────────────────────────────────

    def _setup_scene(self):
        # Sadece kendi node'larımızı temizle (mrmlScene.Clear() KULLANMA!)
        self._cleanup_nodes()

        # Mevcut view node ID'lerini al — layout bunları oluşturmuş olmalı
        view_ids = get_3d_view_node_ids()
        print(f"  Mevcut 3D view node'ları: {view_ids}")

        view1_id = view_ids[0] if len(view_ids) >= 1 else None
        view2_id = view_ids[1] if len(view_ids) >= 2 else None

        mean_lm = self.current_ssm["mean_shape"].reshape(-1, 3)

        for side in SIDES:
            if side not in self.base_polydata:
                continue

            pd_src = self.base_polydata[side]

            # Gri model → View1
            g_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
            g_node.SetName(f"mean_{self.current_label}_{side}")
            g_pd = vtk.vtkPolyData()
            g_pd.DeepCopy(pd_src)
            g_node.SetAndObservePolyData(g_pd)
            g_node.CreateDefaultDisplayNodes()
            dn = g_node.GetDisplayNode()
            dn.SetColor(0.72, 0.72, 0.72)
            dn.SetOpacity(0.90)
            dn.SetVisibility3D(True)
            dn.SetVisibility2D(False)
            dn.SetBackfaceCulling(False)
            dn.SetAmbient(0.25)
            dn.SetDiffuse(0.75)
            if view1_id:
                dn.SetViewNodeIDs([view1_id])
            self.gray_nodes[side] = g_node
            print(f"  Gri model oluşturuldu: mean_{self.current_label}_{side} → {view1_id}")

            # Mavi model → View2
            b_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
            b_node.SetName(f"model_{self.current_label}_{side}")
            b_pd = vtk.vtkPolyData()
            b_pd.DeepCopy(pd_src)
            b_node.SetAndObservePolyData(b_pd)
            b_node.CreateDefaultDisplayNodes()
            dn2 = b_node.GetDisplayNode()
            dn2.SetColor(0.18, 0.42, 0.92)
            dn2.SetOpacity(0.90)
            dn2.SetVisibility3D(True)
            dn2.SetVisibility2D(False)
            dn2.SetBackfaceCulling(False)
            dn2.SetAmbient(0.25)
            dn2.SetDiffuse(0.75)
            if view2_id:
                dn2.SetViewNodeIDs([view2_id])
            self.blue_nodes[side] = b_node
            print(f"  Mavi model oluşturuldu: model_{self.current_label}_{side} → {view2_id}")

        # Landmark marker'lar
        self._setup_landmarks(mean_lm, view1_id, view2_id)

        # Kameraları sıfırla
        lm_mgr = slicer.app.layoutManager()
        for vi in range(2):
            try:
                w = lm_mgr.threeDWidget(vi)
                if w:
                    w.threeDView().resetFocalPoint()
                    w.threeDView().rotateToViewAxis(3)
            except Exception as e:
                print(f"  Kamera {vi} sıfırlanamadı: {e}")

        # Mavi mesh'i b=0 ile güncelle
        self._update_blue_mesh()

    def _setup_landmarks(self, mean_lm, view1_id, view2_id):
        pt_labels = [f"{s[0].upper()}_{d}" for s in SIDES for d in DIRECTIONS]

        # View1 — sabit ortalama landmark
        n1 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        n1.SetName(f"mean_lm_{self.current_label}")
        for pt, lbl in zip(mean_lm, pt_labels):
            idx = n1.AddControlPoint([float(pt[0]), float(pt[1]), float(pt[2])])
            n1.SetNthControlPointLabel(idx, lbl)
        dn1 = n1.GetDisplayNode()
        dn1.SetColor(1.0, 0.85, 0.0)
        dn1.SetSelectedColor(1.0, 1.0, 0.0)
        dn1.SetGlyphScale(3.5)
        dn1.SetTextScale(0.0)
        dn1.SetGlyphType(13)
        if view1_id:
            dn1.SetViewNodeIDs([view1_id])
        self.mean_landmark_node = n1

        # View2 — deforme landmark
        n2 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        n2.SetName(f"model_lm_{self.current_label}")
        for pt, lbl in zip(mean_lm, pt_labels):
            idx = n2.AddControlPoint([float(pt[0]), float(pt[1]), float(pt[2])])
            n2.SetNthControlPointLabel(idx, lbl)
        dn2 = n2.GetDisplayNode()
        dn2.SetColor(1.0, 0.85, 0.0)
        dn2.SetSelectedColor(1.0, 1.0, 0.0)
        dn2.SetGlyphScale(3.5)
        dn2.SetTextScale(0.0)
        dn2.SetGlyphType(13)
        if view2_id:
            dn2.SetViewNodeIDs([view2_id])
        self.deformed_landmark_node = n2

    # ── Mavi Mesh Güncelleme ────────────────────────────────────────────────

    def _update_blue_mesh(self):
        ssm = self.current_ssm
        if ssm is None or self.base_landmarks is None:
            return

        target_lm = compute_deformed_landmarks(ssm, self.b_sigma)  # (12,3)
        source_lm = self.base_landmarks                             # (12,3)

        # Landmark marker'larını güncelle
        if self.deformed_landmark_node:
            n_pts = self.deformed_landmark_node.GetNumberOfControlPoints()
            for j in range(min(n_pts, len(target_lm))):
                pt = target_lm[j]
                self.deformed_landmark_node.SetNthControlPointPosition(
                    j, float(pt[0]), float(pt[1]), float(pt[2])
                )

        # TPS deformasyonu
        for side in SIDES:
            if side not in self.base_polydata or side not in self.blue_nodes:
                continue
            deformed = deform_mesh_tps(
                self.base_polydata[side], source_lm, target_lm
            )
            self.blue_nodes[side].SetAndObservePolyData(deformed)

    # ── Scatter Plot ────────────────────────────────────────────────────────

    def _update_scatter_plot(self):
        ssm      = self.current_ssm
        scores   = ssm["scores"]
        subjects = [str(s) for s in ssm["subjects"]]
        eigvals  = ssm["eigenvalues"]

        # Eski node'ları temizle
        if self.plot_chart_node and self.plot_chart_node.GetScene():
            for i in range(self.plot_chart_node.GetNumberOfPlotSeriesNodes()):
                sn = self.plot_chart_node.GetNthPlotSeriesNode(i)
                if sn:
                    slicer.mrmlScene.RemoveNode(sn)
            slicer.mrmlScene.RemoveNode(self.plot_chart_node)
        if self.scatter_table_node and self.scatter_table_node.GetScene():
            slicer.mrmlScene.RemoveNode(self.scatter_table_node)

        # Tablo
        tbl_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "PCA_Scatter")
        tbl = tbl_node.GetTable()
        c_pc1  = vtk.vtkFloatArray();  c_pc1.SetName("PC1")
        c_pc2  = vtk.vtkFloatArray();  c_pc2.SetName("PC2")
        c_subj = vtk.vtkStringArray(); c_subj.SetName("Konu")
        for i, subj in enumerate(subjects):
            c_pc1.InsertNextValue(float(scores[i, 0]))
            c_pc2.InsertNextValue(float(scores[i, 1]) if scores.shape[1] > 1 else 0.0)
            c_subj.InsertNextValue(subj.replace("IXI", "S").split("-")[0])
        tbl.AddColumn(c_pc1)
        tbl.AddColumn(c_pc2)
        tbl.AddColumn(c_subj)
        self.scatter_table_node = tbl_node

        # Seri
        series = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLPlotSeriesNode", f"scatter_{self.current_label}"
        )
        series.SetAndObserveTableNodeID(tbl_node.GetID())
        series.SetXColumnName("PC1")
        series.SetYColumnName("PC2")
        series.SetLabelColumnName("Konu")
        series.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
        series.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleCircle)
        series.SetMarkerSize(18)
        series.SetColor(0.18, 0.48, 0.92)
        series.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleNone)

        tot  = float(np.sum(eigvals))
        pct1 = 100.0 * float(eigvals[0]) / tot if tot > 0 else 0.0
        pct2 = (100.0 * float(eigvals[1]) / tot if len(eigvals) > 1 and tot > 0 else 0.0)

        chart = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode", "PCA Scatter Plot")
        chart.AddAndObservePlotSeriesNodeID(series.GetID())
        chart.SetTitle(f"PCA  —  {self.current_label}")
        chart.SetXAxisTitle(f"PC1 ({pct1:.1f}%)")
        chart.SetYAxisTitle(f"PC2 ({pct2:.1f}%)")
        chart.SetLegendVisibility(True)
        self.plot_chart_node = chart

        try:
            lm = slicer.app.layoutManager()
            pw = lm.plotWidget(0)
            if pw:
                pw.mrmlPlotViewNode().SetPlotChartNodeID(chart.GetID())
        except Exception as e:
            print(f"  Scatter plot bağlanamadı: {e}")

    # ── SSM Tablo ───────────────────────────────────────────────────────────

    def _update_ssm_table(self):
        ssm      = self.current_ssm
        scores   = ssm["scores"]
        subjects = [str(s) for s in ssm["subjects"]]
        eigvals  = ssm["eigenvalues"]
        n_modes  = int(ssm["n_modes"])

        if self.ssm_table_node and self.ssm_table_node.GetScene():
            slicer.mrmlScene.RemoveNode(self.ssm_table_node)

        tbl_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", "SSM_Skorlar")
        tbl = tbl_node.GetTable()

        c_s = vtk.vtkStringArray(); c_s.SetName("Subject")
        for s in subjects:
            c_s.InsertNextValue(s)
        tbl.AddColumn(c_s)

        for i in range(n_modes):
            sigma = float(np.sqrt(max(float(eigvals[i]), 1e-12)))
            col   = vtk.vtkFloatArray()
            col.SetName(f"PC{i+1}/σ")
            for j in range(len(subjects)):
                col.InsertNextValue(round(float(scores[j, i]) / sigma, 3))
            tbl.AddColumn(col)

        c_d = vtk.vtkFloatArray(); c_d.SetName("‖b‖ (σ)")
        for j in range(len(subjects)):
            d = sum(
                (float(scores[j, i]) / float(np.sqrt(max(float(eigvals[i]), 1e-12)))) ** 2
                for i in range(n_modes)
            )
            c_d.InsertNextValue(round(float(np.sqrt(d)), 3))
        tbl.AddColumn(c_d)
        self.ssm_table_node = tbl_node

        try:
            lm = slicer.app.layoutManager()
            tw = lm.tableWidget(0)
            if tw:
                tw.mrmlTableViewNode().SetTableNodeID(tbl_node.GetID())
        except Exception as e:
            print(f"  Tablo bağlanamadı: {e}")

    # ── Callback'ler ────────────────────────────────────────────────────────

    def _on_label_changed(self, idx):
        self._load_label(self.labels[idx])

    def _on_subject_changed(self, idx):
        ssm = self.current_ssm
        if ssm is None:
            return
        if idx == 0:
            subject, _ = find_closest_to_mean(ssm)
        else:
            subject = str(ssm["subjects"][idx - 1])
        self.base_subject = subject
        self._reload_base_meshes(subject)
        for side, g_node in self.gray_nodes.items():
            if side in self.base_polydata:
                pd_copy = vtk.vtkPolyData()
                pd_copy.DeepCopy(self.base_polydata[side])
                g_node.SetAndObservePolyData(pd_copy)
        self._update_blue_mesh()

    def _on_slider(self, mode_idx, slider_val):
        b_val = slider_val / SLIDER_RES * SIGMA_RANGE
        self.b_sigma[mode_idx] = b_val
        self.mode_widgets[mode_idx]["val_lbl"].setText(f"{b_val:+.2f}σ")
        self._update_blue_mesh()

    def _on_reset(self):
        self.b_sigma = [0.0] * N_MODES
        for mw in self.mode_widgets:
            mw["slider"].blockSignals(True)
            mw["slider"].setValue(0)
            mw["val_lbl"].setText("0.00σ")
            mw["slider"].blockSignals(False)
        self._update_blue_mesh()

    def _on_show_subject(self):
        ssm = self.current_ssm
        if ssm is None:
            return
        idx = self.subject_combo.currentIndex()
        if idx == 0:
            self._on_reset()
            return
        subj_idx = idx - 1
        scores  = ssm["scores"]
        eigvals = ssm["eigenvalues"]
        n_modes = int(ssm["n_modes"])
        for i in range(N_MODES):
            if i < n_modes:
                sigma  = float(np.sqrt(max(float(eigvals[i]), 1e-12)))
                b_val  = float(scores[subj_idx, i]) / sigma
                b_val  = max(-SIGMA_RANGE, min(SIGMA_RANGE, b_val))
                sv     = int(round(b_val / SIGMA_RANGE * SLIDER_RES))
            else:
                b_val, sv = 0.0, 0
            self.b_sigma[i] = b_val
            mw = self.mode_widgets[i]
            mw["slider"].blockSignals(True)
            mw["slider"].setValue(sv)
            mw["val_lbl"].setText(f"{b_val:+.2f}σ")
            mw["slider"].blockSignals(False)
        self._update_blue_mesh()


# ─────────────────────────────────────────────────────────────────────────────
# Başlatma
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*58)
print("  SSM Viewer başlatılıyor...")
print("="*58)

# Önceki instance'ı kapat
try:
    if "_ssm_viewer" in dir() and _ssm_viewer is not None:
        try:
            _ssm_viewer._cleanup_nodes()
            _ssm_viewer.dock.close()
            _ssm_viewer.dock.deleteLater()
        except Exception:
            pass
except Exception:
    pass

_ssm_viewer = SSMViewer()

print(f"\n  HAZIR  —  {len(_ssm_viewer.labels)} label")
print(f"  Sol görünüm (GRİ)  = referans mesh (b=0)")
print(f"  Sağ görünüm (MAVİ) = model çıktısı (slider ile değişir)")
print("="*58 + "\n")
