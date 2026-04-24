# Thalamic Nuclei Morphometrics + Statistical Shape Model

Atlas-tabanlı warp ile elde edilmiş thalamic nuclei label maskelerinden morphometrik analiz,
istatistiksel şekil modelleme (SSM) ve 3D Slicer görselleştirme.

---

## Proje Yapısı

```
BrainSeg/
├── data/
│   ├── atlas/
│   │   ├── MNI152_T1_1mm.nii.gz              ← MNI152 atlas T1 (kaynak uzayı)
│   │   └── labels/
│   │       ├── left/   AD.nii.gz ...         ← Atlas label hacimleri (sol)
│   │       └── right/  AD.nii.gz ...         ← Atlas label hacimleri (sağ)
│   └── subjects/
│       └── IXI002-Guys-0828/                 (× 5 subject)
│           ├── raw/
│           │   ├── T1.nii.gz
│           │   ├── T2.nii.gz
│           │   └── MRA.nii.gz
│           └── warped/
│               ├── atlas_warped.nii.gz       ← Atlas → subject uzayı warp
│               └── labels/
│                   ├── left_AD_warped.nii.gz
│                   └── ... (78 label maskesi)
│
├── scripts/
│   ├── compute_morphometrics.py  ← Centroid + geometrik metrikler → .fcsv + CSV
│   ├── export_3d_shapes.py       ← VTK yüzey mesh üretimi (~350 mesh)
│   ├── compute_quality.py        ← Warp kalite/doğruluk analizi (VPI, simetri, CV)
│   ├── compute_ssm.py            ← PCA tabanlı şekil modeli → outputs/ssm/
│   ├── generate_report.py        ← HTML rapor oluşturucu
│   ├── slicer_load_scene.py      ← 3D Slicer: tüm nuclei + centroid pinleri
│   ├── slicer_ssm_viewer.py      ← 3D Slicer: interaktif SSM viewer
│   └── utils/
│       └── metrics.py            ← Geometrik morfometri hesaplamaları
│
└── outputs/
    ├── morphometrics/            ← 5×.fcsv pin noktaları + morphometrics_all.csv
    ├── shapes/                   ← ~350 nucleus VTK mesh + 5×brain_surface.vtk
    ├── quality/                  ← quality_report.csv + summary.txt
    ├── ssm/                      ← 35×<label>_ssm.npz + global_morpho_pca.npz
    └── report.html               ← Kapsamlı HTML raporu
```

**5 subject:** IXI002-Guys-0828 · IXI012-HH-1211 · IXI013-HH-1212 · IXI015-HH-1258 · IXI016-Guys-0697  
**39 thalamic nuclei** × sol + sağ = **78 label** per subject  
**390 morphometrik ölçüm** + **35 istatistiksel şekil modeli**

---

## Kullanım Sırası

### 1. Centroid + geometrik ölçümler
```bash
python scripts/compute_morphometrics.py
```
Çıktı: `outputs/morphometrics/morphometrics_all.csv`, `*.fcsv` (pin noktaları)

### 2. 3D yüzey mesh üretimi
```bash
python scripts/export_3d_shapes.py
```
Çıktı: `outputs/shapes/<subject>/*.vtk` (~350 mesh)

### 3. Warp kalite analizi
```bash
python -X utf8 scripts/compute_quality.py
```
Çıktı: `outputs/quality/quality_report.csv`

### 4. İstatistiksel Şekil Modeli (SSM)
```bash
python scripts/compute_ssm.py
```
Çıktı: `outputs/ssm/<label>_ssm.npz` (35 label, PCA eigenvalues/eigenvectors/scores)

### 5. HTML rapor
```bash
python -X utf8 scripts/generate_report.py
```
Çıktı: `outputs/report.html` — tarayıcıda açılır.

### 6. 3D Slicer — tüm nuclei görselleştirme
3D Slicer → `View > Python Interactor`:
```python
exec(open(r"C:/Users/ahmet/Desktop/BrainSeg-20260327T093216Z-1-001/scripts/slicer_load_scene.py").read())
```

### 7. 3D Slicer — SSM interaktif viewer
```python
exec(open(r"C:/Users/ahmet/Desktop/BrainSeg-20260327T093216Z-1-001/scripts/slicer_ssm_viewer.py").read())
```
Sol panel: label seç + P1–P4 slider → sağ 3D görünüm gerçek zamanlı deforme olur.

---

## Hesaplanan Metrikler

### Morphometri (`compute_morphometrics.py`)

| Metrik | Açıklama |
|--------|----------|
| `centroid_x/y/z_ras_mm` | Centroid — RAS mm (3D Slicer pin noktası) |
| `volume_mm3` | Hacim (mm³) |
| `surface_area_mm2` | Yüzey alanı — marching cubes (mm²) |
| `compactness` | Küresellik — 1.0 = mükemmel küre |
| `elongation` | Uzama oranı — PCA λ₁/λ₃ |
| `flatness` | Yassılık oranı — PCA λ₂/λ₃ |
| `bbox_x/y/z_mm` | Bounding box boyutları (mm) |
| `bbox_fill_ratio` | Bounding box doluluk oranı |
| `connected_components` | Bağlı bileşen sayısı |
| `skeleton_length_mm` | İskelet uzunluğu (mm) |

### Kalite / Doğruluk (`compute_quality.py`)

| Metrik | Açıklama |
|--------|----------|
| `VPI_left/right` | Hacim Koruma İndeksi = warped/atlas — 1.0 = mükemmel |
| `VPI_tier` | IYI (0.70–1.30) · ORTA (0.50–1.50) · ZAYIF |
| `LR_ratio` | Sol/sağ hacim oranı — simetri ölçüsü |
| `symmetry_tier` | SIMETRIK (0.75–1.25) · HAFIF_ASIM · ASIMETRIK |
| `cross_subject_CV` | Varyasyon katsayısı (5 subject arası tutarlılık) |

### İstatistiksel Şekil Modeli (`compute_ssm.py`)

| Veri | Açıklama |
|------|----------|
| `mean_shape` (36,) | 12 yüzey landmark × 3 koordinat — ortalama şekil |
| `eigenvectors` (36, n_modes) | PCA mod vektörleri — şekil varyasyon eksenleri |
| `eigenvalues` (n_modes,) | Her modun varyansı (mm²) |
| `scores` (5, n_modes) | Her subject'ın PCA uzayındaki koordinatları |

**Şekil vektörü:** 6 yön (Ön/Arka/Üst/Alt/Lat/Med) × 2 taraf (sol+sağ) × 3 koordinat = **36 boyut**

---

## Sonuçlar

### Morphometri Özeti (390 ölçüm)

| Metrik | Değer |
|--------|-------|
| Ortalama label hacmi | 242 mm³ |
| En büyük yapı | PuM — 1374 mm³ |
| En küçük yapı | Pv — 6.8 mm³ |
| Ortalama kompaktlık | 0.701 |
| Elongation/Flatness eksik | %12.6 (küçük yapılar, < 30 voxel) |

### Warp Kalite Sonuçları (5 subject, 195 ölçüm)

#### Hacim Koruma İndeksi (VPI)

| Tier | Aralık | Ölçüm | Oran |
|------|--------|-------|------|
| **IYI** | VPI 0.70–1.30 | 156 | **80.0%** |
| **ORTA** | VPI 0.50–1.50 | 33 | 16.9% |
| **ZAYIF** | VPI < 0.50 veya > 1.50 | 6 | 3.1% |

→ **Ölçümlerin %80'i atlas hacminin ±%30'u içinde** — warp genel olarak başarılı.

En iyi VPI: **RN** (0.984) · **STh** (0.977) · **Pv** (0.964)  
En düşük VPI: **Hb** (0.616) · **Li** (0.649) · **AM** (0.711)

#### Sol/Sağ Simetri

| Tier | Ölçüm | Oran |
|------|-------|------|
| **SİMETRİK** | 173 | **88.7%** |
| **HAFIF_ASİMETRİ** | 17 | 8.7% |
| **ASİMETRİK** | 5 | 2.6% |

En simetrik: VLpd (2.7%) · RN (2.9%) · VLpv (3.5%)  
En asimetrik: MV (50.7%) · AD (45.2%) · LP (38.2%)

#### Cross-Subject Tutarlılık (Varyasyon Katsayısı)

En tutarlı (CV < 0.10): **VPLp** (0.031) · **VLa** (0.046) · **STh** (0.047) · **VApc** (0.049) · **VPI** (0.051)

En değişken (CV > 0.25): **MV** (0.394) · **AM** (0.381) · **Hb** (0.340) · **sPf** (0.304) · **AD** (0.282)

### SSM Sonuçları (35 label, 4 mod)

| Metrik | Değer |
|--------|-------|
| SSM hesaplanan label | 35 / 39 (MV hariç — yüzey landmark'ı yok) |
| PCA modları | 4 mod (5 subject − 1) |
| PC1 ortalama varyans | **82.9%** |
| PC1 aralığı | 71.3% (thalamus_body) – 92.8% (AM) |

**En yüksek PC1 varyansı (en basit şekil varyasyonu):** AM (92.8%) · VAmc (90.5%) · AV (91.8%)  
**En düşük PC1 varyansı (çok boyutlu şekil varyasyonu):** thalamus_body (71.3%) · LGNmc (76.6%) · SG (76.9%)

### Dikkat Gerektiren Yapılar

| Label | Sorun | Neden |
|-------|-------|-------|
| MV | SSM dışı, VPI: N/A | Hiç yüzey landmark'ı yok |
| AD, Pv | Yüksek CV (>0.28), Elongation eksik | < 10 mm³ — çok küçük |
| Hb, AM, sPf | Yüksek CV (>0.30), düşük VPI | < 40 mm³ — küçük yapı |
| Li | En düşük VPI (0.649) | Habenulaya yakın, zor warp bölgesi |

> **Not:** VPI sorunu olan yapıların büyük çoğunluğu < 50 mm³'tür. Küçük yapılarda birkaç voxel farkı bile yüksek VPI sapmasına yol açar — istatistiksel sınırlılık, mutlaka hatalı segmentasyon değil.

---

## 3D Slicer Görselleştirme

### Klasik görünüm (`slicer_load_scene.py`)
- `atlas_warped.nii.gz` — arka plan MRI (axial/sagittal/coronal)
- `brain_surface.vtk` — yarı saydam beyin kabuğu (%25 opacity)
- 39 nucleus × sol+sağ renkli VTK mesh (%75 opacity)
- Her label için renkli centroid + 6 yön yüzey landmark noktaları

### SSM Viewer (`slicer_ssm_viewer.py`)
- **Sol 3D görünüm (gri):** referans mesh — ortalamaya en yakın subject
- **Sağ 3D görünüm (mavi):** model çıktısı — P1–P4 slider'larıyla TPS deformasyonu
- **PCA Scatter Plot:** 5 subject'ın PC1/PC2 uzayındaki konumu
- **SSM Skor Tablosu:** her subject için σ-normalize PCA skorları

Slider aralığı: ±3σ — şeklin biyolojik varyasyon sınırları içinde kalır.

---

## Bağımlılıklar

```
nibabel
numpy
scipy
scikit-image
pandas
scikit-learn   # compute_ssm.py global PCA için
```

3D Slicer 5.11.0+ (Python konsolu dahili, VTK 9.x).

---

## Lisans

MIT License — bkz. [LICENSE](LICENSE)

---

*Araştırma amaçlıdır. Klinik karar destek aracı değildir.*
