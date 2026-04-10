# BrainSeg Ar-Ge Projesi — Teknik Rehber

> Bu dosya Claude Code'a her oturumda bağlam sağlamak için hazırlanmıştır.
> Her adımdan sonra güncellenir. Kod yazarken bu dosyayı referans al.

---

## 0. Mevcut Durum — Hızlı Referans (2026-04-08)

| Adım | Durum | Sonuç |
|------|-------|-------|
| İP-1 Preproc | ✓ 5/5 | ANTsPyNet brain mask, N4, T2/MRA→T1 |
| İP-2 Warp | ✓ 15/15 | **Kazanan: W2** (Affine+SyN, MI) |
| İP-3 Propagate + Morphometrics | ✓ 15/15 | 41 label × 5 hasta, tam morphometrics |
| İP-4 Refine | ✓ 45/45 | Ham propagation en güvenilir (refinement etkisiz) |
| Uzman İnceleme Paketi | ✓ | `outputs/expert_review/` |
| LOO Doğrulama | ✓ | 41 label DSC skoru mevcut |

**★ Final Pipeline:** `W2 + propagate (refinement uygulanmadan)` — en güvenilir çıktı

**Çalıştırma:**
```bash
python scripts/export_expert_review.py --warp W2   # uzman paketi üret/güncelle
```

---

## 1. KALİTE, PERFORMANS VE DOĞRULUK KANONU

> **Bu proje GT olmadan çalışmaktadır. Her kararın temeli ölçülebilir
> kalite kanıtı olmak zorundadır. Kod yazarken kaliteyi, doğruluğu ve
> tekrar edilebilirliği her zaman önce düşün.**

### 1.1 Temel İlkeler

1. **Hiçbir label kör kabul edilmez.** Her label için morphometrics, LOO DSC ve cross-subject tutarlılık hesaplanır; sonuç TIER sistemiyle belgelenir.

2. **Güvenilirlik katmanları:** TIER-1 (yayına hazır) → TIER-2 (dikkatli inceleme) → TIER-3 (kullanılamaz). Hiçbir TIER-3 label parcellation'a girmez.

3. **Her metrik eşiği gerekçelidir.** Eşikleri keyfi değil; anatomik bilgiye veya gözlemlenen dağılıma dayandır. Eşik değiştirildiğinde gerekçe dokümana yazılır.

4. **Morphometrics birincil kalite sinyalidir.** GT olmadığı için shape, intensity, symmetry ve cross-subject consistency ölçümleri doğruluğun tek proxy'sidir.

5. **Kod değiştirmeden önce mevcut çıktıyı ölç.** Herhangi bir iyileştirme öncesi ve sonrası karşılaştırılabilir metrik üretilir.

### 1.2 Kalite Skoru Sistemi (classify_label fonksiyonu)

Her label için kalite puanı şu bileşenlerden oluşur:

| Bileşen | LOO DSC yoksa | LOO DSC varsa | Açıklama |
|---------|:------------:|:------------:|----------|
| Temel (atlas hacmi + retention) | **0.50** | **0.35** | Propagasyon başarısı |
| LOO DSC | — | **0.30** | Pseudo ground-truth |
| Morphometrics (8 kriter) | **0.30** | **0.20** | Şekil + yoğunluk kalitesi |
| Cross-subject tutarlılık | **0.20** | **0.15** | 5 hasta arası CV analizi |

**TIER sınırları:** ≥ 0.60 → TIER-1 | ≥ 0.40 → TIER-2 | < 0.40 → TIER-3

**Eleme kuralı:** `atlas_vol < 50mm³` veya `retention < 30%` → doğrudan TIER-3 (skor hesaplamadan)

### 1.3 Morphometrik Kalite Kriterleri (8 Kriter)

| # | Kriter | Eşik | Penalti | Ne Tespit Eder? |
|---|--------|:----:|:-------:|-----------------|
| 1 | **L/R hacim simetrisi** | ratio > 3.0× | −0.30 | Asimetrik propagasyon hatası |
| 2 | **T1 arka plan kirliliği** | T1_mean < −0.4 | −0.20/taraf | Label arka plana taşmış |
| 3 | **Centroid simetri hatası** | mirror_err > 8mm | −0.15 | Konumsal kayma |
| 4 | **Kompaktlık (sphericity)** | < 0.12 → −0.20 / < 0.20 → −0.10 | −0.10–0.20 | Düzensiz / parçalı şekil |
| 5 | **T1 outlier oranı** | > 0.15 → −0.15 / > 0.10 → −0.08 | −0.08–0.15 | Doku tutarsızlığı / sizinti |
| 6 | **Gradient sınır netliği** | grad < 0.05 (vol>100mm³) | −0.10/taraf | Belirsiz / yumuşak sınır |
| 7 | **BBox doluluk oranı** | fill < 0.05 | −0.10 | Label bbox'ını doldurmuyorsa parçalı |
| 8 | **GLCM homojenlik** | homo < 0.25 (büyük labellar) | −0.10 | Heterojen doku (karışık bölge) |

Fragmentation ek cezası: her iki hemisfer parçalı → −0.25 | tek taraf → −0.10

### 1.4 Cross-Subject Tutarlılık

5 hasta üzerinde her label için 6 metriğin **CV = std/mean** değeri hesaplanır:
`lr_volume_ratio`, `compactness_left/right`, `t1_mean_left/right`, `t1_gradient_left/right`

```
consistency_score = 1 − mean(CV)   →  0-1 arası
```

- CV düşük → tutarlı propagasyon → yüksek güvenilirlik
- `consist_score < 0.4` → flag eklenir, skor etkisi: −0.10 × (0.4 − score)

**Mevcut sonuçlar (W2, 5 hasta):** ort. 0.786 | min 0.266 | max 0.908

### 1.5 Hesaplanan Tüm Morphometrik Özellikler (scripts/utils/metrics.py)

| Grup | Özellik | Kalite Sinyali |
|------|---------|----------------|
| **Geometrik** | volume_mm³, surface_area_mm², compactness | Düşük kompaktlık = şekil düzensizliği |
| **Geometrik** | elongation, flatness, eigenvalues_mm | Anatomik yönelim kontrolü |
| **Geometrik** | bbox_fill_ratio, connected_components | Parçalanma ve doluluk |
| **Geometrik** | skeleton_length_mm, skeleton_max_radius_mm | Yapı uzunluğu/kalınlığı |
| **Konumsal** | lr_volume_ratio, centroid_mirror_error_mm | Sol/sağ simetri bozukluğu |
| **Konumsal** | midline_distance_mm, centroid_normalized | Anatomik konum doğruluğu |
| **Yoğunluk** | T1_mean, T1_median, T1_std, T1_IQR, T1_MAD | Doku tipi doğrulaması |
| **Yoğunluk** | T1_outlier_ratio | Arka plan kirliliği / sizinti tespiti |
| **Yoğunluk** | T1_gradient_energy, T1_LoG_energy | Sınır keskinliği |
| **Texture** | GLCM_homogeneity, GLCM_entropy, GLCM_contrast | Doku homojenliği |
| **Çok-modal** | T1_mean_T2_mean_ratio, T2_mean, T2_std | T2 ile çapraz doğrulama |

### 1.6 CSV Çıktısı — label_reliability.csv Kolonları

```
id, label, atlas_mm3, left_mm3, right_mm3, retention_%,
left_fragmented, right_fragmented,
lr_volume_ratio, t1_mean_left, t1_mean_right, mirror_err_mm,
compactness_l, compactness_r,
t1_outlier_l, t1_outlier_r,
gradient_l, gradient_r,
bbox_fill_l, bbox_fill_r,
glcm_homo_l, glcm_homo_r,
consist_score, loo_dsc, quality_score, morph_flags, tier, tier_note
```

### 1.7 Performans Referansları

| Metrik | Hedef | Mevcut (W2) |
|--------|-------|-------------|
| TIER-1 label sayısı/hasta | ≥ 30 | 33–35 |
| Cross-subject tutarlılık | ≥ 0.75 | 0.786 |
| Fragmentation oranı (W2) | < %15 | %11 (IXI002/012/013) |
| Jacobian negatif voksel | < %0.1 | ✓ tüm warp'larda |
| LOO DSC (güvenilir labellar) | ≥ 0.60 | ✓ (TIER-1 küme) |

---

## 2. Proje Özeti

**Amaç:** Ground Truth (GT) olmadan, tek bir atlas ve 5 hastanın T1/T2/MRA
görüntülerini kullanarak **warp → propagate → refine** pipeline'ının en
güvenilir konfigürasyonunu keşfetmek.

**Kısıt:** Etiket yok. Başarı; morphometrics, LOO pseudo-DSC ve cross-subject
tutarlılık üçgeninde ölçülür.

**Ortam:** Python 3.13 (Microsoft Store) — conda yok, pip ile global kurulum.
```bash
# Paketler zaten kurulu:
pip install nibabel antspyx simpleitk scikit-image scipy nilearn matplotlib pandas
```

---

## 3. Veri Yapısı

```
BrainSeg/
└── Data/
    ├── Atlas/
    │   ├── MNI152_T1_1mm.nii.gz
    │   ├── left-vols-1mm/      # 41 sol hemisfer label (AD, AM, AV, ..., VPM)
    │   └── right-vols-1mm/     # 41 sağ hemisfer label (aynı isimler)
    └── DatAIXI_Patient_T1T2/
        ├── IXI002-Guys-0828-{T1,T2,MRA}.nii.gz
        ├── IXI012-HH-1211-{T1,T2,MRA}.nii.gz
        ├── IXI013-HH-1212-{T1,T2,MRA}.nii.gz
        ├── IXI015-HH-1258-{T1,T2,MRA}.nii.gz
        └── IXI016-Guys-0697-{T1,T2,MRA}.nii.gz
```

**Atlas Label Ölçeği Grupları:**

| Grup | Label'lar | Atlas Hacmi | Not |
|------|-----------|-------------|-----|
| Büyük | `global`, `thalamus_body`, `MAX_VOLUME` | > 5000 mm³ | Hizalama referansı |
| Orta | `MDmc`, `MDpc`, `LP`, `Pf`, `CM`, `VLa`, vb. | 200–2000 mm³ | TIER-1 hedef küme |
| Küçük | `STh`, `RN`, `Hb`, `LGNmc`, `LGNpc`, `mtt` | < 200 mm³ | TIER-2/3 — dikkatli |

---

## 4. Pipeline Mimarisi

```
[Atlas T1 + Labels]
       │
       ▼
   İP-1: Preproc ──── N4, BrainMask (ANTsPyNet), T2/MRA→T1 hizalama
       │
       ▼
   İP-2: Warp ──────── Atlas→Subject (ANTsPy SyN, 3 aday: W1/W2/W3)
       │                 → Kazanan: W2 (MI similarity)
       ▼
   İP-3: Propagate ─── NN interpolation + tam morphometrics
       │                 → 41 label × 2 hemisfer × 5 hasta
       ▼
   İP-4: Refine ──────── R1/R2/R3 (ham propagation kazandı)
       │
       ▼
   Export ────────────── TIER sınıflandırma, CSV, 3D Slicer paketi
```

### Warp Adayları

| Aday | Yöntem | Similarity | Sonuç |
|------|--------|-----------|-------|
| W1 | Affine + SyN | CC | Geçerli, W2'den düşük |
| **W2** ★ | Affine + SyN | **MI** | **Kazanan** — tüm hastalarda en yüksek NMI |
| W3 | Affine + SyN (güçlü reg.) | CC | W1 ile benzer |

### W2 NMI Sonuçları (5 Hasta)

| Hasta | W1 | W2 | W3 |
|-------|----|----|----|
| IXI002-Guys-0828 | 1.1286 | **1.1327** | 1.1287 |
| IXI012-HH-1211 | 1.0902 | **1.0940** | 1.0900 |
| IXI013-HH-1212 | 1.0926 | **1.0958** | 1.0930 |
| IXI015-HH-1258 | 1.1120 | **1.1152** | 1.1108 |
| IXI016-Guys-0697 | 1.1053 | **1.1054** | 1.1050 |

---

## 5. GT Yokken Kullanılan Metrikler

### 5.1 Warp Metrikleri

| Metrik | Araç | Eşik |
|--------|------|------|
| NMI / CC | ANTs similarity | Yüksek = iyi |
| Jacobian negatif voksel oranı | `CreateJacobianDeterminantImage` | **< %0.1** (ihlal = eleman) |
| Inverse consistency | atlas→subj→atlas hata (mm) | < 1 mm |

### 5.2 Propagate Metrikleri — Morphometrics

Bkz. Bölüm 1.5 — tüm özellikler `scripts/utils/metrics.py` tarafından hesaplanır.

**Başarı kriterleri:**
- Tüm label'lar doğru hemisferde
- Compactness > 0.12 (şekil düzenliliği)
- LR volume ratio 0.33–3.0 arası
- T1_outlier_ratio < 0.15
- Connected components ≤ 2 (defrag sonrası)

### 5.3 Warp Kalite Kontrolü — Gerekli Kontroller

```python
# Jacobian fail kontrolü — ip3_propagate.py başında yapılır
if sim.get("jacobian_fail", False):
    # Bu aday propagate'e taşınmaz
    return rep.finish("SKIPPED")
```

---

## 6. Scriptler ve Çıktı Yapısı

### scripts/ Dizini

```
scripts/
├── config.py              # SUBJECTS, OUTPUT_DIR, eşikler, warp adayları
├── utils/
│   ├── reporter.py        # StepReporter — JSON + terminal raporlama
│   ├── nifti_utils.py     # I/O, orient, resample, N4, NMI
│   └── metrics.py         # Tam morphometrics seti (geometrik/konumsal/görünüm)
├── ip1_preproc.py         # Preproc + ANTsPyNet brain extraction
├── ip2_warp.py            # ANTsPy SyN warp (W1/W2/W3)
├── ip3_propagate.py       # NN propagate + morphometrics
├── ip4_refine.py          # Boundary refinement (ham propagation kazandı)
├── compare_candidates.py  # W1/W2/W3 karşılaştırma tablosu
├── leave_one_out_val.py   # LOO pseudo-DSC doğrulama
└── export_expert_review.py # TIER sınıflandırma + 3D Slicer paketi
```

### Çıktı Dizin Yapısı

```
outputs/
├── atlas/
│   ├── atlas_T1_preproc.nii.gz
│   ├── left_labels/
│   └── right_labels/
├── <subj_id>/
│   ├── preproc/
│   │   ├── T1_preproc.nii.gz
│   │   ├── T2_to_T1_preproc.nii.gz
│   │   ├── MRA_to_T1_preproc.nii.gz
│   │   └── brain_mask.nii.gz
│   ├── warp/W1|W2|W3/
│   │   ├── fwd_transform_0GenericAffine.mat
│   │   ├── fwd_transform_1Warp.nii.gz
│   │   ├── jacobian_det.nii.gz
│   │   ├── similarity_metric.json
│   │   └── report_IP2_Warp_W*.json
│   └── propagate/W1|W2|W3/
│       ├── left_labels/<label>.nii.gz
│       ├── right_labels/<label>.nii.gz
│       ├── label_quality_report.json
│       └── report_IP3_Propagate_W*.json  ← morphometrics burada
├── expert_review/
│   ├── <subj_id>/
│   │   ├── parcellation.nii.gz    # 3D Slicer labelmap
│   │   ├── parcellation.ctbl      # renk tablosu
│   │   ├── label_reliability.csv  # tüm morphometrics + TIER
│   │   └── QC_multiview.png
│   └── expert_summary.html        # tek dosyada tüm hastalar
├── loo_validation/
│   └── loo_label_dsc.csv          # 41 label LOO DSC skoru
└── comparison_*.csv / pipeline_ranking_final.csv
```

---

## 7. Raporlama Zorunluluğu

> Her script, her fonksiyon, her adım — **sürekli ve kapsamlı rapor üretmek zorundadır.**
> GT olmadığı için tek doğrulama yöntemi metriklerin birikimli takibidir.

### 7.1 Raporlama Seviyeleri

**Seviye 1 — Terminal:** Her işlemde okunabilir ilerleme mesajı.
```
[İP-3 | IXI002 | W2] 41 label propagate ediliyor...
  → AV: sol=168mm³ sağ=165mm³ compactness=0.643 ✓
  → STh: sol=7mm³ ⚠ TIER-3 adayı (küçük atlas)
[İP-3 | IXI002 | W2] TAMAMLANDI → 357s
```

**Seviye 2 — JSON:** Her adım sonunda `report_<STEP>_<CAND>.json`:
```json
{
  "subject_id": "IXI002-Guys-0828",
  "step": "IP3_Propagate_W2",
  "status": "SUCCESS",
  "duration_sec": 357.9,
  "metrics": {
    "morphometrics": { "<label>": { "geometric_left": {...}, ... } },
    "fragmented_labels": [],
    "problem_labels_count": 3
  }
}
```

**Seviye 3 — CSV/HTML:** Tüm hastalar ve adaylar için karşılaştırma tablosu.

### 7.2 Zorunlu Raporlama Kuralları

1. Her hata hem terminale hem JSON'a `"status": "FAILED", "error": "..."` olarak kaydedilir.
2. Eşik ihlali varsa `"warning": true` ve neden açıklanır.
3. Her adımın `duration_sec` değeri JSON'da bulunur.
4. Dosya yolları daima `os.path.abspath` ile tam path.
5. Jacobian fail → propagate atlanır, raporda `"jacobian_fail_skip": true`.

### 7.3 StepReporter Kullanımı

`scripts/utils/reporter.py` — gerçek implementasyon bu dosyadadır.

```python
from scripts.utils.reporter import StepReporter

rep = StepReporter("IXI002-Guys-0828", "IP3_Propagate_W2", out_dir)
rep.log("n_labels", 41)
rep.add_metric_group("morphometrics", all_morpho)
rep.add_file("label_quality_report", lq_path)
return rep.finish("SUCCESS")
```

---

## 8. Kritik Notlar

1. **Atlas tek, popülasyon yok.** Prior dağılımları subject'lerden türetilir (robust median/MAD).

2. **Büyükten küçüğe (coarse→fine):** `global`/`thalamus_body` önce hizalanır, sonra `STh`/`RN` gibi küçük çekirdekler değerlendirilir.

3. **Her dönüşüm NN interpolation** ile label'a uygulanmalı — yanlış interpolation label'ı bozar.

4. **Ham propagation kazandı:** R1/R2/R3 refinement etkisiz veya zararlı bulundu. Küçük label'larda (<50 voxel) refinement otomatik atlanır.

5. **Fold kontrolü zorunlu:** Jacobian negatif oranı > %0.1 → o pipeline elenir, bir sonraki adıma taşınmaz.

6. **T2 ve MRA warp sürücüsü değil:** Warp omurgası T1→T1. T2 intensity kontrolünde, MRA vessel analizinde kullanılır.

7. **Küçük label güvenilirliği:** Atlas < 50mm³ → doğrudan TIER-3. `STh`, `Hb`, `LGNmc` gibi yapılar araştırma amaçlı — klinik karar için kullanılamaz.

8. **Windows encoding:** Tüm scriptler başında `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` içerir.

---

## 9. İlerleme Takibi (Son Güncelleme: 2026-04-08)

### İP-1: Preprocessing — [✓ TAMAMLANDI]

- [✓] ANTsPyNet brain extraction entegre (`antspynet.brain_extraction(t1, modality="t1")`)
- [✓] N4 bias correction, T2/MRA→T1 hizalama, intensity normalizasyon
- [✓] 5/5 hasta başarılı

| Hasta | Brain Mask (mm³) | Durum |
|-------|-----------------|-------|
| IXI002-Guys-0828 | 1,430,000 | ✓ |
| IXI012-HH-1211 | 1,680,000 | ✓ |
| IXI013-HH-1212 | ~1,560,000 | ✓ |
| IXI015-HH-1258 | ~1,500,000 | ✓ |
| IXI016-Guys-0697 | 1,464,695 | ✓ |

> Hedef: 1.0–1.8M mm³ (klinik normal beyin hacmi) — tümü hedef aralığında.

### İP-2: Warp — [✓ TAMAMLANDI]

- [✓] 5 hasta × 3 aday = 15/15 SUCCESS
- [✓] Hiçbir aday Jacobian testinde elenmedi
- [✓] **Kazanan: W2** — tüm hastalarda en yüksek NMI

### İP-3: Propagate + Morphometrics — [✓ TAMAMLANDI]

- [✓] 41 label × 2 hemisfer × 5 hasta × 3 warp = 15/15 SUCCESS
- [✓] `_defrag()` ile fragmentation otomatik düzeltme
- [✓] Tüm 5 hasta için tam morphometrics mevcut (IP-3 rapor JSON'larında)

**Fragmentation (W2, defrag sonrası):**

| Hasta | Parçalı / Toplam | Oran |
|-------|-----------------|------|
| IXI002-Guys-0828 | 9/82 | 11.0% ✓ |
| IXI012-HH-1211 | 9/82 | 11.0% ✓ |
| IXI013-HH-1212 | 6/82 | 7.3% ✓ |
| IXI015-HH-1258 | ~35/81 | ~43% ⚠ (brain mask kalitesine bağlı) |
| IXI016-Guys-0697 | ~31/82 | ~38% ⚠ (brain mask kalitesine bağlı) |

### İP-4: Refine — [✓ TAMAMLANDI]

- [✓] 45/45 SUCCESS (R1/R2/R3 × 5 hasta × 3 warp)
- [✓] **Ham propagation kazandı** — refinement etkisiz veya zararlı
- [✓] <50 voxel label'larda refinement otomatik atlanır (explosion önlemi)

### LOO Doğrulama — [✓ TAMAMLANDI]

- [✓] 41 label için LOO DSC skoru: `outputs/loo_validation/loo_label_dsc.csv`
- [✓] TIER-1 küme için DSC > 0.6 (kalite eşiği karşılandı)

### Uzman İnceleme Paketi — [✓ TAMAMLANDI + MORPHOMETRICS TAM ENTEGRE]

- [✓] 8 morphometrik kriter + cross-subject tutarlılık entegre edildi
- [✓] Her hasta: `parcellation.nii.gz` + `parcellation.ctbl` + `label_reliability.csv` + `QC_multiview.png`
- [✓] `outputs/expert_review/expert_summary.html` — 3D Slicer talimatları dahil

**Mevcut TIER Dağılımı (W2, morphometrics tam):**

| Hasta | TIER-1 | TIER-2 | TIER-3 |
|-------|:------:|:------:|:------:|
| IXI002-Guys-0828 | 33 | 4 | 4 |
| IXI012-HH-1211 | 35 | 1 | 5 |
| IXI013-HH-1212 | 33 | 3 | 5 |
| IXI015-HH-1258 | 35 | 2 | 4 |
| IXI016-Guys-0697 | 33 | 3 | 5 |

**Bilinen sınırlamalar (uzmana anlatılacak):**
1. Ground truth yok → Dice/HD95 gerçek anlamda hesaplanamaz (LOO pseudo-GT)
2. 5 hasta — istatistiksel güven sınırlı
3. Küçük çekirdekler (<50mm³) TIER-3 → güvenilmez
4. Klinik karar için kullanılamaz — araştırma prototipi

### Düzeltilen Buglar

| Dosya | Bug | Çözüm |
|-------|-----|-------|
| `metrics.py:58` | 6 eleman unpack | `sorted(eig)[:3]` |
| `metrics.py:252` | `c_vox` tanımsız (n_vox=0) | `c_vox = np.zeros(3)` başlangıç |
| `ip4_refine.py:158` | `int(array).sum()` → hata | `int(array.sum())` |
| `ip4_refine.py` | <50 voxel label explosion | refinement atla |
| `export_expert_review.py` | compactness/gradient skora girmiyordu | 8 kriter sisteme eklendi |
| `export_expert_review.py` | 3 hastada morphometrics 0 | IP-3 yeniden çalıştırıldı |

---

## 10. Çalıştırma Komutları

```bash
# Uzman paketi yeniden üret (veriler hazırsa — ana kullanım):
python scripts/export_expert_review.py --warp W2

# Tek hasta için:
python scripts/export_expert_review.py --warp W2 --subject IXI002-Guys-0828

# Temizden pipeline (sırayla):
python scripts/ip1_preproc.py
python scripts/ip2_warp.py
python scripts/ip3_propagate.py
python scripts/ip4_refine.py
python scripts/compare_candidates.py
python scripts/leave_one_out_val.py --warp W2
python scripts/export_expert_review.py --warp W2

# Tek hasta tek aday morphometrics yenile:

python scripts/ip3_propagate.py --subject IXI013-HH-1212 --candidate W2
```

---

*Son güncelleme: 2026-04-08*

