"""
Morphometrics hesaplama modülü — İP-3 ve İP-4 için kapsamlı metrik seti.

Bölüm 6.2 (Context dosyası) — Geometrik, Konumsal ve Görünüm metrikleri.
"""

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

try:
    from skimage import measure, morphology, feature
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("  ⚠  scikit-image bulunamadı; bazı shape metrikleri devre dışı.")


# ══════════════════════════════════════════════════════════════════════════════
# 6.2.1  Geometrik / Şekil Morphometrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_geometric_morphometrics(
    mask: np.ndarray, voxel_spacing: tuple
) -> dict:
    """
    Parameters
    ----------
    mask          : binary 3D array (bool veya 0/1)
    voxel_spacing : (dx, dy, dz) mm

    Returns
    -------
    dict — hacim, yüzey, kompaktlık, elongation, flatness, skeleton vb.
    """
    mask = (mask > 0).astype(bool)
    voxel_vol = float(np.prod(voxel_spacing))
    n_voxels  = int(mask.sum())

    if n_voxels == 0:
        return _empty_geom()

    volume_mm3 = n_voxels * voxel_vol

    # Yüzey alanı (marching cubes)
    surface_area_mm2 = _surface_area(mask, voxel_spacing)

    # Kompaktlık (Sphericity)
    if surface_area_mm2 > 0:
        compactness = (36 * np.pi * volume_mm3 ** 2) ** (1 / 3) / surface_area_mm2
    else:
        compactness = 0.0

    # PCA eigenvalues → elongation & flatness
    coords = np.array(np.where(mask)).T.astype(float)
    coords *= np.array(voxel_spacing)   # mm'ye dönüştür
    eigenvalues = _pca_eigenvalues(coords)
    lam1, lam2, lam3 = (sorted(eigenvalues, reverse=True) + [1e-10, 1e-10, 1e-10])[:3]
    elongation = float(lam1 / (lam3 + 1e-10))
    flatness   = float(lam2 / (lam3 + 1e-10))

    # Bounding box
    bbox = _bounding_box_mm(mask, voxel_spacing)

    # Bounding box doluluk oranı
    bbox_vol = float(np.prod(bbox["size_mm"]))
    fill_ratio = volume_mm3 / bbox_vol if bbox_vol > 0 else 0.0

    # Connected component sayısı
    _, n_cc = ndimage.label(mask)

    # Skeleton
    skel_info = _skeleton_info(mask, voxel_spacing)

    return {
        "volume_mm3":             round(volume_mm3, 2),
        "surface_area_mm2":       round(surface_area_mm2, 2),
        "compactness":            round(float(compactness), 4),
        "elongation":             round(elongation, 4),
        "flatness":               round(flatness, 4),
        "eigenvalues_mm":         [round(float(e), 3) for e in [lam1, lam2, lam3]],
        "bbox_size_mm":           bbox["size_mm"],
        "bbox_origin_vox":        bbox["origin_vox"],
        "bbox_fill_ratio":        round(fill_ratio, 4),
        "connected_components":   n_cc,
        "skeleton_length_mm":     skel_info["length_mm"],
        "skeleton_max_radius_mm": skel_info["max_radius_mm"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.2.2  Konumsal Morphometrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_positional_morphometrics(
    mask_L: np.ndarray,
    mask_R: np.ndarray,
    brain_mask: np.ndarray,
    voxel_spacing: tuple,
    label_name: str,
    neighbor_masks: dict = None,
) -> dict:
    """
    Sol ve sağ hemisfer maskelerine göre konum ve simetri metrikleri.

    neighbor_masks : {"label_name": binary_array, ...}
    """
    voxel_spacing = np.array(voxel_spacing)
    brain_bbox    = _bounding_box_mm(brain_mask > 0, voxel_spacing)
    brain_dims    = np.array(brain_bbox["size_mm"]) + 1e-10

    def centroid_mm(m):
        if m.sum() == 0:
            return None
        c_vox = np.array(ndimage.center_of_mass(m > 0))
        return (c_vox * voxel_spacing).tolist()

    c_L = centroid_mm(mask_L)
    c_R = centroid_mm(mask_R)

    vol_L = float(mask_L.sum()) * float(np.prod(voxel_spacing))
    vol_R = float(mask_R.sum()) * float(np.prod(voxel_spacing))
    lr_vol_ratio = vol_L / (vol_R + 1e-10)

    # Midline — x-ekseni ortası (brain_bbox origin + size/2)
    midline_x_mm = float(brain_bbox["origin_mm"][0]) + brain_dims[0] / 2

    midline_dist_L = abs(c_L[0] - midline_x_mm) if c_L else None
    midline_dist_R = abs(c_R[0] - midline_x_mm) if c_R else None

    # Centroid yansıma hatası (simetri bozukluğu)
    if c_L and c_R:
        mirror_x_R = 2 * midline_x_mm - c_R[0]
        centroid_mirror_error_mm = abs(c_L[0] - mirror_x_R)
    else:
        centroid_mirror_error_mm = None

    # Normalized centroid
    norm_centroid_L = (
        [round(c_L[i] / brain_dims[i], 4) for i in range(3)] if c_L else None
    )

    # Komşuluk grafı
    adjacency = {}
    if neighbor_masks:
        combined_L = mask_L > 0
        combined_R = mask_R > 0
        struct = ndimage.generate_binary_structure(3, 1)
        dilated_L = ndimage.binary_dilation(combined_L, structure=struct)
        dilated_R = ndimage.binary_dilation(combined_R, structure=struct)
        for nb_name, nb_mask in neighbor_masks.items():
            adjacency[nb_name] = {
                "left":  bool((dilated_L & (nb_mask > 0)).any()),
                "right": bool((dilated_R & (nb_mask > 0)).any()),
            }

    return {
        "label_name":                label_name,
        "volume_mm3_left":           round(vol_L, 2),
        "volume_mm3_right":          round(vol_R, 2),
        "centroid_mm_left":          [round(v, 2) for v in c_L] if c_L else None,
        "centroid_mm_right":         [round(v, 2) for v in c_R] if c_R else None,
        "centroid_normalized_left":  norm_centroid_L,
        "lr_volume_ratio":           round(float(lr_vol_ratio), 4),
        "midline_distance_mm_left":  round(float(midline_dist_L), 2) if midline_dist_L is not None else None,
        "midline_distance_mm_right": round(float(midline_dist_R), 2) if midline_dist_R is not None else None,
        "centroid_mirror_error_mm":  round(float(centroid_mirror_error_mm), 3) if centroid_mirror_error_mm is not None else None,
        "adjacency":                 adjacency,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.2.3  Görünüm (Intensity / Texture) Morphometrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_intensity_morphometrics(
    mask: np.ndarray,
    t1_data: np.ndarray,
    t2_data: np.ndarray = None,
) -> dict:
    """
    Label maskesi içindeki T1 (ve opsiyonel T2) intensite + texture metrikleri.
    """
    roi_t1 = t1_data[mask > 0].astype(np.float64)
    if len(roi_t1) == 0:
        return {}

    result = {}

    # ── Temel istatistikler ──────────────────────────────────────────────────
    result["T1_mean"]   = round(float(roi_t1.mean()), 4)
    result["T1_median"] = round(float(np.median(roi_t1)), 4)
    result["T1_std"]    = round(float(roi_t1.std()), 4)
    q1, q3 = np.percentile(roi_t1, [25, 75])
    result["T1_IQR"]    = round(float(q3 - q1), 4)

    # Robust z-score histogramı (median / MAD)
    mad = float(np.median(np.abs(roi_t1 - np.median(roi_t1))))
    result["T1_MAD"]         = round(mad, 4)
    result["T1_outlier_ratio"] = round(
        float((np.abs((roi_t1 - np.median(roi_t1)) / (mad + 1e-10)) > 3).mean()), 4
    )

    # ── Gradient enerjisi (sınır keskinliği) ────────────────────────────────
    gx, gy, gz = np.gradient(t1_data.astype(float))
    grad_mag   = np.sqrt(gx**2 + gy**2 + gz**2)
    result["T1_gradient_energy"] = round(float(grad_mag[mask > 0].mean()), 4)

    # ── LoG enerjisi (kenar yoğunluğu) ─────────────────────────────────────
    from scipy.ndimage import gaussian_laplace
    log_img = gaussian_laplace(t1_data.astype(float), sigma=1.5)
    result["T1_LoG_energy"] = round(float(np.abs(log_img[mask > 0]).mean()), 4)

    # ── GLCM Texture (2D dilim tabanlı ortalama) ────────────────────────────
    if SKIMAGE_AVAILABLE:
        glcm_metrics = _glcm_from_volume(mask, t1_data)
        result.update(glcm_metrics)

    # ── T1/T2 oranı ─────────────────────────────────────────────────────────
    if t2_data is not None:
        roi_t2 = t2_data[mask > 0].astype(np.float64)
        if len(roi_t2) > 0 and roi_t2.mean() != 0:
            result["T1_mean_T2_mean_ratio"] = round(
                float(roi_t1.mean() / (roi_t2.mean() + 1e-10)), 4
            )
            result["T2_mean"]   = round(float(roi_t2.mean()), 4)
            result["T2_std"]    = round(float(roi_t2.std()), 4)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Birleşik Label Kalite Raporu
# ══════════════════════════════════════════════════════════════════════════════

def compute_label_quality_report(
    mask: np.ndarray,
    voxel_spacing: tuple,
    label_name: str,
    side: str,
    brain_mask: np.ndarray = None,
) -> dict:
    """
    İP-3'ün label_quality_report.json şemasına uygun çıktı üretir.
    """
    mask   = (mask > 0).astype(bool)
    vvol   = float(np.prod(voxel_spacing))
    n_vox  = int(mask.sum())
    volume = n_vox * vvol

    # Centroid
    c_vox = np.zeros(3)
    if n_vox > 0:
        c_vox = np.array(ndimage.center_of_mass(mask))
        c_mm  = (c_vox * np.array(voxel_spacing)).tolist()
    else:
        c_mm = [0.0, 0.0, 0.0]

    # Normalized centroid
    if brain_mask is not None and brain_mask.sum() > 0:
        b_vox   = np.array(ndimage.center_of_mass(brain_mask > 0))
        b_bbox  = _bounding_box_vox(brain_mask > 0)
        b_dims  = np.maximum(np.array(b_bbox["size_vox"]), 1)
        centroid_norm = [
            round(float((c_vox[i] - b_bbox["origin_vox"][i]) / b_dims[i]), 4)
            for i in range(3)
        ]
    else:
        centroid_norm = [round(v, 4) for v in c_mm]

    # Connected components
    _, n_cc = ndimage.label(mask)

    # Kompaktlık
    surf = _surface_area(mask, voxel_spacing)
    compactness = (
        (36 * np.pi * volume ** 2) ** (1 / 3) / surf
        if surf > 0 and volume > 0 else 0.0
    )

    # Midline mesafesi (x-ekseni ortası)
    midline_x_vox = mask.shape[0] / 2.0
    midline_x_mm  = midline_x_vox * voxel_spacing[0]
    midline_dist  = abs(c_mm[0] - midline_x_mm)

    return {
        "label_name":              label_name,
        "side":                    side,
        "volume_mm3":              round(volume, 2),
        "centroid_mm":             [round(v, 2) for v in c_mm],
        "centroid_normalized":     centroid_norm,
        "connected_components":    n_cc,
        "compactness":             round(float(compactness), 4),
        "midline_distance_mm":     round(float(midline_dist), 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Yardımcı (iç) fonksiyonlar
# ══════════════════════════════════════════════════════════════════════════════

def _surface_area(mask: np.ndarray, voxel_spacing: tuple) -> float:
    if not SKIMAGE_AVAILABLE or mask.sum() < 4:
        return 0.0
    try:
        verts, faces, _, _ = measure.marching_cubes(
            mask.astype(float), level=0.5, spacing=voxel_spacing
        )
        # Yüzey alanı = tüm üçgen alanlarının toplamı
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        area  = 0.5 * np.sqrt((cross ** 2).sum(axis=1)).sum()
        return float(area)
    except Exception:
        return 0.0


def _pca_eigenvalues(coords: np.ndarray) -> list:
    if len(coords) < 4:
        return [0.0, 0.0, 0.0]
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals = np.linalg.eigvalsh(cov)
    return sorted(eigvals.tolist(), reverse=True)


def _bounding_box_mm(mask: np.ndarray, voxel_spacing) -> dict:
    voxel_spacing = np.array(voxel_spacing)
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return {"origin_vox": [0, 0, 0], "origin_mm": [0.0, 0.0, 0.0], "size_mm": [0.0, 0.0, 0.0]}
    mins = [int(c.min()) for c in coords]
    maxs = [int(c.max()) for c in coords]
    size_vox = [maxs[i] - mins[i] + 1 for i in range(3)]
    size_mm  = [size_vox[i] * float(voxel_spacing[i]) for i in range(3)]
    origin_mm = [mins[i] * float(voxel_spacing[i]) for i in range(3)]
    return {"origin_vox": mins, "origin_mm": origin_mm, "size_mm": [round(s, 2) for s in size_mm]}


def _bounding_box_vox(mask: np.ndarray) -> dict:
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return {"origin_vox": [0, 0, 0], "size_vox": [1, 1, 1]}
    mins = [int(c.min()) for c in coords]
    maxs = [int(c.max()) for c in coords]
    size_vox = [maxs[i] - mins[i] + 1 for i in range(3)]
    return {"origin_vox": mins, "size_vox": size_vox}


def _skeleton_info(mask: np.ndarray, voxel_spacing: tuple) -> dict:
    if not SKIMAGE_AVAILABLE or mask.sum() < 10:
        return {"length_mm": 0.0, "max_radius_mm": 0.0}
    try:
        skel  = morphology.skeletonize_3d(mask)
        sp    = float(np.prod(voxel_spacing) ** (1 / 3))  # ortalama spacing
        length_mm = float(skel.sum()) * sp

        # EDT → skeleton üzerindeki max inscribed sphere radius
        edt   = ndimage.distance_transform_edt(mask, sampling=voxel_spacing)
        if skel.sum() > 0:
            max_radius = float(edt[skel > 0].max())
        else:
            max_radius = 0.0
        return {"length_mm": round(length_mm, 2), "max_radius_mm": round(max_radius, 3)}
    except Exception:
        return {"length_mm": 0.0, "max_radius_mm": 0.0}


def _glcm_from_volume(mask: np.ndarray, data: np.ndarray, n_slices: int = 5) -> dict:
    """
    Birkaç 2D axial dilimdeki GLCM özelliklerinin ortalamasını al.
    """
    from skimage.feature import graycomatrix, graycoprops

    z_indices = np.where(mask.sum(axis=(0, 1)) > 50)[0]
    if len(z_indices) == 0:
        return {}
    step = max(1, len(z_indices) // n_slices)
    selected = z_indices[::step][:n_slices]

    contrasts, homos, entropies, correls = [], [], [], []

    for z in selected:
        sl_mask = mask[:, :, z]
        sl_data = data[:, :, z]
        if sl_mask.sum() < 10:
            continue

        roi = sl_data[sl_mask > 0]
        lo, hi = np.percentile(roi, 1), np.percentile(roi, 99)
        quantized = np.clip(
            ((sl_data - lo) / (hi - lo + 1e-10) * 63).astype(int), 0, 63
        )
        quantized[sl_mask == 0] = 0

        try:
            glcm = graycomatrix(
                quantized.astype(np.uint8), [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                levels=64, symmetric=True, normed=True,
            )
            contrasts.append(graycoprops(glcm, "contrast").mean())
            homos.append(graycoprops(glcm, "homogeneity").mean())
            correls.append(graycoprops(glcm, "correlation").mean())

            p = glcm.mean(axis=(2, 3))
            p_nz = p[p > 0]
            entropies.append(-np.sum(p_nz * np.log2(p_nz + 1e-10)))
        except Exception:
            continue

    if not contrasts:
        return {}
    return {
        "GLCM_contrast":    round(float(np.mean(contrasts)), 4),
        "GLCM_homogeneity": round(float(np.mean(homos)), 4),
        "GLCM_entropy":     round(float(np.mean(entropies)), 4),
        "GLCM_correlation": round(float(np.mean(correls)), 4),
    }


def _empty_geom() -> dict:
    return {
        "volume_mm3": 0.0, "surface_area_mm2": 0.0, "compactness": 0.0,
        "elongation": 0.0, "flatness": 0.0, "eigenvalues_mm": [0.0, 0.0, 0.0],
        "bbox_size_mm": [0.0, 0.0, 0.0], "bbox_origin_vox": [0, 0, 0],
        "bbox_fill_ratio": 0.0, "connected_components": 0,
        "skeleton_length_mm": 0.0, "skeleton_max_radius_mm": 0.0,
    }
