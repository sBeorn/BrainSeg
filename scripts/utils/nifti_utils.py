"""
NIfTI yardımcı fonksiyonları — I/O, orientation, resample, maskeleme.
"""

import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk


# ── ANTsPy ↔ nibabel dönüştürücüler ──────────────────────────────────────────
# antspyx 0.6.x'te to_nibabel() yok; spacing/origin/direction'dan manuel affine.

def ants_to_nib(ants_img) -> nib.Nifti1Image:
    """ANTsImage → nibabel NIfTI (to_nibabel() olmadan)."""
    data      = ants_img.numpy().astype(np.float32)
    spacing   = np.array(ants_img.spacing, dtype=float)
    origin    = np.array(ants_img.origin,  dtype=float)
    direction = np.array(ants_img.direction, dtype=float).reshape(3, 3)
    affine    = np.eye(4)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3,  3] = origin
    return nib.Nifti1Image(data, affine)


def nib_to_ants(nib_img: nib.Nifti1Image):
    """nibabel NIfTI → ANTsImage (from_nibabel() olmadan)."""
    import ants
    data      = nib_img.get_fdata().astype(np.float32)
    affine    = nib_img.affine
    spacing   = list(np.sqrt((affine[:3, :3] ** 2).sum(axis=0)))
    direction = (affine[:3, :3] / np.array(spacing)).flatten().tolist()
    origin    = affine[:3, 3].tolist()
    return ants.from_numpy(data, origin=origin, spacing=spacing, direction=direction)


def win_path(path: str) -> str:
    """Windows backslash → forward slash (ANTsPy outprefix için)."""
    return path.replace("\\", "/")


# ── I/O ────────────────────────────────────────────────────────────────────

def load_nifti(path: str) -> nib.Nifti1Image:
    """NIfTI dosyasını yükle; dosya yoksa hata fırlat."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"NIfTI bulunamadı: {path}")
    return nib.load(path)


def save_nifti(data: np.ndarray, affine: np.ndarray, path: str,
               header=None) -> None:
    """numpy array'i NIfTI olarak kaydet; dizin yoksa oluştur."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img = nib.Nifti1Image(data, affine, header)
    nib.save(img, path)


def get_voxel_volume(img: nib.Nifti1Image) -> float:
    """Voksel hacmini mm³ cinsinden döndür."""
    zooms = img.header.get_zooms()[:3]
    return float(np.prod(zooms))


# ── Orientation ────────────────────────────────────────────────────────────

def check_orientation(img: nib.Nifti1Image) -> str:
    """Görüntünün orientation kodunu döndür (örn. 'RAS', 'LPS')."""
    return "".join(nib.aff2axcodes(img.affine))


def reorient_to_ras(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Görüntüyü RAS orientasyonuna çevir."""
    return nib.as_closest_canonical(img)


# ── Resample ───────────────────────────────────────────────────────────────

def resample_to_spacing(
    img: nib.Nifti1Image,
    target_spacing=(1.0, 1.0, 1.0),
    interpolation: str = "linear",
) -> nib.Nifti1Image:
    """
    Görüntüyü hedef voksel boyutuna yeniden örnekle.
    interpolation: 'linear' | 'nearest' | 'bspline'
    """
    sitk_img = _nib_to_sitk(img)
    original_spacing = sitk_img.GetSpacing()
    original_size    = sitk_img.GetSize()

    new_size = [
        int(round(original_size[i] * (original_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]

    interp_map = {
        "linear":  sitk.sitkLinear,
        "nearest": sitk.sitkNearestNeighbor,
        "bspline": sitk.sitkBSpline,
    }
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(list(target_spacing))
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interp_map.get(interpolation, sitk.sitkLinear))
    resampler.SetDefaultPixelValue(0)

    resampled = resampler.Execute(sitk_img)
    return _sitk_to_nib(resampled)


# ── Brain Mask ─────────────────────────────────────────────────────────────

def apply_brain_mask(
    img: nib.Nifti1Image, mask: nib.Nifti1Image
) -> nib.Nifti1Image:
    """Binary maske ile görüntüyü maskele."""
    data      = img.get_fdata()
    mask_data = (mask.get_fdata() > 0).astype(data.dtype)
    return nib.Nifti1Image(data * mask_data, img.affine, img.header)


def compute_brain_mask_simple(
    t1_img: nib.Nifti1Image, threshold_percentile: float = 15.0
) -> nib.Nifti1Image:
    """
    Konservatif beyin maskesi: kafa maskesi + erozyonla skull çıkarma.
    HD-BET/BET mevcut değilse fallback olarak kullanılır.
    """
    from scipy import ndimage

    data    = t1_img.get_fdata().astype(np.float32)
    spacing = np.array(t1_img.header.get_zooms()[:3])
    nz      = data[data > 0]

    # ── 1. Adım: kaba kafa maskesi (düşük eşik) ───────────────────────────
    try:
        from skimage.filters import threshold_otsu
        thr_otsu = threshold_otsu(nz)
    except Exception:
        thr_otsu = np.percentile(nz, threshold_percentile)

    # Düşük eşik → kafa + beyin birlikte
    head_mask = (data > thr_otsu * 0.2).astype(bool)

    struct6 = ndimage.generate_binary_structure(3, 1)   # 6-bağlantı
    struct26 = ndimage.generate_binary_structure(3, 2)  # 26-bağlantı

    # Gürültü temizle
    head_mask = ndimage.binary_opening(head_mask, structure=struct6, iterations=1)
    head_mask = ndimage.binary_fill_holes(head_mask)

    # En büyük bileşen (kafa)
    labeled, n = ndimage.label(head_mask)
    if n > 1:
        sizes   = ndimage.sum(head_mask, labeled, range(1, n + 1))
        largest = int(np.argmax(sizes)) + 1
        head_mask = (labeled == largest)

    # ── 2. Adım: skull stripping — erozyon ~6mm ───────────────────────────
    # 1mm iso'da 6 iterasyon ≈ 6mm iç büzülme (skull kalınlığı ~4–7mm)
    erosion_mm  = 6.0
    erosion_itr = max(1, int(np.ceil(erosion_mm / spacing.mean())))
    brain_mask  = ndimage.binary_erosion(
        head_mask, structure=struct6, iterations=erosion_itr
    )

    # ── 3. Adım: delik doldur + en büyük bileşen ──────────────────────────
    brain_mask = ndimage.binary_fill_holes(brain_mask)
    labeled, n = ndimage.label(brain_mask)
    if n > 1:
        sizes   = ndimage.sum(brain_mask, labeled, range(1, n + 1))
        largest = int(np.argmax(sizes)) + 1
        brain_mask = (labeled == largest)

    # ── 4. Adım: hafif dilation (CSF dahil) ───────────────────────────────
    brain_mask = ndimage.binary_dilation(
        brain_mask, structure=struct6, iterations=2
    )
    brain_mask = ndimage.binary_fill_holes(brain_mask).astype(np.uint8)

    return nib.Nifti1Image(brain_mask, t1_img.affine, t1_img.header)


# ── Intensity Normalization ─────────────────────────────────────────────────

def normalize_intensity(
    img: nib.Nifti1Image,
    mask: nib.Nifti1Image = None,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> nib.Nifti1Image:
    """
    Robust percentile clipping + z-score normalization.
    mask varsa yalnızca maske içi vokseller kullanılır.
    """
    data = img.get_fdata().astype(np.float32)
    if mask is not None:
        roi = data[mask.get_fdata() > 0]
    else:
        roi = data[data > 0]

    lo  = np.percentile(roi, lower_pct)
    hi  = np.percentile(roi, upper_pct)
    data = np.clip(data, lo, hi)

    mean = roi.clip(lo, hi).mean()
    std  = roi.clip(lo, hi).std()
    if std > 0:
        data = (data - mean) / std

    return nib.Nifti1Image(data, img.affine, img.header)


# ── N4 Bias Correction ─────────────────────────────────────────────────────

def n4_bias_correction(
    img: nib.Nifti1Image,
    mask: nib.Nifti1Image = None,
    n_iterations: list = None,
) -> tuple:
    """
    SimpleITK N4 bias field correction.
    Döndürür: (corrected_img, convergence_iters)
    """
    if n_iterations is None:
        n_iterations = [50, 50, 25]

    sitk_img = _nib_to_sitk(img)
    sitk_img = sitk.Cast(sitk_img, sitk.sitkFloat32)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(n_iterations)

    if mask is not None:
        sitk_mask = _nib_to_sitk(mask)
        sitk_mask = sitk.Cast(sitk_mask, sitk.sitkUInt8)
        corrected = corrector.Execute(sitk_img, sitk_mask)
    else:
        corrected = corrector.Execute(sitk_img)

    return _sitk_to_nib(corrected), n_iterations


# ── Similarity Metrics ─────────────────────────────────────────────────────

def compute_nmi(img1: nib.Nifti1Image, img2: nib.Nifti1Image,
                mask: nib.Nifti1Image = None, bins: int = 64) -> float:
    """Normalized Mutual Information hesapla."""
    d1 = img1.get_fdata().ravel()
    d2 = img2.get_fdata().ravel()

    if mask is not None:
        m = mask.get_fdata().ravel() > 0
        d1, d2 = d1[m], d2[m]

    joint_hist, _, _ = np.histogram2d(d1, d2, bins=bins)
    joint_hist = joint_hist / joint_hist.sum()

    p1 = joint_hist.sum(axis=1)
    p2 = joint_hist.sum(axis=0)

    eps = 1e-10
    h1  = -np.sum(p1[p1 > eps] * np.log(p1[p1 > eps]))
    h2  = -np.sum(p2[p2 > eps] * np.log(p2[p2 > eps]))
    h12 = -np.sum(joint_hist[joint_hist > eps] * np.log(joint_hist[joint_hist > eps]))

    return float((h1 + h2) / (h12 + eps))


# ── SimpleITK ↔ nibabel dönüştürücüler ────────────────────────────────────

def _nib_to_sitk(img: nib.Nifti1Image) -> sitk.Image:
    data   = img.get_fdata().astype(np.float32)
    sitk_img = sitk.GetImageFromArray(np.transpose(data, (2, 1, 0)))
    spacing = [float(s) for s in img.header.get_zooms()[:3]]
    sitk_img.SetSpacing(spacing)
    origin  = img.affine[:3, 3].tolist()
    sitk_img.SetOrigin(origin)
    return sitk_img


def _sitk_to_nib(sitk_img: sitk.Image) -> nib.Nifti1Image:
    data    = sitk.GetArrayFromImage(sitk_img)
    data    = np.transpose(data, (2, 1, 0))
    spacing = sitk_img.GetSpacing()
    origin  = sitk_img.GetOrigin()
    affine  = np.diag(list(spacing) + [1.0])
    affine[:3, 3] = origin
    return nib.Nifti1Image(data, affine)
