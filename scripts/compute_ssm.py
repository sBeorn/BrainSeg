"""
SSM Hesaplama — Statistical Shape Model
=========================================
outputs/morphometrics/*_surface.fcsv dosyalarındaki yüzey landmark'larından
her thalamic nucleus için PCA tabanlı şekil modeli hesaplar.

Kullanım:
    python scripts/compute_ssm.py

Çıktı:
    outputs/ssm/<label>_ssm.npz   — her label için SSM verisi
    outputs/ssm/global_morpho_pca.npz  — genel morfometri PCA
"""

import os
import sys
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MORPH_DIR   = os.path.join(PROJECT_DIR, "outputs", "morphometrics")
SSM_DIR     = os.path.join(PROJECT_DIR, "outputs", "ssm")

SUBJECTS = [
    "IXI002-Guys-0828",
    "IXI012-HH-1211",
    "IXI013-HH-1212",
    "IXI015-HH-1258",
    "IXI016-Guys-0697",
]

# 6 yön × 2 taraf = 12 landmark noktası, 36 boyutlu şekil vektörü
DIRECTIONS = ["On", "Arka", "Ust", "Alt", "Lat", "Med"]
SIDES      = ["left", "right"]
N_MODES    = 4   # max PCA modu sayısı


# ─────────────────────────────────────────────────────────────────────────────
# FCSV Okuma
# ─────────────────────────────────────────────────────────────────────────────

def load_surface_fcsv(path):
    """
    Surface FCSV → {label: {side: {direction: [x,y,z]}}}
    Örnek landmark etiketi: "AM_left_On"
    """
    data = {}
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


def build_shape_vector(subject_data, label):
    """
    Bir label için 36-boyutlu şekil vektörü:
    [left_On_x, left_On_y, left_On_z, left_Arka_x, ..., right_Med_z]
    Herhangi bir nokta eksikse None döndürür.
    """
    vec = []
    for side in SIDES:
        for d in DIRECTIONS:
            try:
                pt = subject_data[label][side][d]
            except KeyError:
                return None
            vec.extend(pt)
    return np.array(vec, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# SSM Hesaplama
# ─────────────────────────────────────────────────────────────────────────────

def compute_label_ssm(shape_matrix, subject_list):
    """
    shape_matrix: (n_subjects, 36) float
    Döndürür: dict with keys mean_shape, eigenvalues, eigenvectors, scores, subjects
    """
    X    = np.array(shape_matrix, dtype=np.float64)
    mean = X.mean(axis=0)
    Xc   = X - mean

    n_modes = min(len(X) - 1, N_MODES)

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    eigenvalues  = S[:n_modes] ** 2 / (len(X) - 1)   # varyans per mod
    eigenvectors = Vt[:n_modes].T                      # (36, n_modes)
    scores       = Xc @ eigenvectors                   # (n_subjects, n_modes)

    return {
        "mean_shape":   mean,
        "eigenvalues":  eigenvalues,
        "eigenvectors": eigenvectors,
        "scores":       scores,
        "subjects":     np.array(subject_list),
        "n_modes":      n_modes,
        "n_features":   36,
    }


def main():
    os.makedirs(SSM_DIR, exist_ok=True)

    # ── Tüm subject'ların surface FCSV'lerini yükle ──────────────────────────
    print(f"\n{'='*60}")
    print(f"  SSM Hesaplama  —  {len(SUBJECTS)} subject")
    print(f"{'='*60}")
    print(f"  SSM çıktı: {SSM_DIR}\n")

    all_fcsvs = {}
    for subj in SUBJECTS:
        path = os.path.join(MORPH_DIR, f"{subj}_surface.fcsv")
        if not os.path.exists(path):
            print(f"  UYARI: {path} bulunamadı — subject atlanıyor")
            continue
        all_fcsvs[subj] = load_surface_fcsv(path)

    if not all_fcsvs:
        print("  HATA: Hiç surface.fcsv bulunamadı.")
        print("  Önce: python scripts/compute_morphometrics.py")
        sys.exit(1)

    # ── Her label için SSM ───────────────────────────────────────────────────
    all_labels = sorted(
        set().union(*(set(d.keys()) for d in all_fcsvs.values()))
    )
    print(f"  {len(all_labels)} label bulundu\n")

    ok_count   = 0
    skip_count = 0

    for label in all_labels:
        shapes  = []
        valid_s = []
        for subj in SUBJECTS:
            if subj not in all_fcsvs:
                continue
            vec = build_shape_vector(all_fcsvs[subj], label)
            if vec is not None:
                shapes.append(vec)
                valid_s.append(subj)

        if len(shapes) < 2:
            print(f"  ATLANDI  {label:<15s}  ({len(shapes)} subject — yetersiz)")
            skip_count += 1
            continue

        ssm = compute_label_ssm(shapes, valid_s)

        np.savez(
            os.path.join(SSM_DIR, f"{label}_ssm.npz"),
            **ssm
        )

        ev  = ssm["eigenvalues"]
        tot = float(np.sum(ev))
        pcts = [f"P{i+1}:{100*ev[i]/tot:.1f}%" for i in range(len(ev))] if tot > 0 else []
        print(f"  OK  {label:<15s}  {len(shapes)} subj  {ssm['n_modes']} mod  "
              f"[{' '.join(pcts)}]")
        ok_count += 1

    # ── Global Morfometri PCA ────────────────────────────────────────────────
    print(f"\n  Global morfometri PCA...")
    morph_csv = os.path.join(PROJECT_DIR, "outputs", "morphometrics", "morphometrics_all.csv")
    if os.path.exists(morph_csv):
        df = pd.read_csv(morph_csv)
        feat_cols = ["volume_mm3", "surface_area_mm2", "compactness",
                     "bbox_x_mm", "bbox_y_mm", "bbox_z_mm", "bbox_fill_ratio"]
        df_sub = df[["subject", "label", "side"] + feat_cols].dropna(subset=["volume_mm3"])
        df_sub = df_sub.fillna(df_sub[feat_cols].median())

        df_wide = df_sub.pivot_table(
            index="subject", columns=["label", "side"], values=feat_cols
        )
        df_wide.columns = ["_".join(map(str, c)) for c in df_wide.columns]
        df_wide = df_wide.fillna(df_wide.mean())

        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA as SklearnPCA

        X_g   = StandardScaler().fit_transform(df_wide.values)
        n_c   = min(N_MODES, X_g.shape[0] - 1)
        pca_g = SklearnPCA(n_components=n_c)
        pca_g.fit(X_g)
        scores_g = pca_g.transform(X_g)

        np.savez(
            os.path.join(SSM_DIR, "global_morpho_pca.npz"),
            subjects       = np.array(df_wide.index.tolist()),
            scores         = scores_g,
            variance_ratio = pca_g.explained_variance_ratio_,
            feature_names  = np.array(list(df_wide.columns)),
        )

        vr = pca_g.explained_variance_ratio_
        print(f"  {df_wide.shape[0]} subject  {df_wide.shape[1]} özellik  "
              f"PC1={vr[0]*100:.1f}%  PC2={vr[1]*100:.1f}%")
    else:
        print(f"  UYARI: morphometrics_all.csv bulunamadı — atlandı")

    # ── Özet ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TAMAMLANDI — {ok_count} SSM oluşturuldu, {skip_count} atlandı")
    print(f"  Çıktı klasörü: {SSM_DIR}")
    print(f"{'='*60}")
    print(f"\n  Sonraki adım:")
    print(f"  3D Slicer'ı aç, Python konsolunda:")
    print(f"  exec(open(r'scripts/slicer_ssm_viewer.py').read())")


if __name__ == "__main__":
    main()
