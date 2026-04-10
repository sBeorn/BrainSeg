"""
Leave-One-Out Cross Validation
================================
Her hastaya sirasyla atlas gibi davranir, diger hastalara propagate eder.
Pseudo-DSC (Dice Similarity Coefficient) hesaplar.

Bu script ground-truth olmadan dogruluk tahmini saglar:
  - 5 hasta -> 5 fold (her seferinde 1 "atlas", 4 "hedef")
  - Her label icin: DSC, Precision, Recall, Hausdorff-95
  - Guvenilir (TIER-1) label'larda DSC > 0.6 hedef

Kullanim:
    python scripts/leave_one_out_val.py
    python scripts/leave_one_out_val.py --warp W2 --label RN
"""

import sys, os, argparse, json, csv, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import nibabel as nib
import ants
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import SUBJECTS, OUTPUT_DIR, LEFT_LABELS, RIGHT_LABELS, WARP_CANDIDATES

# ── Sabitler ─────────────────────────────────────────────────────────────────
LOO_DIR = os.path.join(OUTPUT_DIR, "loo_validation")


# ── Metrikler ─────────────────────────────────────────────────────────────────

def dice(pred, gt):
    pred, gt = (pred > 0).astype(bool), (gt > 0).astype(bool)
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    return float(2 * inter / denom) if denom > 0 else 1.0

def precision(pred, gt):
    pred, gt = (pred > 0).astype(bool), (gt > 0).astype(bool)
    return float((pred & gt).sum() / (pred.sum() + 1e-10))

def recall(pred, gt):
    pred, gt = (pred > 0).astype(bool), (gt > 0).astype(bool)
    return float((pred & gt).sum() / (gt.sum() + 1e-10))

def hausdorff_95(pred, gt, spacing=(1.0, 1.0, 1.0)):
    pred, gt = (pred > 0).astype(bool), (gt > 0).astype(bool)
    if not pred.any() or not gt.any():
        return float("nan")
    try:
        edt_gt   = ndimage.distance_transform_edt(~gt,   sampling=spacing)
        edt_pred = ndimage.distance_transform_edt(~pred, sampling=spacing)
        d_pred_to_gt = edt_gt[pred]
        d_gt_to_pred = edt_pred[gt]
        return float(max(
            np.percentile(d_pred_to_gt, 95),
            np.percentile(d_gt_to_pred, 95)
        ))
    except Exception:
        return float("nan")


# ── Label ismi listesi ────────────────────────────────────────────────────────

def get_label_names(label_filter=None):
    names = [f.replace(".nii.gz","") for f in sorted(os.listdir(LEFT_LABELS))
             if f.endswith(".nii.gz")]
    if label_filter:
        names = [n for n in names if n in label_filter]
    return names


# ── Tek LOO fold: atlas_subj -> target_subj ───────────────────────────────────

def run_loo_fold(atlas_subj, target_subj, warp_id, label_names):
    """
    atlas_subj T1'ini kullanarak target_subj T1'ine kayit yap,
    ardından atlas_subj'in propagate edilmis labellarini target'e aktar.
    target_subj'in kendi propagate labellarini GT olarak kullan.
    """
    fold_dir = os.path.join(LOO_DIR, f"atlas-{atlas_subj}_target-{target_subj}", warp_id)
    os.makedirs(fold_dir, exist_ok=True)

    # Daha once yapildiysa atla
    result_path = os.path.join(fold_dir, "metrics.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)

    atlas_t1_path  = os.path.join(OUTPUT_DIR, atlas_subj,  "preproc", "T1_preproc.nii.gz")
    target_t1_path = os.path.join(OUTPUT_DIR, target_subj, "preproc", "T1_preproc.nii.gz")
    atlas_mask_path = os.path.join(OUTPUT_DIR, atlas_subj, "preproc", "brain_mask.nii.gz")

    if not os.path.exists(atlas_t1_path) or not os.path.exists(target_t1_path):
        return None

    fixed_ants  = ants.image_read(target_t1_path)
    moving_ants = ants.image_read(atlas_t1_path)
    mask_ants   = ants.image_read(atlas_mask_path) if os.path.exists(atlas_mask_path) else None

    cfg = WARP_CANDIDATES[warp_id]

    print(f"    Registration {atlas_subj} -> {target_subj} ({warp_id})...")
    t0 = time.time()
    reg_kwargs = dict(
        fixed             = fixed_ants,
        moving            = moving_ants,
        type_of_transform = cfg["type_of_transform"],
        aff_metric        = cfg.get("aff_metric", "mattes"),
        syn_metric        = cfg.get("syn_metric", "CC"),
        syn_sampling      = cfg.get("syn_metric_params", [4])[0],
        grad_step         = cfg.get("grad_step", 0.1),
        flow_sigma        = cfg.get("flow_sigma", 3.0),
        total_sigma       = cfg.get("total_sigma", 0.0),
        reg_iterations    = tuple(cfg.get("syn_iterations", [100, 70, 50, 20])),
        outprefix         = os.path.join(fold_dir, "reg_").replace("\\","/"),
        verbose           = False,
    )
    if mask_ants is not None:
        reg_kwargs["mask"] = mask_ants
    tx = ants.registration(**reg_kwargs)
    print(f"      Tamamlandi ({time.time()-t0:.0f}s)")

    # Atlas propagate labellarini target uzayina tasi
    atlas_left  = os.path.join(OUTPUT_DIR, atlas_subj, "propagate", warp_id, "left_labels")
    atlas_right = os.path.join(OUTPUT_DIR, atlas_subj, "propagate", warp_id, "right_labels")
    gt_left     = os.path.join(OUTPUT_DIR, target_subj, "propagate", warp_id, "left_labels")
    gt_right    = os.path.join(OUTPUT_DIR, target_subj, "propagate", warp_id, "right_labels")

    if not os.path.exists(atlas_left) or not os.path.exists(gt_left):
        return None

    spacing = tuple(float(s) for s in nib.load(target_t1_path).header.get_zooms()[:3])
    metrics = {}

    for lname in label_names:
        for side, src_dir, gt_dir in [
            ("left",  atlas_left,  gt_left),
            ("right", atlas_right, gt_right),
        ]:
            src_p = os.path.join(src_dir, f"{lname}.nii.gz")
            gt_p  = os.path.join(gt_dir,  f"{lname}.nii.gz")
            if not os.path.exists(src_p) or not os.path.exists(gt_p):
                continue

            # Atlas label'i target uzayina tasi
            moving_label = ants.image_read(src_p)
            warped_label = ants.apply_transforms(
                fixed=fixed_ants, moving=moving_label,
                transformlist=tx["fwdtransforms"],
                interpolator="nearestNeighbor",
            )
            pred = warped_label.numpy()
            gt   = nib.load(gt_p).get_fdata()

            key = f"{lname}_{side}"
            metrics[key] = {
                "dice":         round(dice(pred, gt),         4),
                "precision":    round(precision(pred, gt),    4),
                "recall":       round(recall(pred, gt),       4),
                "hausdorff_95": round(hausdorff_95(pred, gt, spacing), 2),
                "pred_vol_mm3": round(float(pred.sum()), 1),
                "gt_vol_mm3":   round(float(gt.sum()),   1),
            }

    result = {
        "atlas":   atlas_subj,
        "target":  target_subj,
        "warp":    warp_id,
        "metrics": metrics,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


# ── Sonuclari ozetle ──────────────────────────────────────────────────────────

def summarize_results(all_results, label_names, out_dir):
    # Her label icin ortalama DSC
    label_dsc = {}
    for lname in label_names:
        dsc_vals = []
        for res in all_results:
            if res is None:
                continue
            for side in ["left", "right"]:
                key = f"{lname}_{side}"
                if key in res["metrics"]:
                    d = res["metrics"][key]["dice"]
                    if not np.isnan(d):
                        dsc_vals.append(d)
        if dsc_vals:
            label_dsc[lname] = {
                "mean_dsc":   round(float(np.mean(dsc_vals)), 4),
                "std_dsc":    round(float(np.std(dsc_vals)),  4),
                "min_dsc":    round(float(np.min(dsc_vals)),  4),
                "max_dsc":    round(float(np.max(dsc_vals)),  4),
                "n_samples":  len(dsc_vals),
            }

    # CSV kaydet
    csv_path = os.path.join(out_dir, "loo_label_dsc.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["label","mean_dsc","std_dsc","min_dsc","max_dsc","n_samples"])
        w.writeheader()
        for lname, v in sorted(label_dsc.items(), key=lambda x: -x[1]["mean_dsc"]):
            w.writerow({"label": lname, **v})

    # Ekrana yazdir
    print(f"\n{'='*60}")
    print(f"  LEAVE-ONE-OUT SONUCLARI")
    print(f"{'='*60}")
    print(f"{'Label':15s} {'Mean DSC':>9} {'Std':>6} {'Min':>6} {'Max':>6}")
    print(f"{'-'*50}")
    good  = [(k,v) for k,v in label_dsc.items() if v["mean_dsc"] >= 0.6]
    ok    = [(k,v) for k,v in label_dsc.items() if 0.4 <= v["mean_dsc"] < 0.6]
    poor  = [(k,v) for k,v in label_dsc.items() if v["mean_dsc"] < 0.4]

    for group, name in [(good,"IYI  (DSC>=0.6)"), (ok,"ORTA (0.4-0.6)"), (poor,"ZAYIF(<0.4)")]:
        print(f"\n  --- {name} ---")
        for lname, v in sorted(group, key=lambda x: -x[1]["mean_dsc"]):
            print(f"  {lname:15s} {v['mean_dsc']:>9.4f} {v['std_dsc']:>6.4f} "
                  f"{v['min_dsc']:>6.4f} {v['max_dsc']:>6.4f}")

    overall_mean = np.mean([v["mean_dsc"] for v in label_dsc.values()])
    print(f"\n  GENEL ORTALAMA DSC: {overall_mean:.4f}")
    print(f"  Iyi (>=0.6): {len(good)}/{len(label_dsc)} label")
    print(f"  Kaydedildi: {csv_path}")
    print(f"{'='*60}\n")

    return label_dsc


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warp",  default="W2", choices=list(WARP_CANDIDATES.keys()))
    parser.add_argument("--label", default=None, help="Tek label testi (orn: RN)")
    args = parser.parse_args()

    label_names = get_label_names([args.label] if args.label else None)
    os.makedirs(LOO_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Leave-One-Out Validation — Warp: {args.warp}")
    print(f"  {len(SUBJECTS)} hasta, {len(label_names)} label")
    print(f"  Toplam fold: {len(SUBJECTS)*(len(SUBJECTS)-1)} registration")
    print(f"{'='*60}\n")

    all_results = []
    for atlas in SUBJECTS:
        for target in SUBJECTS:
            if atlas == target:
                continue
            print(f"  Fold: atlas={atlas} -> target={target}")
            res = run_loo_fold(atlas, target, args.warp, label_names)
            if res:
                all_results.append(res)

    summarize_results(all_results, label_names, LOO_DIR)


if __name__ == "__main__":
    main()
