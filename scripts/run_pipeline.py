"""
Master Pipeline Runner
======================
Tüm İP adımlarını sırayla veya seçici olarak çalıştırır.

Kullanım:
    python scripts/run_pipeline.py                          # tam pipeline, tüm hastalar
    python scripts/run_pipeline.py --steps ip1 ip2          # sadece belirtilen adımlar
    python scripts/run_pipeline.py --subject IXI002-Guys-0828
    python scripts/run_pipeline.py --steps ip1 --skip-n4   # hızlı test

Adım sırası: ip1 → ip2 → ip3 → ip4 → compare
"""

import argparse
import os
import sys
import time
from datetime import datetime

# Windows terminal encoding fix
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import SUBJECTS, OUTPUT_DIR, WARP_CANDIDATES, REFINE_CANDIDATES


STEP_ORDER = ["ip1", "ip2", "ip3", "ip4", "compare", "ip5"]


def run_ip1(subjects, kwargs):
    from scripts.ip1_preproc import preprocess_subject
    results = []
    for subj in subjects:
        r = preprocess_subject(subj, skip_n4=kwargs.get("skip_n4", False))
        results.append(r)
    return results


def run_ip2(subjects, kwargs):
    from scripts.ip2_warp import warp_subject_candidate
    candidates = kwargs.get("warp_candidates", list(WARP_CANDIDATES.keys()))
    results = []
    for subj in subjects:
        for cand in candidates:
            r = warp_subject_candidate(subj, cand)
            results.append(r)
    return results


def run_ip3(subjects, kwargs):
    from scripts.ip3_propagate import propagate_subject_candidate
    candidates    = kwargs.get("warp_candidates",   list(WARP_CANDIDATES.keys()))
    label_filter  = kwargs.get("label_filter",      None)
    results = []
    for subj in subjects:
        for cand in candidates:
            r = propagate_subject_candidate(subj, cand, label_filter)
            results.append(r)
    return results


def run_ip4(subjects, kwargs):
    from scripts.ip4_refine import refine_subject
    warps   = kwargs.get("warp_candidates",   list(WARP_CANDIDATES.keys()))
    refines = kwargs.get("refine_candidates", list(REFINE_CANDIDATES.keys()))
    results = []
    for subj in subjects:
        for w in warps:
            for r in refines:
                rep = refine_subject(subj, w, r)
                results.append(rep)
    return results


def run_compare(subjects, kwargs):
    from scripts.compare_candidates import (
        compare_warp_candidates, compare_propagate_labels,
        compare_refine_candidates, build_pipeline_ranking,
        build_html_report, _save_csv,
    )
    import pandas as pd

    df_warp = compare_warp_candidates(subjects)
    df_prop = compare_propagate_labels(subjects)
    df_ref  = compare_refine_candidates(subjects)
    df_rank = build_pipeline_ranking(df_warp, df_prop, df_ref)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _save_csv(df_warp, os.path.join(OUTPUT_DIR, "comparison_warp_candidates.csv"))
    _save_csv(df_prop, os.path.join(OUTPUT_DIR, "comparison_propagate_labels.csv"))
    _save_csv(df_ref,  os.path.join(OUTPUT_DIR, "comparison_refine_candidates.csv"))
    _save_csv(df_rank, os.path.join(OUTPUT_DIR, "pipeline_ranking_final.csv"))
    build_html_report(
        df_warp, df_prop, df_ref, df_rank,
        os.path.join(OUTPUT_DIR, "QC_report.html"),
    )
    return [{"status": "SUCCESS", "step": "compare"}]


def run_ip5(subjects, kwargs):
    from scripts.ip5_freesurfer_gt import (
        check_freesurfer,
        run_recon_all,
        run_thalamic_segmentation,
        extract_labels_from_segmentation,
        compare_subject,
        save_comparison_report,
    )
    from scripts.utils.reporter import StepReporter

    fs = check_freesurfer()
    if not fs["ok"]:
        print("\n  ✗ FreeSurfer kurulu değil veya yapılandırılmamış:")
        for issue in fs["issues"]:
            print(f"    - {issue}")
        return [{"status": "FAILED", "step": "ip5", "reason": "freesurfer_not_available"}]

    skip_recon = kwargs.get("skip_recon", False)
    use_t2     = not kwargs.get("no_t2", False)
    warp_id    = kwargs.get("warp_id", "W2")
    all_rows   = []
    results    = []

    for subj in subjects:
        rep = StepReporter(
            subj, "IP5_FreeSurfer",
            os.path.join(OUTPUT_DIR, subj, "freesurfer_gt")
        )
        try:
            subj_dir  = run_recon_all(subj, skip=skip_recon)
            seg_mgz   = run_thalamic_segmentation(subj, subj_dir, use_t2=use_t2)
            out_dir   = os.path.join(OUTPUT_DIR, subj, "freesurfer_gt")
            fs_labels = extract_labels_from_segmentation(seg_mgz, out_dir)
            rows      = compare_subject(subj, fs_labels, warp_id)
            all_rows.extend(rows)
            rep.log("fs_labels_extracted", len(fs_labels))
            rep.log("comparisons_made",    len(rows))
            results.append(rep.finish("SUCCESS"))
        except Exception as exc:
            rep.record_exception(exc)
            results.append(rep.finish("FAILED"))

    if all_rows:
        save_comparison_report(all_rows)

    return results


STEP_RUNNERS = {
    "ip1":     run_ip1,
    "ip2":     run_ip2,
    "ip3":     run_ip3,
    "ip4":     run_ip4,
    "compare": run_compare,
    "ip5":     run_ip5,
}


# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BrainSeg Master Pipeline Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--steps", nargs="+", default=STEP_ORDER,
        choices=STEP_ORDER,
        help=f"Çalıştırılacak adımlar. Varsayılan: {STEP_ORDER}",
    )
    parser.add_argument("--subject", type=str, default=None,
                        help="Tek hasta ID; belirtilmezse tüm hastalar.")
    parser.add_argument("--warp",    type=str, default=None,
                        choices=list(WARP_CANDIDATES.keys()),
                        help="Tek warp adayı.")
    parser.add_argument("--refine",  type=str, default=None,
                        choices=list(REFINE_CANDIDATES.keys()),
                        help="Tek refine adayı.")
    parser.add_argument("--skip-n4", action="store_true",
                        help="N4 bias correction atla (hızlı test).")
    parser.add_argument("--labels",  type=str, default=None,
                        help="Virgülle ayrılmış label listesi: STh,RN,Hb")
    parser.add_argument("--skip-recon", action="store_true",
                        help="İP-5: recon-all atla (zaten tamamlandıysa).")
    parser.add_argument("--no-t2",  action="store_true",
                        help="İP-5: T2 kullanma, yalnızca T1 ile çalış.")
    parser.add_argument("--warp-id", type=str, default="W2",
                        help="İP-5: karşılaştırılacak warp adayı (varsayılan: W2).")
    args = parser.parse_args()

    subjects = [args.subject] if args.subject else SUBJECTS
    kwargs = {
        "skip_n4":           args.skip_n4,
        "warp_candidates":   [args.warp]   if args.warp   else list(WARP_CANDIDATES.keys()),
        "refine_candidates": [args.refine] if args.refine else list(REFINE_CANDIDATES.keys()),
        "label_filter":      args.labels.split(",") if args.labels else None,
        "skip_recon":        args.skip_recon,
        "no_t2":             args.no_t2,
        "warp_id":           args.warp_id,
    }

    pipeline_start = time.time()
    print(f"\n{'#'*60}")
    print(f"  BrainSeg Pipeline Başlıyor")
    print(f"  Tarih     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Adımlar   : {args.steps}")
    print(f"  Hastalar  : {subjects}")
    print(f"{'#'*60}\n")

    summary = {}

    for step in args.steps:
        if step not in STEP_RUNNERS:
            print(f"  ⚠  Bilinmeyen adım: {step} — atlandı")
            continue

        print(f"\n{'-'*60}")
        print(f"  ADIM: {step.upper()}")
        print(f"{'-'*60}")

        t0 = time.time()
        try:
            results = STEP_RUNNERS[step](subjects, kwargs)
        except Exception as exc:
            print(f"\n  ✗ {step} HATA: {exc}")
            summary[step] = {"status": "FAILED", "error": str(exc)}
            continue

        elapsed = round(time.time() - t0, 1)
        ok   = sum(1 for r in results if r.get("status") in ("SUCCESS", "SKIPPED"))
        fail = len(results) - ok
        summary[step] = {
            "status":   "SUCCESS" if fail == 0 else "PARTIAL",
            "ok":       ok,
            "fail":     fail,
            "duration": elapsed,
        }
        print(f"\n  {step.upper()} → ✓ {ok} / ✗ {fail}  ({elapsed}s)")

    # Genel Ozet
    total = round(time.time() - pipeline_start, 1)
    print(f"\n{'='*60}")
    print(f"  Pipeline Tamamlandi -- Toplam sure: {total}s")
    print(f"{'='*60}")
    for step, info in summary.items():
        icon = "OK" if info["status"] == "SUCCESS" else ("~~" if info["status"] == "PARTIAL" else "XX")
        detail = (f"ok={info.get('ok')}, fail={info.get('fail')}, "
                  f"sn={info.get('duration', '-')}")
        print(f"  [{icon}]  {step:<10} {detail}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
