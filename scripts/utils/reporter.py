"""
StepReporter — Her pipeline adımı için anlık terminal çıktısı ve JSON raporu.

Kullanım:
    rep = StepReporter("IXI002-Guys-0828", "IP1_Preproc", "outputs/IXI002/preproc")
    rep.log("T1_orientation", "RAS")
    rep.log("jacobian_neg_ratio", 0.0008,
            warn_if=lambda x: x > 0.0005,
            fail_if=lambda x: x > 0.001)
    rep.add_file("T1_preproc", "outputs/IXI002/preproc/T1_preproc.nii.gz")
    rep.finish("SUCCESS")
"""

import json
import os
import time
from datetime import datetime

# Windows terminal encoding fix
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(_sys.stderr, "reconfigure"):
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")



class StepReporter:
    def __init__(self, subject_id: str, step_name: str, output_dir: str):
        self.subject_id = subject_id
        self.step_name  = step_name
        self.output_dir = output_dir
        self.start_time = time.time()
        self._failed    = False

        os.makedirs(output_dir, exist_ok=True)

        self.report = {
            "subject_id":   subject_id,
            "step":         step_name,
            "timestamp":    datetime.now().isoformat(timespec="seconds"),
            "status":       "RUNNING",
            "duration_sec": None,
            "metrics":      {},
            "qc_flags":     {},
            "warnings":     [],
            "errors":       [],
            "output_files": {},
        }
        print(f"\n[{step_name} | {subject_id}] Başlıyor...")

    # ── Metrik Kayıt ────────────────────────────────────────────────────────
    def log(self, key: str, value, warn_if=None, fail_if=None, unit: str = ""):
        """
        Bir metriği kaydet; isteğe bağlı warn/fail eşiği kontrol et.
        warn_if / fail_if → callable(value) → bool
        """
        self.report["metrics"][key] = value
        label = f"{value} {unit}".strip()

        failed = fail_if and fail_if(value)
        warned = warn_if and warn_if(value)

        if failed:
            self._failed = True
            self.report["qc_flags"][key + "_fail"] = True
            msg = f"HATA: {key} = {label} → ELENDİ"
            self.report["errors"].append(msg)
            print(f"  ✗ {msg}")
        elif warned:
            self.report["qc_flags"][key + "_warn"] = True
            msg = f"UYARI: {key} = {label}"
            self.report["warnings"].append(msg)
            print(f"  ⚠  {msg}")
        else:
            print(f"  → {key}: {label} ✓")

        return not failed

    def set_flag(self, flag: str, value: bool, reason: str = ""):
        """QC flag'ini doğrudan set et."""
        self.report["qc_flags"][flag] = value
        if value and reason:
            self.report["warnings"].append(f"{flag}: {reason}")
            print(f"  ⚑  {flag}: {reason}")

    def add_file(self, key: str, path: str):
        """Çıktı dosyasını rapora ekle (tam path)."""
        self.report["output_files"][key] = os.path.abspath(path)

    def add_metric_group(self, group_name: str, metrics: dict):
        """İç içe metrik grubu ekle (örn. her label için morphometrics)."""
        self.report["metrics"][group_name] = metrics

    # ── Bitirme ─────────────────────────────────────────────────────────────
    def finish(self, status: str = None) -> dict:
        """
        Raporu tamamla ve JSON'a yaz.
        status belirtilmezse: hata varsa FAILED, yoksa SUCCESS.
        """
        if status is None:
            status = "FAILED" if self._failed else "SUCCESS"

        self.report["status"]       = status
        self.report["duration_sec"] = round(time.time() - self.start_time, 1)

        report_path = os.path.join(
            self.output_dir, f"report_{self.step_name}.json"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)

        icon = "✓" if status == "SUCCESS" else "✗"
        print(
            f"[{self.step_name} | {self.subject_id}] {icon} {status} "
            f"({self.report['duration_sec']} sn) → {report_path}"
        )
        return self.report

    # ── Hata Yakalama Yardımcısı ─────────────────────────────────────────
    def record_exception(self, exc: Exception):
        """Exception'ı tam traceback ile rapora yaz; finish() çağrısı hâlâ gereklidir."""
        import traceback
        msg       = f"{type(exc).__name__}: {exc}"
        tb_lines  = traceback.format_exc().splitlines()
        tb_str    = "\n".join(tb_lines)
        self.report["errors"].append(msg)
        self.report["traceback"] = tb_str
        self._failed = True
        print(f"  ✗ {msg}")
        # Traceback'i stderr'e de yaz (pipeline debug için)
        import sys
        print(tb_str, file=sys.stderr)

    @property
    def has_failure(self) -> bool:
        return self._failed
