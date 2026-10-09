"""Measure real platform theme callbacks and subsequent Qt rendering.

Uses a temporary empty database; GPU probing and background refresh are disabled.
Example: python TrainingPlatform/tools/benchmark_theme_switch.py --native 
         --monkeyqt-source E:/path/to/MonkeyQt --mode fast --iterations 6
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monkeyqt-source", type=Path)
    parser.add_argument("--mode", choices=("fast", "full"), default="fast")
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--native", action="store_true", help="Use the native Qt backend, with no visible window")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    if args.monkeyqt_source:
        sys.path[:0] = [str(args.monkeyqt_source), str(args.monkeyqt_source / "monkeyqt-icons")]
    if not args.native:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import QApplication
    from monkeyqt import ThemeEngine
    from TrainingPlatform.db.database import DatabaseManager
    from TrainingPlatform.ui import main_window as mw
    from TrainingPlatform.ui.components.gpu_card import GPUCard

    class App(QApplication):
        qss_calls = 0

        def setStyleSheet(self, qss):
            self.qss_calls += 1
            super().setStyleSheet(qss)

    app = App([])
    ThemeEngine.enable_palette_theme(args.mode == "fast")
    ThemeEngine.set_theme("Elegant Light")
    results = {"mode": args.mode, "backend": app.platformName(), "scenarios": []}
    with tempfile.TemporaryDirectory(prefix="labelpaw-theme-") as temp_dir:
        db = DatabaseManager(str(Path(temp_dir) / "benchmark.db"))
        with patch.object(mw, "DatabaseManager", return_value=db), \
             patch.object(mw.DashboardView, "refresh_data", lambda *a, **k: None), \
             patch.object(GPUCard, "refresh_gpu_status", lambda *a, **k: None):
            window = mw.MainWindow()
            window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            window.resize(1350, 880)
            window.show()

            def settle():
                for timer in window.findChildren(QTimer):
                    timer.stop()
                # Flush deferred chart/layout work before measuring theme changes.
                deadline = time.perf_counter() + 0.4
                while time.perf_counter() < deadline:
                    app.processEvents()
                    time.sleep(0.005)
                window.grab()

            def measure(name):
                settle()
                ThemeEngine.set_theme("Dark Mode (OLED)")
                settle()
                ThemeEngine.set_theme("Elegant Light")
                settle()
                calls = app.qss_calls
                samples = []
                for index in range(args.iterations):
                    start = time.perf_counter()
                    ThemeEngine.set_theme("Dark Mode (OLED)" if index % 2 == 0 else "Elegant Light")
                    switched = time.perf_counter()
                    app.processEvents()
                    # Force widget painting, including Matplotlib's pending idle draw.
                    window.grab()
                    painted = time.perf_counter()
                    samples.append({"handler_ms": round((switched-start)*1000, 2),
                                    "render_ms": round((painted-switched)*1000, 2),
                                    "total_ms": round((painted-start)*1000, 2)})
                result = {"name": name, "widgets": len(app.allWidgets()),
                          "global_qss_resets": app.qss_calls-calls,
                          "median_handler_ms": statistics.median(s["handler_ms"] for s in samples),
                          "median_total_ms": statistics.median(s["total_ms"] for s in samples),
                          "samples": samples}
                results["scenarios"].append(result)

            measure("dashboard")
            project = window.view_project_detail
            detail = project._ensure_experiment_detail_view()
            measure("dashboard_with_hidden_detail")
            window.stacked_widget.setCurrentWidget(project)
            project.content_stack.setCurrentWidget(detail)
            measure("experiment_overview")
            detail.pages_stack.setCurrentWidget(detail.tab_train)
            measure("experiment_charts")
            from TrainingPlatform.ui.components.thesis_chart import ThesisChartCard
            series = {"benchmark": {"epochs": list(range(1, 81)),
                                     "values": [0.2 + i * 0.008 for i in range(80)]}}
            for chart in detail.tab_train.findChildren(ThesisChartCard):
                chart.update_series(series)
            measure("experiment_charts_with_data")
            window.close()
            window.deleteLater()
            app.processEvents()
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
