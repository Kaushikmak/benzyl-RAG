import sys
import time
import threading
import logging
import warnings
import os


class AnimatedProgress:
    """Smooth animated progress bar with cycling dots (. .. ...) and log suppression."""

    def __init__(self, label: str = "Working", start_pct: int = 10, target_pct: int = 95, bar_len: int = 32):
        self.label = label
        self.current_pct = float(start_pct)
        self.target_pct = float(target_pct)
        self.bar_len = bar_len
        self._stop_event = threading.Event()
        self._thread = None
        self._old_stderr = None
        self._devnull = None
        self._disabled_logging_level = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.finish(f"{self.label} Complete", 100)
        else:
            self.stop()

    def start(self):
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        warnings.filterwarnings("ignore")

        for log_name in ("httpx", "urllib3", "transformers", "sentence_transformers", "chromadb", "qdrant_client", "app.rag"):
            logging.getLogger(log_name).setLevel(logging.ERROR)
        self._disabled_logging_level = logging.root.manager.disable
        logging.disable(logging.CRITICAL)

        self._old_stderr = sys.stderr
        self._devnull = open(os.devnull, "w")
        sys.stderr = self._devnull

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def set_label(self, label: str, target_pct: int | None = None):
        self.label = label
        if target_pct is not None:
            self.target_pct = float(target_pct)

    def _animate(self):
        dots_cycle = [".  ", ".. ", "...", ".  "]
        dot_idx = 0
        while not self._stop_event.is_set():
            # Smooth asymptotic progress toward target_pct
            if self.current_pct < self.target_pct:
                step = max(0.5, (self.target_pct - self.current_pct) * 0.15)
                self.current_pct = min(self.target_pct, self.current_pct + step)

            pct_int = int(self.current_pct)
            filled = int(self.bar_len * pct_int / 100)
            bar = "█" * filled + "░" * (self.bar_len - filled)
            dots = dots_cycle[dot_idx % len(dots_cycle)]
            dot_idx += 1

            line = f"\r[{bar}] {pct_int:3d}% | {self.label}{dots}   "
            sys.stdout.write(line)
            sys.stdout.flush()
            time.sleep(0.2)

    def finish(self, final_label: str = "Ready", final_pct: int = 100):
        self.stop()
        filled = self.bar_len
        bar = "█" * filled
        sys.stdout.write(f"\r[{bar}] {final_pct:3d}% | {final_label}                 \n")
        sys.stdout.flush()

    def stop(self):
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=1.0)
        if self._devnull:
            sys.stderr = self._old_stderr
            self._devnull.close()
            self._devnull = None
        if self._disabled_logging_level is not None:
            logging.disable(self._disabled_logging_level)
