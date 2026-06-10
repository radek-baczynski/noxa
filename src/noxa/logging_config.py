from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    fmt = "%(asctime)s %(levelname)-5s [%(name)s] %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    # Quiet noisy third-party loggers unless debugging
    for name in ("httpx", "httpcore", "filelock", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)
