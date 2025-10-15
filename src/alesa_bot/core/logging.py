# ===================== FILE: src/alesa_bot/core/logging.py =====================
import logging


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=level)