import logging


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure basic logging for the package."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

