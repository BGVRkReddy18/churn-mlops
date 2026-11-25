"""Stage 1 - download the raw dataset."""

import logging

import httpx

from churn.config import RAW_DATA_PATH, load_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    url = load_params()["data"]["url"]
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s", url)
    response = httpx.get(url, timeout=60, follow_redirects=True)
    response.raise_for_status()
    RAW_DATA_PATH.write_bytes(response.content)
    log.info("Saved %d bytes to %s", len(response.content), RAW_DATA_PATH)


if __name__ == "__main__":
    main()
