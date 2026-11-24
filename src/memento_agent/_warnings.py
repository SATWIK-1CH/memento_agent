"""Filter import-time noise before urllib3/httpx are loaded."""

from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message="SOCKS support in urllib3 requires the installation of optional dependencies",
)
