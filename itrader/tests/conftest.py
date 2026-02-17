import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure absolute imports like `import itrader...` work regardless of pytest rootdir.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def tiny_ohlcv_path(tmp_path: Path) -> Path:
    n = 500
    base = np.linspace(100.0, 120.0, n, dtype=np.float64)
    noise = np.sin(np.arange(n) / 10.0) * 0.1
    close = base + noise
    df = pd.DataFrame(
        {
            "OPEN": close - 0.05,
            "HIGH": close + 0.15,
            "LOW": close - 0.15,
            "CLOSE": close,
            "VOLUME": np.linspace(1000.0, 5000.0, n, dtype=np.float64),
        }
    )
    out = tmp_path / "tiny_ohlcv.parquet"
    df.to_parquet(out, index=False)
    return out
