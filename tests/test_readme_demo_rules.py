from pathlib import Path

import pytest

from analysis.demos.plot_jaxnav_readme_curve import load_bins
from analysis.demos.render_readme_comparison import file_sha256


def test_file_sha256_is_stable(tmp_path: Path):
    path = tmp_path / "checkpoint.pkl"
    path.write_bytes(b"fixed checkpoint")
    assert file_sha256(path) == "5186699cfb853e8396f1e4b38e612525c891ab07559d52a1bf80a87d013ffb91"


def test_curve_loader_uses_real_bins_without_filling_tails(tmp_path: Path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        '{"episode": 4900, "win_rate": 0.2}\n'
        '{"episode": 5100, "win_rate": 0.4}\n'
        '{"episode": 14900, "win_rate": 0.6}\n'
    )
    assert load_bins(path) == pytest.approx({5000: 0.3, 15000: 0.6})
