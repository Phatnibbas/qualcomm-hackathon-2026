"""The full Colab notebook is syntactically valid without executing training."""

from __future__ import annotations

import ast
import unittest

from notebooks.build_full_pipeline_colab import build_notebook


class TestFullNotebook(unittest.TestCase):
    def test_every_code_cell_compiles_and_notebook_declares_full_arms(self):
        notebook = build_notebook()
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertIn("satellite-only", source)
        self.assertIn("station-only", source)
        self.assertIn("fused", source)
        self.assertIn("satpy", source.lower())
        self.assertIn("full_runtime.py", source)
        self.assertIn("full_dashboard.py", source)
        self.assertIn("runtime_source", source.lower())
        self.assertIn('LAG_SENSITIVITY_MINUTES = (0, 10, 20, 30)', source)
        self.assertIn('if completed + pd.Timedelta(minutes=lag) < issue:', source)
        self.assertIn('pd.date_range(first_slot, last_slot, freq="30min", tz="UTC")', source)
        self.assertIn('P0_BOUNDARIES', source)
        self.assertNotIn('window_start_utc.dt.date == subset.issue_time_utc.dt.date', source)
        self.assertIn('block.index.to_series().diff().iloc[1:].eq(pd.Timedelta(minutes=5)).all()', source)
        self.assertNotIn('np.diff(block.index.asi8)', source)
        self.assertIn('HISTORICAL_ROI_KM = 25', source)
        self.assertIn('HISTORICAL_SEGMENTS = (4,)', source)
        self.assertIn('ThreadPoolExecutor(max_workers=8)', source)
        self.assertIn('pip_install_satellite_only', source)
        self.assertNotIn('pip_install("numpy"', source)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))


if __name__ == "__main__":
    unittest.main()
