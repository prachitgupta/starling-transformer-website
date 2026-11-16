import json
import pathlib
import re
import sys
import unittest
from urllib.parse import unquote


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from index_videos import discover  # noqa: E402


class SiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import build_site

        build_site.main()
        cls.html = (ROOT / "index.html").read_text()

    def test_case_b_gpt_scope_is_explicit(self):
        self.assertIn("Case B · conditional score · GPT selector only", self.html)
        self.assertIn("conditional residual score", self.html)
        self.assertNotRegex(self.html, re.compile(r"case\s*A", re.I))
        self.assertNotIn("heuristic", self.html.lower())

    def test_finalized_simulation_assets_are_present(self):
        expected = {
            "score_histogram_caseB_gpt.png",
            "scene_tube_caseB_gpt.png",
            "coverage_caseB_gpt.png",
            "tightness_caseB_gpt.png",
            "literature_comparison_caseB_gpt.png",
        }
        image_dir = ROOT / "assets/images/simulation"
        self.assertTrue(expected.issubset({path.name for path in image_dir.glob("*.png")}))
        for name in expected:
            self.assertIn(f"assets/images/simulation/{name}", self.html)

    def test_export_contains_only_case_b_gpt_certificate(self):
        data = json.loads((ROOT / "data/case_b_gpt.json").read_text())
        self.assertEqual(data["case"], "B")
        self.assertEqual(data["expert_selector"], "gpt")
        self.assertEqual(data["held_out_rows"], 2598)
        self.assertEqual(data["certificate"]["calibration_size"], 200)
        self.assertAlmostEqual(data["target_coverage"], 0.9)

    def test_hardware_placeholder_is_truthfully_labeled(self):
        self.assertIn("Synthetic placeholder · no measured hardware benchmark yet", self.html)
        self.assertIn("not a coverage experiment", self.html)

    def test_all_videos_are_routed_and_deployed(self):
        videos = discover()
        discovered = [item for items in videos.values() for item in items]
        local = [path for path in (ROOT / "videos").iterdir() if path.suffix.lower() == ".mp4"]
        self.assertEqual(len(discovered), len(local))
        self.assertFalse(videos["additional"])
        self.assertTrue(all((ROOT / "dist" / unquote(item["src"])).exists() for item in discovered))

    def test_video_files_fit_github_limits(self):
        for path in (ROOT / "videos").rglob("*"):
            if path.suffix.lower() in {".mp4", ".webm", ".ogv", ".m4v"}:
                self.assertLess(path.stat().st_size, 95 * 1024 * 1024)

    def test_no_unreplaced_template_markers(self):
        self.assertNotIn("<!-- CONTENT -->", self.html)
        self.assertTrue((ROOT / "dist/.nojekyll").exists())
        self.assertTrue((ROOT / "dist/assets/js/site.js").exists())


if __name__ == "__main__":
    unittest.main()
