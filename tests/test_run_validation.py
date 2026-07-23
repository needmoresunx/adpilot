import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import check_latest_run


class RunValidationTests(unittest.TestCase):
    def _write_json(self, directory: Path, name: str, value) -> None:
        (directory / name).write_text(json.dumps(value), encoding="utf-8")

    def test_audited_validation_rejects_unresolved_identity_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "report.html").write_text("ok", encoding="utf-8")
            self._write_json(run_dir, "generation_backend.json", {"name": "flux_kontext", "used_fallback": False})
            self._write_json(run_dir, "video_backend.json", {"name": "wan_i2v", "used_fallback": False})
            self._write_json(run_dir, "planner_metadata.json", {"name": "qwen2_5_vl", "used_fallback": False})
            self._write_json(run_dir, "critique_report.json", [{"passed": False}])
            self._write_json(run_dir, "video_critique_report.json", [{"passed": True}])

            with patch("sys.argv", ["check_latest_run.py", "--run-dir", str(run_dir), "--require-audited"]):
                with redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        check_latest_run.main()

        self.assertEqual(error.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
