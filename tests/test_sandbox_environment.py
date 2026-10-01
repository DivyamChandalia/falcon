from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from falcon.sandbox_runtime import prepare_python_environment


class SandboxEnvironmentTests(unittest.TestCase):
    def test_overlay_inherits_packages_and_supports_local_pip_installs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(source)], check=True, capture_output=True)
            query = "import sysconfig; print(sysconfig.get_path('purelib'))"
            packages = Path(subprocess.check_output([str(source / "bin/python"), "-c", query], text=True).strip())
            (packages / "inherited_probe.py").write_text("VALUE = 'base'\n")
            workspace = root / "work"
            workspace.mkdir()
            target = workspace / "environment"
            spec = {"workspace": str(workspace), "python_environment": {"source": str(source), "path": str(target)}}
            prepare_python_environment(spec)
            wheel = workspace / "overlay_probe-1.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as bundle:
                bundle.writestr("overlay_probe/__init__.py", "VALUE = 'installed'\n")
                bundle.writestr("overlay_probe-1.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: overlay-probe\nVersion: 1.0\n")
                bundle.writestr("overlay_probe-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
                bundle.writestr("overlay_probe-1.0.dist-info/RECORD", "")
            python = str(target / "bin/python")
            subprocess.run([python, "-m", "pip", "install", "--no-index", "--no-deps", "--no-cache-dir", str(wheel)], check=True, capture_output=True)
            result = json.loads(subprocess.check_output([python, "-c", "import json,sys,inherited_probe,overlay_probe; print(json.dumps([sys.prefix,inherited_probe.VALUE,overlay_probe.VALUE]))"], text=True))
            self.assertEqual(result, [str(target), "base", "installed"])
            self.assertFalse((packages / "overlay_probe").exists())
            prepare_python_environment(spec)  # Pod replacement must retain installs.
            self.assertTrue((target / ".falcon-environment.json").exists())
            marker = target / ".falcon-environment.json"
            marker.write_text(json.dumps({"source": "/different-base"}))
            with self.assertRaisesRegex(ValueError, "another base"):
                prepare_python_environment(spec)

    def test_overlay_cannot_escape_workspace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            work.mkdir()
            with self.assertRaisesRegex(ValueError, "escapes workspace"):
                prepare_python_environment({"workspace": str(work), "python_environment": {"source": sys.prefix, "path": str(root / "outside")}})

    def test_image_only_mode_needs_no_overlay(self):
        prepare_python_environment({})
