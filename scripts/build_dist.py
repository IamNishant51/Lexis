"""Build clean Lexis distributions (sdist + wheel).

Verifies the build system, purges leftover artifacts, then runs
`python -m build`. Usage:  python scripts/build_dist.py [--no-isolation]
"""

from __future__ import annotations

import shutil
import subprocess  # noqa: S404
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIN_PY = (3, 10)


def _run(cmd: list[str], cwd: Path = ROOT) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)  # noqa: S603


def _verify() -> None:
    if sys.version_info < MIN_PY:
        need = ".".join(map(str, MIN_PY))
        sys.exit(f"build requires Python >={need} (have {sys.version.split()[0]})")
    for name in ("pyproject.toml", "README.md"):
        if not (ROOT / name).exists():
            sys.exit(f"missing required file: {name}")
    try:
        import build  # noqa: F401
    except ImportError:
        print("`build` not installed — installing it now.")
        _run([sys.executable, "-m", "pip", "install", "build>=1.0"])
    print(f"build system OK (python {sys.version.split()[0]})")


def _purge() -> None:
    for target in ("dist", "build"):
        path = ROOT / target
        if path.exists():
            shutil.rmtree(path)
            print(f"purged {target}/")
    for egg in ROOT.glob("*.egg-info"):
        shutil.rmtree(egg)
        print(f"purged {egg.name}/")


def _confirm(dist: Path) -> None:
    sdists = sorted(dist.glob("*.tar.gz"))
    wheels = sorted(dist.glob("*.whl"))
    if not sdists or not wheels:
        sys.exit("build incomplete: need one .tar.gz and one .whl in dist/")
    for artifact in (*sdists, *wheels):
        kb = artifact.stat().st_size / 1024
        print(f"  {artifact.name}  ({kb:.1f} KB)")
    with zipfile.ZipFile(wheels[-1]) as zf:
        names = zf.namelist()
    if not any(n.startswith("lexis_local/") and n.endswith("__init__.py") for n in names):
        sys.exit("wheel missing lexis_local package — check [tool.setuptools.packages.find]")
    entry = [n for n in names if n.endswith("entry_points.txt")]
    print(f"wheel OK: {len(names)} files, entry_points present: {bool(entry)}")


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    _verify()
    _purge()
    _run([sys.executable, "-m", "build", *args])
    _confirm(ROOT / "dist")
    print("dist/ ready — install with:  pip install dist/lexis_local-*.whl")


if __name__ == "__main__":
    main()
