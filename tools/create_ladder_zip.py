"""Builds the zip to upload to AI Arena.

    python -m tools.create_ladder_zip [--out publish/bot.zip]

python-sc2 is copied from the installed package (the venv's pinned version) instead of living in
the repo, so the ladder runs the exact version the bot was developed against.
"""

import argparse
import subprocess
import zipfile
from importlib.metadata import version
from pathlib import Path

import sc2

REPO_ROOT = Path(__file__).resolve().parents[1]
AI_ARENA_UPLOAD_LIMIT_BYTES = 50 * 1024 * 1024


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()


def zip_entries() -> list[tuple[Path, str]]:
    """(source file, path inside the zip) for everything the ladder needs, all at the zip root."""
    folders = {
        REPO_ROOT / "bot": "bot",
        Path(sc2.__file__).parent: "sc2",
    }
    entries = [
        (REPO_ROOT / "tools" / "ladder_run.py", "run.py"),
        (REPO_ROOT / "config.py", "config.py"),
    ]
    for source_folder, zip_folder in folders.items():
        for source in sorted(source_folder.rglob("*")):
            if source.is_file() and "__pycache__" not in source.parts:
                entries.append((source, f"{zip_folder}/{source.relative_to(source_folder).as_posix()}"))
    return entries


def create_ladder_zip(out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for source, name in zip_entries():
            archive.write(source, name)

    size = out.stat().st_size
    if size > AI_ARENA_UPLOAD_LIMIT_BYTES:
        raise SystemExit(f"{out} is {size / 1024 / 1024:.1f} MB, over AI Arena's 50 MB upload limit")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="publish/bot.zip")
    args = parser.parse_args()

    out = create_ladder_zip(REPO_ROOT / args.out)
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    print(f"Created {out} ({out.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"python-sc2 {version('burnysc2')} from {Path(sc2.__file__).parent}")
    print(f"bot at commit {git('rev-parse', '--short', 'HEAD')}"
          + (" plus uncommitted changes" if dirty else ""))


if __name__ == "__main__":
    main()
