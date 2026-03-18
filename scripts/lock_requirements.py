"""Generate a locked requirements file with pinned versions.

Creates requirements-lock.txt with:
- All installed packages with exact versions
- Generation timestamp and git commit
- Python version

Usage:
    python scripts/lock_requirements.py --output requirements-lock.txt
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_git_info() -> dict:
    """Get current git commit and branch info."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return {"commit": commit, "branch": branch}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "branch": "unknown"}


def get_installed_packages() -> list[str]:
    """Get list of installed packages with versions via pip freeze."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(result.stdout.strip().split("\n"))


def main():
    parser = argparse.ArgumentParser(description="Generate locked requirements file")
    parser.add_argument("--output", type=Path, default=Path("requirements-lock.txt"),
                        help="Output path for requirements-lock.txt")
    args = parser.parse_args()

    git_info = get_git_info()
    packages = get_installed_packages()

    lines = [
        "# Locked requirements for blue-frogs",
        f"# Generated: {datetime.utcnow().isoformat()}Z",
        f"# Git commit: {git_info['commit']}",
        f"# Git branch: {git_info['branch']}",
        f"# Python: {sys.version.split()[0]}",
        "#",
        "# To recreate this environment:",
        "#   pip install -r requirements-lock.txt",
        "#",
        "",
    ]
    lines.extend(packages)

    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Locked {len(packages)} packages to {args.output}")


if __name__ == "__main__":
    main()
