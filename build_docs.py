#!/usr/bin/env python3
"""
Script to build documentation locally using Sphinx.
"""

import sys
import subprocess
from pathlib import Path


def build_docs():
    """Build the documentation."""
    repo_root = Path(__file__).resolve().parent
    docs_dir = repo_root / "docs"
    source_dir = docs_dir / "source"
    build_dir = docs_dir / "_build" / "html"

    print("Building HTML documentation...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "html",
            str(source_dir),
            str(build_dir),
        ],
        check=True,
    )

    html_path = build_dir / "index.html"
    print("Documentation built successfully!")
    print(f"Open: {html_path.absolute()}")

    return html_path

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        pass
    else:
        build_docs()
