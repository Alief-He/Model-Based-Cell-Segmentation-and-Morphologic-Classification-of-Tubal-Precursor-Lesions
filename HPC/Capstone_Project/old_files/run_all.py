from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "01_extract_epi_polygons_Huang3.py",
    "02_cellpose_in_polygons_hpc_Huang3.py",
    "02_cellpose_in_polygons_hpc_Huang3_MODIFIED.py",
    "03_make_contrast_panels.py",
    "04_color_cellvit_masks_by_type.py",
]


def run_script(script_name: str) -> None:
    script_path = PROJECT_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Cannot find script: {script_path}")

    print(f"\n=== Running {script_name} ===", flush=True)
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_DIR,
        check=True,
    )
    print(f"=== Finished {script_name} ===", flush=True)


def main() -> None:
    for script_name in SCRIPTS:
        run_script(script_name)

    print("\nAll scripts finished successfully.", flush=True)


if __name__ == "__main__":
    main()
