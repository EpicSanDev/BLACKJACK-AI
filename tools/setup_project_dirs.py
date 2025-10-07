from __future__ import annotations

import os
from pathlib import Path

# Assuming the script is run from the project root directory
ROOT_DIR = Path.cwd()

# List of directories to create for prerequisite assets.
# Other directories for generated data will be created by the pipeline scripts.
DIRS_TO_CREATE = [
    ROOT_DIR / "dataset" / "png",
    ROOT_DIR / "dataset" / "backgrounds",
]

def main():
    """Creates the directories defined in DIRS_TO_CREATE."""
    print("Creating missing asset directories...")
    for d in DIRS_TO_CREATE:
        try:
            d.mkdir(parents=True, exist_ok=True)
            print(f"  - Ensured directory exists: {d}")
        except OSError as e:
            print(f"Error creating directory {d}: {e}")
    print("Done.")

if __name__ == "__main__":
    main()
