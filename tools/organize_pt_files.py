import os
import shutil
from pathlib import Path
from typing import Optional
from tqdm import tqdm


def organize_pt_files(source_dir: str, dest_dir: str) -> None:
    """
    Organize .pt files into subfolders based on their prefix (part before the first dash).
    Files not matching the format or extension remain in the top level.

    Args:
        source_dir (str): Path to the source directory containing .pt files
        dest_dir (Optional[str]): Path to the destination directory. If None, creates a 'organized' directory
                                 next to the source directory.
    """
    # Convert to Path objects
    source_path = Path(source_dir).resolve()

    # Set up destination directory
    dest_path = Path(dest_dir).resolve()

    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)

    # Get list of all files to process
    files_to_process = [f for f in source_path.iterdir() if f.is_file()]
    print(f"Found {len(files_to_process)} files to process")

    # Process all files in source directory with progress bar
    for file_path in tqdm(files_to_process, desc="Organizing files", unit="file"):
        # Get filename and extension
        filename = file_path.name
        ext = file_path.suffix

        # Skip if not a .pt file
        if ext != ".pt":
            # Copy non-.pt files directly to destination
            shutil.copy2(file_path, dest_path / filename)
            continue

        # Split filename by first dash
        parts = filename.split("-", 1)
        if len(parts) != 2:
            # If no dash found, copy to top level
            shutil.copy2(file_path, dest_path / filename)
            continue

        prefix = parts[0]

        # Create subdirectory for this prefix
        subdir = dest_path / prefix
        subdir.mkdir(exist_ok=True)

        # Copy file to subdirectory
        shutil.copy2(file_path, subdir / filename)

    print(f"Organization complete. Files organized into {dest_path}")


if __name__ == "__main__":
    DBFS = False
    if DBFS:
        organize_pt_files(
            source_dir="/dbfs/RAW/W00001_Data_Unrestricted/Andrejs/birdclef-2025/cache/",
            dest_dir="/dbfs/RAW/W00001_Data_Unrestricted/Andrejs/birdclef-2025/cache_structured/",
        )
    else:
        organize_pt_files(
            source_dir="data/birdclef-2025/cache/",
            dest_dir="data/birdclef-2025/cache_structured",
        )
