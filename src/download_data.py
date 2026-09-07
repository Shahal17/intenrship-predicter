from pathlib import Path
import shutil

import kagglehub


DATASET_HANDLE = "sakharebharat/indian-student-placement-dataset-2025"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Kaggle dataset: {DATASET_HANDLE}")
    downloaded_path = Path(kagglehub.dataset_download(DATASET_HANDLE))

    csv_files = list(downloaded_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file was found in the downloaded dataset folder: {downloaded_path}"
        )

    for csv_file in csv_files:
        destination = DATA_DIR / csv_file.name
        shutil.copy2(csv_file, destination)
        print(f"Saved: {destination}")

    print("Dataset download complete.")


if __name__ == "__main__":
    main()
