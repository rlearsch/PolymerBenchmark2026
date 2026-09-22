"""Create or update the cached polyBERT embeddings for generated PSMILES data."""

import argparse
import os
import pickle
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).parent
DEFAULT_MODEL_PATH = SCRIPT_DIR / "../../../polyBERT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(os.environ.get("POLYBERT_MODEL_PATH", DEFAULT_MODEL_PATH)),
        help="Local SentenceTransformer model directory; defaults to "
        "POLYBERT_MODEL_PATH or ../../../polyBERT.",
    )
    return parser.parse_args()


def is_csv_like(path: Path) -> bool:
    return path.name.lower().endswith(".csv")


def main() -> None:
    args = parse_args()
    model_path = args.model_path.resolve()
    if not model_path.is_dir():
        raise SystemExit(f"ERROR: polyBERT model not found: {model_path}")

    # Keep --help usable without installing the optional polyBERT stack.
    from sentence_transformers import SentenceTransformer

    dictionary_path = SCRIPT_DIR / "files/PSMILES_pBERT_dict.pkl"
    if dictionary_path.exists():
        with dictionary_path.open("rb") as handle:
            embeddings = pickle.load(handle)
        print(f"Initial dictionary loaded with {len(embeddings)} entries")
    else:
        embeddings = {}
        print("No existing dictionary found. Creating new dictionary from scratch.")

    model = SentenceTransformer(str(model_path))
    input_root = SCRIPT_DIR / "../../Datasets/PSMILES/"
    datasets = [path for path in input_root.rglob("*") if path.is_file() and is_csv_like(path)]
    added = 0

    def save_dictionary() -> None:
        with dictionary_path.open("wb") as handle:
            pickle.dump(embeddings, handle)

    for dataset in datasets:
        try:
            smiles_list = pd.read_csv(dataset).smiles.tolist()
        except Exception as error:
            print(f"Error processing {dataset}: {error}")
            continue

        for smiles in smiles_list:
            if smiles in embeddings:
                continue
            try:
                embeddings[smiles] = model.encode(smiles)
                added += 1
            except Exception as error:
                print(f"Error processing {smiles}: {error}")
            if added and added % 1000 == 0:
                save_dictionary()
                print(f"Dictionary saved with {len(embeddings)} entries")

        save_dictionary()
        print(f"Dictionary saved with {len(embeddings)} entries")


if __name__ == "__main__":
    main()
