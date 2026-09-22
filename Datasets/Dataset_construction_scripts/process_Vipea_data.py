import pandas as pd
from pathlib import Path

# Import centralized conversion functions
from polymer_conversions import wPSMILES_to_PSMILES_alternating

# Use paths relative to this script's location
SCRIPT_DIR = Path(__file__).parent


def process_vipea_data(
    script_dir: Path = SCRIPT_DIR, datasets_root: Path | None = None
) -> None:
    """Generate VIPEA EA/IP datasets in PSMILES and wPSMILES formats."""
    script_dir = Path(script_dir)
    if datasets_root is None:
        datasets_root = script_dir.parent
    else:
        datasets_root = Path(datasets_root)

    wpsmiles_data = pd.read_csv(
        script_dir / "files/polymer-chemprop-data/dataset-poly_chemprop.csv"
    )
    polymer_information = pd.read_csv(
        script_dir / "files/polymer-chemprop-data/dataset.csv"
    )

    df = wpsmiles_data.reset_index().merge(polymer_information.reset_index(), on="index")
    df["EA"] = df["EA (eV)"]
    df["IP"] = df["IP (eV)"]
    for architecture in ("alternating", "random", "block"):
        for quantity in ("EA", "IP"):
            df_temp = df[df.poly_type == architecture].reset_index(drop=True)
            df_temp["smiles"] = df_temp.poly_chemprop_input

            wpsmiles_path = (
                datasets_root
                / "wPSMILES"
                / "Vipea"
                / quantity
                / f"{architecture}_{quantity}.csv"
            )
            wpsmiles_path.parent.mkdir(parents=True, exist_ok=True)
            df_temp[["smiles", quantity]].to_csv(wpsmiles_path, index=False)

            if architecture == "alternating":
                df_temp["smiles"] = df_temp.smiles.apply(
                    wPSMILES_to_PSMILES_alternating
                )
                psmiles_path = (
                    datasets_root
                    / "PSMILES"
                    / "Vipea"
                    / quantity
                    / f"{architecture}_{quantity}.csv"
                )
                psmiles_path.parent.mkdir(parents=True, exist_ok=True)
                df_temp[["smiles", quantity]].to_csv(psmiles_path, index=False)


if __name__ == "__main__":
    process_vipea_data()
