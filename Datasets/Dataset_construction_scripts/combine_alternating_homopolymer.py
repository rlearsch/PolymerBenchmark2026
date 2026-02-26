import pandas as pd
import argparse
import sys
from pathlib import Path
import re
import glob

def main():
    ap = argparse.ArgumentParser()#description="Recursively transform CSVs with pandas.")
    ap.add_argument("input_dir", type=Path, help="Directory to crawl for CSVs")
    args = ap.parse_args()
    in_root = args.input_dir.resolve()

    input_string = str(in_root)
    csvs = glob.glob(f'{input_string}/*.csv')

    #print(f'csvs: {csvs}')
    dfs = [pd.read_csv(csv) for csv in csvs if csv.split('/')[-1].split('_')[0] in {"alternating", "homopolymer"}]
    combined_df = pd.concat(dfs)

    m = re.match(r".*/(?:wPSMILES|PSMILES)/([^/]+/[^/]+/)([^/_]+)(?:_[^/]*)?\.csv$", csvs[0])
    if m:
        two_dirs, token = m.groups()
        SUBPATH = Path(two_dirs)
        out_path = SUBPATH
        #out_path.mkdir(parents=True, exist_ok=True)

    #if not input_file.exists(): #or not input_file.is_dir():
    #    print(f"ERROR: Input directory not found: {input_string}", file=sys.stderr)
    #    sys.exit(1)
    quantity = two_dirs.split('/')[-2]
    combined_df.to_csv(f'{str(input_string)}/homopolymer_alternating_{quantity}.csv',index=False)
    #print(f'{str(input_string)}/homopolymer_alternating_{quantity}.csv')
    #save_datasets(datasets, SUBPATH)

if __name__ == "__main__":
    main()
