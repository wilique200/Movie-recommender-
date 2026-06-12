# utils.py
import pandas as pd
import zipfile
import os

def load_movies_from_zip(zip_path_or_file, csv_ext=".csv"):
    """
    Accepts a path to a zip file or a file-like object and returns a concatenated DataFrame.
    """
    dfs = []
    with zipfile.ZipFile(zip_path_or_file, "r") as z:
        files = [f for f in z.namelist() if f.lower().endswith(csv_ext)]
        for f in files:
            try:
                df = pd.read_csv(z.open(f))
                df["source_file"] = os.path.basename(f)
                dfs.append(df)
            except Exception:
                continue
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)
