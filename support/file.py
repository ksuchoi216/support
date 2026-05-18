import json
import os
import pickle
from ast import Set
from typing import Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from networkx import dfs_labeled_edges
from nibabel import test
from sqlalchemy.sql import table


def load_file(path) -> Any:
    extension = path.split(".")[-1]
    try:
        if extension == "txt":
            with open(path, "r", encoding="utf-8") as f:
                loaded_file = f.read()
        elif extension == "csv":
            with open(path, "r", encoding="utf-8-sig") as f:
                loaded_file = pd.read_csv(f, encoding="utf-8")
        elif extension == "json":
            with open(path, "r", encoding="utf-8-sig") as f:
                loaded_file = json.load(f)
        elif extension == "yaml":
            with open(path, "r", encoding="utf-8") as f:
                loaded_file = yaml.safe_load(f)
        elif extension == "pkl":
            with open(path, "rb") as f:
                loaded_file = pickle.load(f)
        else:
            raise ValueError(
                f"Unsupported file extension: {extension}",
            )
        logger.info(f"Loaded file from: {path}")
        return loaded_file
    except Exception as e:
        raise IOError(
            f"Error loading file: {e}",
        ) from e


def save_file(data: Any, path: str):
    parent_dir = os.path.dirname(path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    sub_folder = os.path.basename(path)
    extension = sub_folder.split(".")[-1]
    try:
        if extension == "txt":
            with open(path, "w", encoding="utf-8-sig") as f:
                f.write(data)
        elif extension == "csv":
            if isinstance(data, pd.DataFrame):
                with open(path, "w", encoding="utf-8-sig") as f:
                    data.to_csv(f, index=False, encoding="utf-8-sig")
            else:
                raise ValueError(
                    "Data must be a pandas DataFrame for CSV format.",
                )
        elif extension == "json":
            with open(path, "w", encoding="utf-8-sig") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        elif extension == "yaml":
            with open(path, "w", encoding="utf-8-sig") as f:
                yaml.dump(data, f, allow_unicode=True, sort_keys=False)
        elif extension == "pkl":
            with open(path, "wb") as f:
                pickle.dump(data, f)
        elif extension in ["npy", "npz"]:
            np.save(path, data)
        else:
            raise ValueError(
                f"Unsupported file extension: {extension}",
            )
        logger.info(f"Saved file to: {path}")
    except Exception as e:
        logger.exception("Error saving file to %s", path)
