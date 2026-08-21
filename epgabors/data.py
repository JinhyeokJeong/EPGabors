"""Dataset and condition utilities for EP Gabor images."""

from __future__ import annotations

import os
import re
from typing import Dict, Iterable, Optional

import pandas as pd
import tifffile
import torch
from torch.utils.data import Dataset


EPGABOR_PATTERN = re.compile(
    r"^GaborArray_m(?P<mean>-?\d+)_sd(?P<sd>\d+)_ss(?P<ss>\d+)_(?P<instance>\d+)\.tiff$"
)


class EPGabors(Dataset):
    """Torch dataset for EP Gabor images with metadata-aware filtering.

    Expected filename format:
        GaborArray_m{mean}_sd{sd}_ss{ss}_{instance}.tiff
    """

    def __init__(
        self,
        img_dir: str,
        include_vertical: bool = True,
        include_single: bool = False,
        include_zerovar: bool = True,
        filter_ss: Optional[Iterable[int]] = None,
        filter_var: Optional[Iterable[int]] = None,
        filter_mean: Optional[Iterable[int]] = None,
        filter_sd: Optional[Iterable[int]] = None,
        filter_instance: Optional[Iterable[int]] = None,
        transform=None,
        return_filename: bool = False,
        return_condition_id: bool = False,
        sort_by: Optional[Iterable[str]] = ("mean", "sd", "ss", "instance", "file_name"),
        validate_files: bool = False,
    ) -> None:
        self.img_dir = img_dir
        self.include_vertical = include_vertical
        self.include_single = include_single
        self.include_zerovar = include_zerovar
        self.transform = transform
        self.return_filename = return_filename
        self.return_condition_id = return_condition_id

        if filter_var is not None and filter_sd is not None:
            raise ValueError("Use either filter_var or filter_sd, not both.")
        if filter_sd is None:
            filter_sd = filter_var

        self.df = self._parse_metadata()
        self.df = self._apply_filters(
            self.df,
            filter_mean=filter_mean,
            filter_sd=filter_sd,
            filter_ss=filter_ss,
            filter_instance=filter_instance,
        )

        if self.df.empty:
            raise ValueError("No EPGabor files remain after filtering.")

        if sort_by is not None:
            self.df = self.df.sort_values(list(sort_by)).reset_index(drop=True)
        else:
            self.df = self.df.reset_index(drop=True)

        if validate_files:
            validate_condition_grid(self.df)

    def _parse_metadata(self) -> pd.DataFrame:
        records = []
        for fname in os.listdir(self.img_dir):
            match = EPGABOR_PATTERN.match(fname)
            if match is None:
                continue
            records.append(
                {
                    "file_name": fname,
                    "mean": int(match.group("mean")),
                    "sd": int(match.group("sd")),
                    "ss": int(match.group("ss")),
                    "instance": int(match.group("instance")),
                }
            )

        if not records:
            raise ValueError(f"No valid EPGabor files were found in: {self.img_dir}")

        return pd.DataFrame.from_records(records)

    def _apply_filters(
        self,
        df: pd.DataFrame,
        filter_mean: Optional[Iterable[int]],
        filter_sd: Optional[Iterable[int]],
        filter_ss: Optional[Iterable[int]],
        filter_instance: Optional[Iterable[int]],
    ) -> pd.DataFrame:
        out = df

        if not self.include_vertical:
            out = out[out["mean"] != 0]

        if not self.include_single:
            out = out[out["ss"] != 1]

        if not self.include_zerovar:
            out = out[out["sd"] != 0]

        if filter_mean is not None:
            out = out[out["mean"].isin(list(filter_mean))]

        if filter_sd is not None:
            out = out[out["sd"].isin(list(filter_sd))]

        if filter_ss is not None:
            out = out[out["ss"].isin(list(filter_ss))]

        if filter_instance is not None:
            out = out[out["instance"].isin(list(filter_instance))]

        return out

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["file_name"])
        image = tifffile.imread(img_path)

        labels = {
            "mean": torch.tensor(int(row["mean"]), dtype=torch.long),
            "sd": torch.tensor(int(row["sd"]), dtype=torch.long),
            "ss": torch.tensor(int(row["ss"]), dtype=torch.long),
            "instance": torch.tensor(int(row["instance"]), dtype=torch.long),
        }

        if self.return_filename:
            labels["file_name"] = row["file_name"]
        if self.return_condition_id:
            labels["condition_id"] = condition_id_from_row(row)

        if self.transform is not None:
            image = self.transform(image)

        return image, labels


def condition_id_from_row(row: pd.Series) -> str:
    return f"m{int(row['mean'])}_sd{int(row['sd'])}_ss{int(row['ss'])}"


def summarize_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per mean/sd/ss condition with instance counts."""
    required = {"mean", "sd", "ss", "instance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Condition summary requires columns: {sorted(missing)}")

    summary = (
        df.groupby(["mean", "sd", "ss"], sort=True)
        .agg(
            n=("instance", "size"),
            instance_min=("instance", "min"),
            instance_max=("instance", "max"),
            unique_instances=("instance", "nunique"),
        )
        .reset_index()
    )
    summary["condition_id"] = summary.apply(condition_id_from_row, axis=1)
    return summary[
        ["condition_id", "mean", "sd", "ss", "n", "instance_min", "instance_max", "unique_instances"]
    ]


def validate_condition_grid(df: pd.DataFrame) -> Dict[str, object]:
    """Validate that each retained condition has a complete, duplicate-free instance set."""
    summary = summarize_conditions(df)
    duplicate_rows = int(df.duplicated(["mean", "sd", "ss", "instance"]).sum())
    incomplete = summary[summary["n"] != summary["unique_instances"]]
    expected_n = int(summary["n"].mode().iloc[0]) if not summary.empty else 0
    uneven = summary[summary["n"] != expected_n]

    if duplicate_rows:
        raise ValueError(f"Found {duplicate_rows} duplicate condition/instance rows.")
    if not incomplete.empty:
        raise ValueError("Found duplicated instance identifiers within at least one condition.")

    return {
        "n_images": int(len(df)),
        "n_conditions": int(len(summary)),
        "expected_instances_per_condition": expected_n,
        "is_balanced": bool(uneven.empty),
        "mean_levels": sorted(int(x) for x in df["mean"].unique()),
        "sd_levels": sorted(int(x) for x in df["sd"].unique()),
        "ss_levels": sorted(int(x) for x in df["ss"].unique()),
    }
