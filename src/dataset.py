"""
GeoGuessr StreetView dataset (4 views) for StreetCLIP.
FIX: ground truth provides `state` as text; we map it to `state_idx` using the provided mapping.
"""

import os, re
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from typing import Any, Dict, List, Tuple, Optional


# ----------------------------
# State mapping (text -> idx)
# ----------------------------
STATE_NAME_TO_IDX = {
    "Alabama": 0,
    "Alaska": 1,
    "Arkansas": 3,
    "California": 4,
    "Colorado": 5,
    "Connecticut": 6,
    "Delaware": 7,
    "Florida": 8,
    "Georgia": 9,
    "Hawaii": 10,
    "Idaho": 11,
    "Illinois": 12,
    "Indiana": 13,
    "Iowa": 14,
    "Kansas": 15,
    "Kentucky": 16,
    "Louisiana": 17,
    "Maine": 18,
    "Maryland": 19,
    "Massachusetts": 20,
    "Michigan": 21,
    "Minnesota": 22,
    "Nevada": 27,
    "South Dakota": 40,
    "Tennessee": 41,
    "Texas": 42,
    "Utah": 43,
    "Vermont": 44,
    "Virginia": 45,
    "Washington": 46,
    "West Virginia": 47,
    "Wisconsin": 48,
    "Wyoming": 49,
}

VALID_STATE_IDXS = sorted(STATE_NAME_TO_IDX.values())
NUM_STATES = 50  # head should output 0-49 indices (even if only 33 appear)

class GeoGuessrDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        csv_file: str,
        processor=None,              # e.g., CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
        transform=None,              # optional torchvision transform (ignored if processor is provided)
        is_test: bool = False,
        strict_files: bool = True,
    ):
        self.images_dir = images_dir
        self.processor = processor
        self.transform = transform
        self.is_test = is_test
        self.strict_files = strict_files

        self.df = pd.read_csv(csv_file)

        required_cols = ["sample_id", "image_north", "image_east", "image_south", "image_west"]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}. Found: {list(self.df.columns)}")

        self.df["sample_id"] = self.df["sample_id"].astype(int)

        if not self.is_test:
            # labels: state (text), latitude, longitude
            for c in ["state", "latitude", "longitude"]:
                if c not in self.df.columns:
                    raise ValueError(
                        f"Train CSV missing label column '{c}'. "
                        f"Expected text label column 'state' plus latitude/longitude."
                    )

            # Normalize state strings and map to indices
            self.df["state"] = self.df["state"].astype(str).str.strip()
            unknown = sorted(set(self.df["state"].unique()) - set(STATE_NAME_TO_IDX.keys()))
            if unknown:
                raise ValueError(
                    "Found states in CSV not present in STATE_NAME_TO_IDX mapping:\n"
                    f"{unknown}\n"
                    "Update the mapping or fix the CSV state names."
                )

            self.df["state_idx"] = self.df["state"].map(STATE_NAME_TO_IDX).astype(int)

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, filename: str) -> Image.Image:
        path = os.path.join(self.images_dir, filename)
        if os.path.exists(path):
            return Image.open(path).convert("RGB")

        if self.strict_files:
            raise FileNotFoundError(f"Missing image: {path}")

        # fallback: black image (256x256 per dataset)
        return Image.new("RGB", (256, 256), color=(0, 0, 0))

    def _encode(self, img: Image.Image) -> torch.Tensor:
        """
        Returns tensor [3,H,W] suitable for CLIPModel.get_image_features(pixel_values=...).
        """
        if self.processor is not None:
            pv = self.processor(images=img, return_tensors="pt")["pixel_values"]  # [1,3,H,W]
            return pv[0]
        if self.transform is None:
            raise ValueError("Provide either `processor` (recommended) or `transform`.")
        return self.transform(img)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        sample_id = int(row["sample_id"])

        img_n = self._load_image(row["image_north"])
        img_e = self._load_image(row["image_east"])
        img_s = self._load_image(row["image_south"])
        img_w = self._load_image(row["image_west"])

        out: Dict[str, Any] = {
            "sample_id": sample_id,
            "pixel_values_n": self._encode(img_n),
            "pixel_values_e": self._encode(img_e),
            "pixel_values_s": self._encode(img_s),
            "pixel_values_w": self._encode(img_w),
        }

        if not self.is_test:
            out["state_name"] = row["state"]  # optional, handy for debugging
            out["state_idx"] = torch.tensor(int(row["state_idx"]), dtype=torch.long)
            out["gps"] = torch.tensor([float(row["latitude"]), float(row["longitude"])], dtype=torch.float32)

        return out


def geoguessr_collate_fn(batch):
    out = {
        "sample_id": torch.tensor([b["sample_id"] for b in batch], dtype=torch.long),
        "pixel_values_n": torch.stack([b["pixel_values_n"] for b in batch], dim=0),
        "pixel_values_e": torch.stack([b["pixel_values_e"] for b in batch], dim=0),
        "pixel_values_s": torch.stack([b["pixel_values_s"] for b in batch], dim=0),
        "pixel_values_w": torch.stack([b["pixel_values_w"] for b in batch], dim=0),
    }

    if "state_idx" in batch[0]:
        out["state_idx"] = torch.stack([b["state_idx"] for b in batch], dim=0)
        out["gps"] = torch.stack([b["gps"] for b in batch], dim=0)
        # keep state_name as a python list (strings)
        out["state_name"] = [b.get("state_name", "") for b in batch]

    return out


# ----------------------------
# Test dataset (no CSV) — scan directory
# ----------------------------
class GeoGuessrTestDirDataset(Dataset):
    """
    Test dataset when there is NO CSV — only an images directory.

    Expects filenames like:
      img_000000_north.jpg
      img_000000_east.jpg
      img_000000_south.jpg
      img_000000_west.jpg

    Builds one sample per id (000000), requiring all 4 views.
    """
    VIEW_ORDER = ["north", "east", "south", "west"]
    PAT = re.compile(r"^img_(\d+)_(" + "|".join(VIEW_ORDER) + r")\.(jpg|jpeg|png|webp)$", re.IGNORECASE)

    def __init__(self, images_dir: str, processor=None, strict_files: bool = True):
        self.images_dir = images_dir
        self.processor = processor
        self.strict_files = strict_files

        # Map: sample_id -> {view: filename}
        by_id: Dict[int, Dict[str, str]] = {}

        files = sorted(os.listdir(images_dir))
        for fn in files:
            m = self.PAT.match(fn)
            if not m:
                continue
            sid = int(m.group(1))
            view = m.group(2).lower()
            by_id.setdefault(sid, {})[view] = fn

        if not by_id:
            raise ValueError(
                f"No matching files found in {images_dir}. Expected pattern like img_000000_north.jpg"
            )

        # Keep only ids that have all 4 views (or raise if strict)
        samples: List[Tuple[int, Dict[str, str]]] = []
        missing_any = []
        for sid, views in by_id.items():
            missing = [v for v in self.VIEW_ORDER if v not in views]
            if missing:
                missing_any.append((sid, missing))
                continue
            samples.append((sid, views))

        if self.strict_files and missing_any:
            # show a few for debugging
            msg = "\n".join([f"  id={sid} missing={miss}" for sid, miss in missing_any[:20]])
            raise ValueError(
                "Some sample_ids are missing required views (north/east/south/west). Examples:\n"
                + msg
                + ("\n..." if len(missing_any) > 20 else "")
            )

        # deterministic ordering by sample_id
        samples.sort(key=lambda x: x[0])
        self.sample_ids = [sid for sid, _ in samples]
        self.views_by_id = {sid: views for sid, views in samples}
        

        print(f"[GeoGuessrTestDirDataset] Found {len(by_id)} ids, using {len(self.sample_ids)} complete samples.")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _load_image(self, filename: str) -> Image.Image:
        path = os.path.join(self.images_dir, filename)
        if os.path.exists(path):
            return Image.open(path).convert("RGB")
        if self.strict_files:
            raise FileNotFoundError(f"Missing image: {path}")
        return Image.new("RGB", (256, 256), color=(0, 0, 0))

    def _encode(self, img: Image.Image) -> torch.Tensor:
        if self.processor is None:
            raise ValueError("Provide a CLIPProcessor (recommended) to encode images.")
        pv = self.processor(images=img, return_tensors="pt")["pixel_values"]  # [1,3,H,W]
        return pv[0]  # [3,H,W]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sid = self.sample_ids[idx]
        views = self.views_by_id[sid]

        fn_n = views["north"]
        fn_e = views["east"]
        fn_s = views["south"]
        fn_w = views["west"]

        img_n = self._load_image(fn_n)
        img_e = self._load_image(fn_e)
        img_s = self._load_image(fn_s)
        img_w = self._load_image(fn_w)

        return {
            "sample_id": sid,
            "pixel_values_n": self._encode(img_n),
            "pixel_values_e": self._encode(img_e),
            "pixel_values_s": self._encode(img_s),
            "pixel_values_w": self._encode(img_w),
            # keep filenames for output CSV
            "image_north": fn_n,
            "image_east": fn_e,
            "image_south": fn_s,
            "image_west": fn_w,
        }

def geoguessr_test_collate_fn(batch):
    return {
        "sample_id": torch.tensor([b["sample_id"] for b in batch], dtype=torch.long),
        "pixel_values_n": torch.stack([b["pixel_values_n"] for b in batch], dim=0),
        "pixel_values_e": torch.stack([b["pixel_values_e"] for b in batch], dim=0),
        "pixel_values_s": torch.stack([b["pixel_values_s"] for b in batch], dim=0),
        "pixel_values_w": torch.stack([b["pixel_values_w"] for b in batch], dim=0),
        "image_north": [b["image_north"] for b in batch],
        "image_east":  [b["image_east"] for b in batch],
        "image_south": [b["image_south"] for b in batch],
        "image_west":  [b["image_west"] for b in batch],
    }
