"""Load image2mass .pklz records
Each record is an lz4 block-compressed pickle holding the WebP image, weight
(pounds), dimensions (inches), thickness mask, and id. 
"""

import argparse
import io
import json
import pickle
from pathlib import Path

import lz4.block
from PIL import Image

# Pickle can execute arbitrary imported globals. The source records only need
# these NumPy constructors, so everything else is rejected during unpickling.
ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype")}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")
        return super().find_class(module, name)


def load_record(path):
    """Safely decompress one record and convert common fields to model-friendly types."""
    # image_data stays in memory: no temporary image file is created.
    raw = _Unpickler(io.BytesIO(lz4.block.decompress(Path(path).read_bytes()))).load()
    return {"id": int(raw["id"]),
            "image": Image.open(io.BytesIO(raw["image_data"])).convert("RGB"),
            "dims": tuple(float(d) for d in raw["dimensions"]),
            "weight": float(raw["weight"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pairs", type=Path)
    parser.add_argument("--root", type=Path, default=Path("amazon_data_10k"))
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()
    with args.pairs.open(encoding="utf-8") as handle:
        # zip(..., range(n)) streams only the requested prefix instead of
        # loading a potentially large pair file in full.
        pairs = [json.loads(line) for line, _ in zip(handle, range(args.n))]
    for pair in pairs:
        for side in ("a", "b"):
            record = load_record(args.root / pair[f"path_{side}"])
            ok = record["id"] == pair[f"id_{side}"] and record["weight"] == pair[f"weight_{side}"]
            print(f"{pair['pair_id']} {side.upper()}: id {record['id']}, image {record['image'].size}, "
                  f"dims {record['dims']}, weight {record['weight']} lb, {'OK' if ok else 'MISMATCH'}")
        print(f"  label {pair['label']}")


if __name__ == "__main__":
    main()
