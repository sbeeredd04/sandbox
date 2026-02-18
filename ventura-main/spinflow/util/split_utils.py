import random
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from spinflow.util.log_utils import *

def sample_splits(
        split_dir:  str,
        ride_names: list[str],
        header: list[str],
        train_frac: float = 0.8,
        val_frac:   float = 0.1,
        test_frac:  float = 0.1,
        seed:       int   = 42
    ) -> None:
        """
        Given a list of ride_names, randomly split them into train/val/test
        according to the provided fractions, and write:
           split_dir/train.txt
           split_dir/val.txt
           split_dir/test.txt
        """
        assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"
        rnd = random.Random(seed)
        names = ride_names.copy()
        rnd.shuffle(names)

        n = len(names)
        n_train = int(train_frac * n)
        n_val   = int(val_frac   * n)

        splits = {
            "train":      names[:n_train],
            "val": names[n_train:n_train + n_val],
            "test":       names[n_train + n_val:]
        }

        split_dir = Path(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_list in splits.items():
            df = pd.DataFrame(splits[split_name], columns=header)
            # Sort by header
            df = df.sort_values(by=header)
            out = split_dir / f"{split_name}.txt"
            df.to_csv(out, index=False, header=True)
        
        logging.info(f"Wrote splits to {split_dir}")