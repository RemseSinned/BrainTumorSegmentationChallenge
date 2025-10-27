from pathlib import Path
import torch
from configs.utils import load_config
from src.prepare_dataset import prepare_data
from src.train import train

cfg = load_config("../configs/paths.yaml")
RAW_DIR = Path(("../" + cfg["raw_dir"]))
PRE_DIR = Path(("../" + cfg["preprocessed_dir"]))


def main():
    # prepare_data()
    train()


if __name__ == "__main__":
    main()
