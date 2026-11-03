from src.utils.data import generate_synth_dataset
from src.config import probing_sets_dict
import argparse

def main():
    parser = argparse.ArgumentParser(description="Synth data generation script.")
    parser.add_argument("-dim", "--dim", type=int, help="dimension of the dataset", default=30000)
    parser.add_argument("-seed", "--seed", type=int, help="Seed for (pseudo) casual generator.", default=42)
    args = parser.parse_args()

    dim = args.dim
    seed = args.seed

    for bias in probing_sets_dict.keys():
        _, _, _ = generate_synth_dataset(dim, probing_sets_dict[bias], dataset_name=bias, seed=seed)

if __name__ == "__main__":
    main()
