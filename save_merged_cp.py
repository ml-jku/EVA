import os
import argparse
import torch

from src.utils import get_merged_model_and_tokenizer


def main(cp_path, dst_path, device):
    print("Loading model and tokenizer from", cp_path)
    model, tokenizer = get_merged_model_and_tokenizer(cp_path, device)
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    model.save_pretrained(dst_path)
    tokenizer.save_pretrained(dst_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp_path", type=str, required=True)
    parser.add_argument("--dst_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    main(args.cp_path, args.dst_path, args.device)