import pandas as pd
import argparse
import re


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess and annotate results")
    parser.add_argument("--input", "-i", type=str, default="listener-gpt-5.2_1_none.csv")

    args = parser.parse_args()

    results = pd.read_csv(args.input, header=0)
    speaker_answer = results["speaker_answer"]

    words = r'\b(left|right|top|bottom|below|above|column|row|next|cell|curtain|dashed)\b'

    exclusion = results[speaker_answer.str.contains(words, case=False, na=False)]

    images_ids = exclusion["image_file"].unique().tolist()
    if images_ids:
        print(f"exclusion list: {images_ids}")
        print(f"number of answers to be excluded: {len(images_ids)}")
    else:
        print("No answers need to be excluded.")