import os
import argparse
import pandas as pd
from pathlib import Path

NUM_EMOTION = 4

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

unused_emtions = set()
def convert_label(label):
    if label == "neutral":
        return 0
    elif label == "joy":
        return 1
    elif label == "anger":
        return 2
    elif label == "sadness":
        return 3
    else:
        unused_emtions.add(label)
        return -1

table = pd.read_csv(args.csv)
table["label"] = table["emotion"].apply(convert_label)
table = table[table["label"] != -1]
spkr_clm = "speaker (gender/id)"
utter_clm = "utterance_id"

first_subtables = []
second_subtables = []
for i in range(NUM_EMOTION):
    subtable = table[table["label"] == i].sort_values(spkr_clm)
    for idx in range(len(subtable)):
        if idx % 2 == 0:
            first_subtables.append(subtable[idx:idx+1])
        else:
            second_subtables.append(subtable[idx:idx+1])

first_subtable = pd.concat(first_subtables)
second_subtable = pd.concat(second_subtables)

first_subtable.to_csv(Path(args.output_dir) / "emotion_dev.csv")
second_subtable.to_csv(Path(args.output_dir) / "emotion_test.csv")
