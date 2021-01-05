import argparse
import pandas as pd
import os
import json
import random

def convert(args):
  if not os.path.isfile(args.tsv_output):
    train_df = pd.read_csv(args.train_input, delimiter = '|')[["event", "relation", "label"]]
    val_df = pd.read_csv(args.val_input, delimiter = '|')[["event", "relation", "label"]]
    df = pd.concat([train_df, val_df], ignore_index = True)
    df["title"] = df.apply(lambda x: x["event"] + "-" + x["relation"], axis = 1)
    df["text"] = df["label"]
    df["passage_id"] = df.index
    df[["passage_id", "title", "text"]].to_csv(args.tsv_output, sep = '\t', index = False)

  indexed_df = pd.read_csv(args.tsv_output, delimiter = '\t')
  # https://stackoverflow.com/a/58435858
  indexed_df = indexed_df.astype(str).groupby("title", sort = False).agg(unique_passage_id = ("passage_id", "unique"), unique_text = ("text", "unique")).reset_index()
  DATASET = "VCR"
  data = []
  size = indexed_df.shape[0]
  #print(indexed_df.head())
  for index, row in indexed_df.iterrows():
    title = row["title"]
    entry = {}
    entry["dataset"] = DATASET
    entry["question"] = title
    entry["answers"] = [] # leave blank for now, may add later
    positive_ctxs = []
    for passage_id, text in zip(row["unique_passage_id"], row["unique_text"]):
      positive_ctxs.append({
        "title": title,
        "text": text,
        "score": 1000, # may change later
        "title_score": 1,
        "passage_id": passage_id
      })
    entry["positive_ctxs"] = positive_ctxs

    # negative_ctxs
    negative_ctxs = []
    negative_index = random.randint(0, size - 1)
    while negative_index == index:
      negative_index = random.randint(0, size - 1)
    negative_row = indexed_df.iloc[negative_index]
    negative_title = negative_row["title"]
    for passage_id, text in zip(negative_row["unique_passage_id"], row["unique_text"]):
      negative_ctxs.append({
        "title": negative_title,
        "text": text,
        "score": 0,
        "title_score": 0,
        "passage_id": passage_id
      })
    entry["negative_ctxs"] = negative_ctxs

    # hard_negative_ctxs
    hard_negative_ctxs = []
    entry["hard_negative_ctxs"] = hard_negative_ctxs

    data.append(entry)

  #data = json.dumps(data)
  with open(args.dpr_input, 'w') as f:
    json.dump(data, f, indent = 4)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--train_input", type = str, required = True, help = "training annotation bsv file")
  parser.add_argument("--val_input", type = str, required = True, help = "validation annotation bsv file")
  parser.add_argument("--tsv_output", type = str, required = True, help = "output location for tsv file")
  parser.add_argument("--dpr_input", type = str, required = True, help = "output location for DPR input (json)")

  args = parser.parse_args()

  convert(args)
