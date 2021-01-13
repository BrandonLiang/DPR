import argparse
import pandas as pd
import os
import json
import random

def convert(args):
  train_df = pd.read_csv(args.train_input, delimiter = '|')[["event", "relation", "label"]]
  val_df = pd.read_csv(args.val_input, delimiter = '|')[["event", "relation", "label"]]

  train_df["title"] = train_df.apply(lambda x: x["event"] + "-" + x["relation"], axis = 1)
  train_df["text"] = train_df["label"]
  train_df["passage_id"] = train_df.index
  train_df[["passage_id", "title", "text"]].to_csv(args.tsv_train_output, sep = '\t', index = False)

  val_df["title"] = val_df.apply(lambda x: x["event"] + "-" + x["relation"], axis = 1)
  val_df["text"] = val_df["label"]
  val_df["passage_id"] = val_df.index
  val_df[["passage_id", "title", "text"]].to_csv(args.tsv_val_output, sep = '\t', index = False)

  all_df = pd.concat([train_df, val_df], ignore_index = True)[["passage_id", "title", "text"]]
  all_df.to_csv(args.tsv_all_output, sep = '\t', index = False)


  #all_df = all_df.astype(str).groupby("title", sort = False).agg(unique_passage_id = ("passage_id", "unique"), unique_text = ("text", "unique")).reset_index()

  convert_to_DPR_input(all_df, args.dpr_all_input)
  convert_to_DPR_input(train_df, args.dpr_train_input)
  convert_to_DPR_input(val_df, args.dpr_val_input)

def convert_to_DPR_input(df, output_location, DATASET = "VCR"):
  # https://stackoverflow.com/a/58435858
  df = df.astype(str).groupby("title", sort = False).agg(unique_passage_id = ("passage_id", "unique"), unique_text = ("text", "unique")).reset_index()
  data = []
  size = df.shape[0]
  #print(df.head())
  for index, row in df.iterrows():
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
    negative_row = df.iloc[negative_index]
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
  with open(output_location, 'w') as f:
    json.dump(data, f, indent = 4)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--train_input", type = str, required = True, help = "training annotation bsv file")
  parser.add_argument("--val_input", type = str, required = True, help = "validation annotation bsv file")
  parser.add_argument("--tsv_all_output", type = str, required = True, help = "output location for tsv file")
  parser.add_argument("--tsv_train_output", type = str, required = True, help = "output location for tsv file")
  parser.add_argument("--tsv_val_output", type = str, required = True, help = "output location for tsv file")
  parser.add_argument("--dpr_all_input", type = str, required = True, help = "output location for DPR train input (json)")
  parser.add_argument("--dpr_train_input", type = str, required = True, help = "output location for DPR train input (json)")
  parser.add_argument("--dpr_val_input", type = str, required = True, help = "output location for DPR val input (json)")

  args = parser.parse_args()

  convert(args)
