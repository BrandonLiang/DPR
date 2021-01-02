import argparse
import pandas as pd
import os
import json

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
  # https://hackersandslackers.com/multiple-aggregations-pandas/
  indexed_df = indexed_df.astype(str).groupby("title", sort = False).agg(unique_passage_id = ("passage_id", "unique"), unique_text = ("text", "unique"))
  DATASET = "VCR"
  data = []


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--train_input", type = str, required = True, help = "training annotation bsv file")
  parser.add_argument("--val_input", type = str, required = True, help = "validation annotation bsv file")
  parser.add_argument("--tsv_output", type = str, required = True, help = "output location for tsv file")
  parser.add_argument("--dpr_input", type = str, required = True, help = "output location for DPR input (json)")

  args = parser.parse_args()

  convert(args)
