import argparse
import pandas as pd
import os
import json
import random
from tqdm import tqdm

def convert(args):
  with open(args.input, 'r') as f:
    dpr_output = json.load(f)

  event_to_labels = {}
  for entry in tqdm(dpr_output):
    event_relation = entry["question"]
    labels = []
    for context in entry["ctxs"]:
      if context["text"] == event_relation:
        continue
      if context["title"] not in labels:
        labels.append(context["title"])
    event_to_labels[event_relation] = labels

  with open(args.prev_train, 'r') as f:
    prev_train_retrieval = json.load(f)

  with open(args.prev_val, 'r') as f:
    prev_val_retrieval = json.load(f)

  new_train_retrieval = []
  for entry in tqdm(prev_train_retrieval):
    event = entry["event"]
    relation = entry["inference_relation"]
    entry["generations"] = event_to_labels.get("{}-{}".format(event, relation), [])
    new_train_retrieval.append(entry)

  with open(args.train_output, 'w') as f:
    json.dump(new_train_retrieval, f, indent = 4)

  new_val_retrieval = []
  for entry in tqdm(prev_val_retrieval):
    event = entry["event"]
    relation = entry["inference_relation"]
    entry["generations"] = event_to_labels["{}-{}".format(event, relation)]
    new_val_retrieval.append(entry)

  with open(args.val_output, 'w') as f:
    json.dump(new_val_retrieval, f, indent = 4)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--input", type = str, required = True, help = "DPR retrieval output json file")
  parser.add_argument("--prev_train", type = str, required = True, help = "previous Embedding KNN retrieval train json")
  parser.add_argument("--prev_val", type = str, required = True, help = "previous Embedding KNN retrieval val json")
  parser.add_argument("--train_output", type = str, required = True, help = "DPR retrieval train json")
  parser.add_argument("--val_output", type = str, required = True, help = "DPR retrieval val json")

  args = parser.parse_args()

  convert(args)
