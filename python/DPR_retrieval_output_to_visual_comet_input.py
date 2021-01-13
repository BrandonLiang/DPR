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

  with open(args.prev, 'r') as f:
    prev_retrieval = json.load(f)

  new_retrieval = []
  for entry in tqdm(prev_retrieval):
    event = entry["event"]
    relation = entry["inference_relation"]
    entry["generations"] = event_to_labels.get("{}-{}".format(event, relation), []) # assert most not empty
    new_retrieval.append(entry)

  with open(args.output, 'w') as f:
    json.dump(new_retrieval, f, indent = 4)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--input", type = str, required = True, help = "DPR retrieval output json file")
  parser.add_argument("--prev", type = str, required = True, help = "previous Embedding KNN retrieval json")
  #parser.add_argument("--prev_val", type = str, required = True, help = "previous Embedding KNN retrieval val json")
  parser.add_argument("--output", type = str, required = True, help = "DPR retrieval json")
  #parser.add_argument("--val_output", type = str, required = True, help = "DPR retrieval val json")

  args = parser.parse_args()

  convert(args)
