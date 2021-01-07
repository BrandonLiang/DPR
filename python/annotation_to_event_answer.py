import argparse
import pandas as pd
import os
import json
import random

def convert(args):
  with open(args.input, 'r') as f:
    data = json.load(f)
  questions = []
  answers = []
  for entry in data:
    questions.append(entry["question"])
    answers.append([])
  df = pd.DataFrame({
    "question": questions,
    "answers": answers
  })

  df.to_csv(args.output, sep = '\t', header = False, index = False)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--input", type = str, required = True, help = "training json file")
  parser.add_argument("--output", type = str, required = True, help = "event to answer csv file")

  args = parser.parse_args()

  convert(args)
