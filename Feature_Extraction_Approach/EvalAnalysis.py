import pandas as pd

if __name__ == '__main__':
    with open('../data/Evaluation/eval_form.csv', 'r') as f:
        raw = f.readlines()
    i = 1
    d = {"id": i, "text": '', 'rating1': [], 'rating2': []}
