#!/usr/bin/env python

import argparse
import os
import numpy as np
from dtw_score import dtw, LABELS
from matplotlib import pyplot as plt

def make_nice_graphs(error, fname):
    plt.figure(figsize=(14, 9))
    x_axis = error.shape[0]
    for i, v in enumerate(LABELS):
        plt.plot(range(x_axis), error[:, i], label=f'{v} error')
    plt.legend()
    plt.title('error of gold and test poses over time (one rep)')
    plt.savefig(f'plots/{fname}-error_report.png', bbox_inches='tight')
    plt.close()

def generate_error_text(error):
    split_data = np.array_split(error, 3, axis=0)
    simplified_error = np.array([np.sum(part, axis=0) for part in split_data])
    out = 'A person has just completed an excerise, and their performance was compared to a gold standard trainer. The Euclidean distance of some key points between the person and the trainer have been calculated. Below is a table describing the total error (in meters), where the first column describes the area, and the next three columns show the total error over the first third, second third, and last third of one rep of the exercise. Notice that the error for the Hip is 0 because all the measurements were taken relative to the hip location (you do not need to repeat that fact).\n'
    max_length = max([len(x) for x in LABELS])
    for j in range(17):
        out += LABELS[j] + ' ' * (max_length - len(LABELS[j])) + ' error: \t'
        for bucket in range(3):
            out += '{:.2f}'.format(simplified_error[bucket, j]) + '\t'
        out += '\n'
    print(out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', dest='gold', type=str, default=None)
    parser.add_argument('--test', dest='test', type=str, default=None)
    args = parser.parse_args()
    assert args.gold is not None or args.test is not None, 'missing paths to gold and test poses'
    error = dtw(args.gold, args.test)
    fname = f'{os.path.basename(args.gold)}-{os.path.basename(args.test)}'
    make_nice_graphs(error, fname)
    generate_error_text(error)
