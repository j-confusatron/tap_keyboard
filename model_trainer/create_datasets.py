import glob
import os
import json
from keyless_keyboard.config import KeylessConfig
import numpy as np
import random

F_RAW_DATA = os.path.join('data')
F_CLEAN_DATA = os.path.join(F_RAW_DATA, 'clean')

keyless_config = KeylessConfig()

def get_raw_data():
    raw_data_files = glob.glob(F_RAW_DATA+os.sep+'*.json')
    raw_data = []
    for f_data in raw_data_files:
        with open(f_data, 'r') as fp_data:
            raw_data += json.load(fp_data)
    return raw_data

def get_clean_data(raw_data):
    clean_data = [{'y': raw_sample['y']} for raw_sample in raw_data]
    for i, raw_sample in enumerate(raw_data):
        points = np.array([x['pts'] for x in raw_sample['x']])
        indices = get_spaced_indices(len(points))
        clean_data[i]['x'] = points[indices].tolist()
    random.shuffle(clean_data)
    return clean_data

def get_spaced_indices(num_elems):
    array = np.arange(num_elems)
    indices = array[np.round(np.linspace(0, len(array)-1, keyless_config.num_frames)).astype(int)]
    return indices

def write_datasets(clean_data):
    ratios = keyless_config.data_ratio
    num_samples = len(clean_data)
    n_train = int(round(ratios[0] * num_samples, 0))
    n_val = int(round(ratios[1] * num_samples, 0))
    datasets = [
        ('train.json', clean_data[:n_train]),
        ('validate.json', clean_data[n_train:n_train+n_val]),
        ('test.json', clean_data[n_train+n_val:])
    ]
    for data in datasets:
        with open(os.path.join(F_CLEAN_DATA, data[0]), 'w') as fp_data:
            json.dump(data[1], fp_data)

if __name__ == '__main__':
    raw_data = get_raw_data()
    clean_data = get_clean_data(raw_data)
    write_datasets(clean_data)