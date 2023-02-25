import json
import os

F_CONFIG = os.path.join('config', 'config.json')

class KeylessConfig():

    def __init__(self):
        with open(F_CONFIG, 'r') as fp_config:
            self.config_data = json.load(fp_config)

    @property
    def fps(self):
        return self.config_data['fps']
    
    @property
    def capture_time(self):
        return self.config_data['capture_time']
    
    @property
    def num_frames(self):
        return int(round((self.capture_time / 1000) * self.fps, 0))
    
    @property
    def num_hand_points(self):
        return self.config_data['num_hand_points']
    
    @property
    def data_ratio(self):
        return tuple(self.config_data['data_ratio'])
    
    @property
    def mean(self):
        return tuple(self.config_data['mean'])
    
    @property
    def stdev(self):
        return tuple(self.config_data['stdev'])
    
    @property
    def thresholds(self):
        return tuple(self.config_data['thresholds'])
    
    @property
    def train(self):
        return self.config_data['train']
    
    @property
    def evaluate(self):
        return self.config_data['evaluate']