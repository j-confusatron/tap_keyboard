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