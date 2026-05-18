import os
import yaml

class ConfigLoader:
    def __init__(self, config_file='configs/default.yaml'):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value

    def save(self, output_file=None):
        if output_file is None:
            output_file = self.config_file
        with open(output_file, 'w') as file:
            yaml.dump(self.config, file)

    def update(self, updates):
        self.config.update(updates)