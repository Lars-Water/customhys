import json

'''
    Load config file from path.

    file_path:  Path to config file.
'''


def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
