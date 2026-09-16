import matplotlib.pyplot as plt
import pickle
import numpy as np


def make_graph(pickle_file_path:str):
    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)
    