
import numpy as np
#import tensorflow as tf
#import keras as kr

from numpy.random import default_rng
from time import time
from glob import glob

rng = default_rng()
dir_dataset = "/home/mauricio/dados/Mauricio/OCT2017"

def _measureTime(func, *args):
        start = time()i
        func(*args)
        return time()-start

def getFileName(directory):
        return np.array([file_path.split for file_path in glob(directory+"/*.jpeg")])
