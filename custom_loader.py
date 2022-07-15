
import numpy as np
import tensorflow as tf
import keras as kr

from numpy.random import default_rng
from time import time
from glob import glob

rng = default_rng()

def _measureTime(func, *args):
	start = time()
	func(*args)
	return time()-start

def getFileNames(directory):
	return np.array([file_path for file_path in glob(directory+"/*/*.jpeg")])

def getImageLabel(file_path):
	x = tf.keras.utils.load_img(file_path)
	y = file_path.split("/")[-2]
	print(y)
	return x, y

