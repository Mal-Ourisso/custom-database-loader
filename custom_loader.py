
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

def getFilePaths(directory):
	return np.array([file_path for file_path in glob(directory+"/*/*.jpeg")])

def getImageLabel(file_path, h=512, w=512):
	img = tf.io.decode_jpeg(tf.io.read_file(file_path), channels=3)
	x = tf.image.resize_with_pad(img, h, w)
	y = file_path.split("/")[-2]
	return x, y

def loadImagesGen(file_paths, rand=True, h=512, w=512):
	if rand:
		np.random.shuffle(file_paths)
	for file_path in file_paths:
		yield getImageLabel(file_path, h, w)

def customDataset(directory, rand=True, batch_size=32, h=512, w=512):
	file_paths = getFilePaths(directory)
	base_dtype = next(loadImagesGen(file_paths, False, h, w))[0].dtype
	return tf.data.Dataset.from_generator(loadImagesGen, args=(file_paths, rand, h, w), output_types=base_dtype).batch(batch_size)
