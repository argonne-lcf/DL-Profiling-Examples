"""
Download MNIST data using Keras interface
"""
from keras.datasets import mnist
import os,keras
download_path = os.getcwd() + '/minst.tgz'

print(download_path)
(x_train, y_train), (x_test, y_test) = mnist.load_data(download_path)
