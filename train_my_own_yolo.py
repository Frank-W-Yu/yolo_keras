import os
import PIL
from PIL import Image
import numpy
import keras

def process_data(image_path, label_path, starting_file=0, batch_size=1000):
    '''
    load the image and labels and preprocess the data
    :param image_path:
    :param label_path:
    :param starting_file:
    :param batch_size:
    :return:
    '''
    images = []
    fns = os.lisdir(image_path)
    for fn in fns[starting_file: starting_file+batch_size]:
        images.append(Image.open(image_path+'fn'))
        