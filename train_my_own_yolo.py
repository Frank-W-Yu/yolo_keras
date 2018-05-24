import os
import PIL
from PIL import Image
import numpy as np
# import keras

def process_data(image_path, label_path, starting_file=0, batch_size=1000, regions=[13, 13]):
    '''
    load the image and labels and preprocess the data
    box params format (class, x_center, y_center, width, height)
    :param image_path:
    :param label_path:
    :param starting_file:
    :param batch_size:
    :return:
    '''
    images = []
    all_labels = []
    fns = os.listdir(image_path)
    max_labels = 0
    for fn in fns[starting_file: starting_file+batch_size]:
        labels = []
        images.append(Image.open(image_path+fn))
        txt_fn = label_path + fn.split('.')[0] + '.txt'
        with open(txt_fn, 'r') as f:
            label_txt = f.read()
            lines = label_txt.split('\n\n')
            f.close()

        for line in lines:
            params = line.split(' ')
            if len(params) == 5:
                labels.append(params[1:]+params[0:1])
        all_labels.append(np.array(labels, dtype=np.float32).reshape((-1, 5)))
        if len(labels) > max_labels:
            max_labels = len(labels)

    ori_size = np.array([images[0].width, images[0].height])
    ori_size = np.expand_dims(ori_size, axis=0)
    n_strips_x, n_strips_y = regions
    n_strips_x = n_strips_x * 32
    n_strips_y = n_strips_y * 32

    '''
    Image preprocessing, yolo only supports resolution of 32*n_strips_x by 32*n_strips_y
    '''
    processed_images = [i.resize((n_strips_x, n_strips_y), Image.BICUBIC) for i in images]
    processed_images = [np.array(image, dtype=np.float) for image in processed_images]
    processed_images = [image/255. for image in processed_images]

    # add zero pad, all training images has the same number of labels
    for i, labels in enumerate(all_labels):
        if labels.shape[0] < max_labels:
            zero_padding = np.zeros((max_labels-labels.shape[0], 5), dtype=np.float32)
            all_labels[i] = np.vstack((labels, zero_padding))
    return np.array(processed_images), np.array(all_labels)

if __name__ == "__main__":
    image_path = '/media/frank/Storage/Project/Kaggle/WAD/input/train_color/'
    label_path = '/media/frank/Storage/Project/Kaggle/WAD/input/train_label_txts/'
    process_data(image_path, label_path, 0, 10)
