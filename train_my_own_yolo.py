import os
import PIL
from PIL import Image
import numpy as np
import argparse
# import keras

def parse_args():
    '''
    parse arguments passed by command line
    :return: parsed args
    '''
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--image_path')
    argparser.add_argument('-l', '--label_path')
    argparser.add_argument('-s', '--starting_file', default=0)
    argparser.add_argument('-b', '--batch_size', default=1000)
    argparser.add_argument('-r', '--regions', default=[13, 13])
    args = argparser.parse_args()
    return args


def process_data(image_path, label_path, starting_file, batch_size, regions):
    '''
    load the image and labels and preprocess the data
    box params format (class, x_center, y_center, width, height)
    :param image_path:
    :param label_path:
    :param starting_file:
    :param batch_size:
    :param regions:
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

def get_detector_mask(boxes, anchors, regions):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    :param boxes: label boxes of the train data set, shape (m, n, 5),
            m: number of samples, n: number of max labels for each image 5: x, y, w, h, c
    :param anchors:
    :return:
    '''
    detectors_mask = [0 for _ in range(len(boxes))]
    matching_true_boxes = [0 for _ in range((len(boxes)))]
    for i, box in enumerate(boxes):
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, regions)
    return np.array(detectors_mask), np.array(matching_true_boxes)

def preprocess_true_boxes(true_boxes, anchors, regions):
    """Find detector in YOLO where ground truth box should appear.

    Parameters
    ----------
    true_boxes : array
        List of ground truth boxes in form of relative x, y, w, h, class.
        Relative coordinates are in the range [0, 1] indicating a percentage
        of the original image dimensions.
        shape: (n, 5), n: number of max labels
    anchors : array
        List of anchors in form of w, h.
        Anchors are assumed to be in the range [0, conv_size] where conv_size
        is the spatial dimension of the final convolutional features.
    image_size : array-like
        List of image dimensions in form of h, w in pixels.

    Returns
    -------
    detectors_mask : array
        0/1 mask for detectors in [conv_height, conv_width, num_anchors, 1]
        that should be compared with a matching ground truth box.
    matching_true_boxes: array
        Same shape as detectors_mask with the corresponding ground truth box
        adjusted for comparison with predicted parameters at training time.
    """
    num_anchors = len(anchors)
    num_box_params = true_boxes.shape[1]
    conv_height, conv_width = regions
    detector_mask = np.zeros((conv_height, conv_width, num_anchors, num_box_params), dtype=np.float32)
    matching_true_boxes = np.zeros((conv_height, conv_width, num_anchors, num_box_params), dtype=np.float32)
    for box in true_boxes:
        cls = box[4:5]
        box = box[0:4] * np.array([conv_width, conv_height, conv_width, conv_height])
        i = np.floor(box[1]).astype('int')
        j = np.floor(box[0]).astype('int')
        best_iou = 0
        best_anchor = 0
        for k, anchor in enumerate(anchors):
            box_maxes = box[2:4] / 2.0
            box_mins = -box_maxes
            anchor_maxes = anchor / 2.0
            anchor_mins = -anchor_maxes

            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.0)
            intersect_area = intersect_wh[0] * intersect_wh[1]
            box_area = box[2] * box[3]
            anchor_area = anchor[0] * anchor[1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)
            if iou > best_iou:
                best_iou = iou
                best_anchor = k

        if best_iou > 0:
            detector_mask[i, j, best_anchor] = 1


def main():
    args = parse_args()
    image_path = args.image_path
    label_path = args.label_path
    starting_file = args.starting_file
    batch_size = args.batch_size
    regions = args.regions

    anchors = None
    processed_images, processed_labels = process_data(image_path, label_path, starting_file, batch_size, regions)
    get_detector_mask(processed_labels, anchors, regions)


if __name__ == "__main__":
    main()
    # image_path = '/media/frank/Storage/Project/Kaggle/WAD/input/train_color/'
    # label_path = '/media/frank/Storage/Project/Kaggle/WAD/input/train_label_txts/'
    # process_data(image_path, label_path, 0, 10)
