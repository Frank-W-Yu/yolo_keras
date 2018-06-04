import os
import PIL
from PIL import Image
import numpy as np
import argparse
# import sys
# sys.path.append('/usr/local/bin/cuda-9.0/lib64')
import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from yad2k.models.keras_yolo import (yolo_body, yolo_eval, yolo_head, yolo_loss)
from yad2k.utils.draw_boxes import draw_boxes

def parse_args():
    '''
    parse arguments passed by command line
    :return: parsed args
    '''
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--image_path')
    argparser.add_argument('-l', '--label_path')
    argparser.add_argument('-a', '--anchors_path')
    argparser.add_argument('-c', '--class_names')

    argparser.add_argument('-s', '--starting_file', default=0)
    argparser.add_argument('-b', '--batch_size', default=900)
    argparser.add_argument('-m', '--max_batches', default=0)
    argparser.add_argument('-r', '--regions', default=[13, 13])
    argparser.add_argument('-p', '--load_previous_trained', default='F')
    argparser.add_argument('-w', '--weights2load', default='trained_stage_3_best.h5')
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
    # cls_dict = {33:0, 34}

    for fn in fns[starting_file: starting_file+batch_size]:
        labels = []
        images.append(Image.open(image_path+fn))
        txt_fn = str(label_path) + str(fn.split('.')[0]) + '.txt'
        with open(txt_fn, 'r') as f:
            label_txt = f.read()
            if '\n\n' in label_txt:
                lines = label_txt.split('\n\n')
            else:
                lines = label_txt.split('\n')
            f.close()

        for line in lines:
            params = line.split(' ')
            if len(params) == 5:
                # labels.append(params[1:]+params[0:1])
                cls = int(params[0]) - 33
                labels.append([params[2], params[1], params[4], params[3], cls])
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
    conv_width, conv_height = regions
    detector_mask = np.zeros((conv_height, conv_width, num_anchors, 1), dtype=np.float32)
    matching_true_boxes = np.zeros((conv_height, conv_width, num_anchors, num_box_params), dtype=np.float32)
    for box in true_boxes:
        cls = box[4:5]
        box = box[0:4] * np.array([conv_height, conv_width, conv_height, conv_width])
        i = np.floor(box[0]).astype('int')
        j = np.floor(box[1]).astype('int')
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
            adjusted_box = np.array([box[0]-i, box[1]-j,
                                     np.log(box[2]/anchors[best_anchor][0]),
                                     np.log(box[3]/anchors[best_anchor][1]), cls],
                                    dtype=np.float32)
            matching_true_boxes[i, j, best_anchor] = adjusted_box
    return detector_mask, matching_true_boxes

def create_model(anchors, class_names, regions, load_pretrained=True, freeze_body=True):
    '''
    create the model
    :param anchors:
    :param class_names:
    :param regions :type list
    :param num_anchors
    :param load_pretrained:
    :param freeze_body:
    :return: YOLO v2 with new output layers
             Yolo v2 with custom loss Lambda Layer
    '''
    conv_x, conv_y = regions
    num_anchors = len(anchors)
    x_shape, y_shape = conv_x * 32, conv_y * 32
    detectors_mask_shape = (conv_y, conv_x, 5, 1)
    matching_boxes_shape = (conv_y, conv_x, 5, num_anchors)

    # Create model input layers
    image_input = Input(shape=(y_shape, x_shape, 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Create model body
    yolo_model = yolo_body(image_input, len(anchors), len(class_names))
    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

    if load_pretrained:
        topless_yolo_path = os.path.join('model_data', 'yolo_topless.h5')
        if not os.path.exists(topless_yolo_path):
            print("Create topless weights file first")
            yolo_path = os.path.join('model_data', 'yolo.h5')
            model_body = load_model(yolo_path)
            model_body = Model(model_body.inputs, model_body.layers[-2].output)
            model_body.save_weights(topless_yolo_path)
        topless_yolo.load_weights(topless_yolo_path)

    if freeze_body:
        for layer in topless_yolo.layers:
            layer.trainable = False
    final_layer = Conv2D(len(anchors)*(5+len(class_names)), (1, 1), activation='linear')(topless_yolo.output)

    model_body = Model(image_input, final_layer)

    with tf.device('/cpu:0'):
        model_loss = Lambda(yolo_loss,
                            output_shape=(1,),
                            name='yolo_loss',
                            arguments={'anchors': anchors,
                                       "num_classes":len(class_names)})(
            [model_body.output, boxes_input, detectors_mask_input, matching_boxes_input])
    model = Model([model_body.input, boxes_input, detectors_mask_input, matching_boxes_input], model_loss)

    return model_body, model

def initial_train(model, class_names, anchors, image_data, boxes, detectors_mask, matching_true_boxes, regions, validation_split=0.1):
    '''

    :param model:
    :param class_names:
    :param anchors:
    :param image_data:
    :param boxes:
    :param detectors_mask:
    :param matching_true_boxes:
    :param validation_split:
    :return:
    '''
    model.compile(optimizer='adam', loss={'yolo_loss':lambda y_true, y_pred: y_pred})
    logging = TensorBoard()
    checkpoint = ModelCheckpoint("trained_stage_3_best.h5", monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')
    # print(image_data.shape)
    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=validation_split,
              batch_size=32,
              epochs=5,
              callbacks=[logging])

    model.save_weights('trained_stage_1.h5')

    model_body, model = create_model(anchors, class_names, regions, load_pretrained=False, freeze_body=False)
    model.load_weights('trained_stage_1.h5')
    model.compile(optimizer='adam', loss={'yolo_loss':lambda y_true, y_pred: y_pred})
    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=validation_split,
              batch_size=8,
              epochs=30,
              callbacks=[logging])
    model.save_weights('trained_stage_2.h5')

    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=validation_split,
              batch_size=8,
              epochs=30,
              callbacks=[logging, checkpoint, early_stopping])
    model.save_weights('trained_stage_3.h5')
    return model

def recur_train(model, class_names, anchors, image_data, boxes, detectors_mask, matching_true_boxes, regions, validation_split=0.1):
    '''

    :param model:
    :param class_names:
    :param anchors:
    :param image_data:
    :param boxes:
    :param detectors_mask:
    :param matching_true_boxes:
    :param validation_split:
    :return:
    '''
    # model.compile(optimizer='adam', loss={'yolo_loss':lambda y_true, y_pred: y_pred})
    logging = TensorBoard()
    checkpoint = ModelCheckpoint("trained_stage_3_best.h5", monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

    # model.compile(optimizer='adam', loss={'yolo_loss':lambda y_true, y_pred: y_pred})
    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=validation_split,
              batch_size=8,
              epochs=30,
              callbacks=[logging])
    model.save_weights('trained_stage_2.h5')

    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=validation_split,
              batch_size=8,
              epochs=30,
              callbacks=[logging, checkpoint, early_stopping])
    model.save_weights('trained_stage_3.h5')
    return model

def draw(model_body, class_names, anchors, image_data, n_epoch, image_set='val',
            weights_name='trained_stage_3_best.h5', out_path="output_images", save_all=True):
    '''
    Draw bounding boxes on image data
    '''
    if image_set == 'train':
        image_data = np.array([np.expand_dims(image, axis=0) for image in image_data[:int(len(image_data)*.9)]])
    elif image_set == 'val':
        image_data = np.array([np.expand_dims(image, axis=0) for image in image_data[int(len(image_data)*.9):]])
    elif image_set == 'all':
        image_data = np.array([np.expand_dims(image, axis=0) for image in image_data])
    else:
        ValueError("draw argument image_set must be 'train', 'val', or 'all'")
    # model.load_weights(weights_name)
    print(image_data.shape)
    model_body.load_weights(weights_name)

    # Create output variables for prediction.
    yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs, input_image_shape, score_threshold=0.07, iou_threshold=0.0)

    # Run prediction on overfit image.
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for i in range(len(image_data)):
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                model_body.input: image_data[i],
                input_image_shape: [image_data.shape[2], image_data.shape[3]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for image.'.format(len(out_boxes)))
        print(out_boxes)

        # Plot image with predicted boxes.
        image_with_boxes = draw_boxes(image_data[i][0], out_boxes, out_classes,
                                    class_names, out_scores)
        # Save the image:
        if save_all or (len(out_boxes) > 0):
            image = PIL.Image.fromarray(image_with_boxes)
            image.save(os.path.join(out_path,str(i)+'_'+str(n_epoch)+'.png'))

        # To display (pauses the program):
        # plt.imshow(image_with_boxes, interpolation='nearest')
        # plt.show()

def get_anchors(anchors_path, region):
    with open(anchors_path, 'r') as f:
        txt = f.read()
        f.close()
    anchor_pairs = txt.split(', ')
    anchors = []
    for anchor_pair in anchor_pairs:
        anchor = np.array(anchor_pair.split(' '), dtype=np.float32)
        anchor = anchor * np.array(region)
        anchors.append(anchor[::-1])
    return np.array(anchors)

def get_max_batches(image_path, batch_size):
    total_file_count = len(os.listdir(image_path))
    batches, residual = divmod(total_file_count, batch_size)
    if residual == 0:
        return batches
    return batches + 1

def get_class_names(class_name_path):
    with open(class_name_path, 'r') as f:
        classes = f.read()
        f.close()
    return classes.split('\n')[:-1]

def get_regions(region):
    regions = region.split('*')
    regions = [int(i) for i in regions]
    return regions

def get_starting_file(arg, batch_size):
    if isinstance(arg, int):
        return arg
    else:
        return int(arg) * batch_size

def main():
    args = parse_args()
    image_path = args.image_path
    label_path = args.label_path
    class_names = get_class_names(args.class_names)
    batch_size = int(args.batch_size)
    starting_file = get_starting_file(args.starting_file, batch_size)
    regions = get_regions(args.regions)
    anchors_path = args.anchors_path
    max_batches = int(args.max_batches)
    previous_train = args.load_previous_trained
    weights2load = args.weights2load

    anchors = get_anchors(anchors_path, regions)

    if previous_train == 'T':
        model_body, model = create_model(anchors, class_names, regions, load_pretrained=False, freeze_body=False)
        model.load_weights(weights2load)
    else:
        model_body, model = create_model(anchors, class_names, regions)

    if max_batches == 0:
        max_batches = get_max_batches(image_path, batch_size)

    processed_images, processed_labels = process_data(image_path, label_path, starting_file,
                                                      batch_size, regions)
    detectors_mask, matching_true_boxes = get_detector_mask(processed_labels, anchors, regions)
    model = initial_train(model, class_names, anchors, processed_images, processed_labels,
          detectors_mask, matching_true_boxes, regions)

    for i in range(1, max_batches):
        processed_images, processed_labels = process_data(image_path, label_path, starting_file+i*batch_size, batch_size, regions)
        detectors_mask, matching_true_boxes = get_detector_mask(processed_labels, anchors, regions)
        print('*'*10, 'Start {}th Training'.format(i), '*'*10)
        model = recur_train(model, class_names, anchors, processed_images, processed_labels,
              detectors_mask, matching_true_boxes, regions)

        if i % 10 == 0:
            draw(model_body, class_names, anchors, processed_images, i,
                image_set='val', weights_name='trained_stage_3_best.h5', save_all=False)
    # '''

if __name__ == "__main__":
    main()
    # image_path = '/media/frank/Storage/Project/Kaggle/WAD/input/train_color/'
    # label_path = '/media/frank/Storage/Project/Kaggle/WAD/input/train_label_txts/'
    # process_data(image_path, label_path, 0, 10)
