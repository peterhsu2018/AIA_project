import os

import numpy as np
import tensorflow as tf


def get_file_path(data_path='datasets/train/'):
    """get each data path and label"""
    img_paths = []
    labels = []
    class_dirs = sorted(os.listdir(data_path))
    if '.DS_Store' in class_dirs:
        class_dirs.remove('.DS_Store')

    dict_class2id = {}
    for label in range(len(class_dirs)):
        class_dir = class_dirs[label]
        dict_class2id[class_dir] = label
        class_path = os.path.join(data_path, class_dir)
        file_names = sorted(os.listdir(class_path))
        if '.DS_Store' in file_names:
            file_names.remove('.DS_Store')

        for file_name in file_names:
            file_path = os.path.join(class_path, file_name)
            img_paths.append(file_path)
            labels.append(label)
    return img_paths, labels


def pre_process(images, size):
    images = tf.image.random_flip_up_down(images)

    images = tf.image.random_flip_left_right(images)

    images = tf.image.random_brightness(images, max_delta=0.3)

    images = tf.image.random_contrast(images, 0.8, 1.2)

    new_size = tf.constant([size, size], dtype=tf.int32)

    images = tf.image.resize_images(images, new_size)

    return images


def get_batch(img_paths, labels, batch_size=32, height=128, width=128, channel=3, workers=4):
    # get image path from disk
    img_paths = np.asarray(img_paths)
    labels = np.asarray(labels)
    if len(img_paths) == 0:
        raise ValueError('No inputs!')

    image_paths = tf.convert_to_tensor(img_paths, dtype=tf.string)
    image_labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    # Build a TF Queue, shuffle data
    x_images, y_labels = tf.train.slice_input_producer(
        [image_paths, image_labels], shuffle=True)
    # Read images from disk
    x_images = tf.read_file(x_images)
    x_images = tf.image.decode_jpeg(x_images, channels=channel)
    x_images = pre_process(x_images, height)
    # # Resize images to a common size
    # x_images = tf.image.resize_images(x_images, [height, width])
    # Normalize
    x_images = 2 * (x_images / 255.0) - 1.0
    # Create batches
    X_batch, y_batch = tf.train.batch([x_images, y_labels], batch_size=batch_size,
                                      capacity=batch_size * workers,
                                      num_threads=4)

    return X_batch, y_batch
