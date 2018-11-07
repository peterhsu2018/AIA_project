import matplotlib.pyplot as plt
import tensorflow as tf

slim = tf.contrib.slim
import options
from model import inception_v4, inception_v4_arg_scope
import os
import datetime
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_v3 import preprocess_input
import time


def get_image(data_path, image_size):
    img = load_img(data_path, target_size=(image_size, image_size))
    tr_x = img_to_array(img)
    tr_x = tr_x[:, :, ::-1]
    tr_x = preprocess_input(tr_x)

    return np.array(tr_x)


def get_one_image_content(path, size):
    if len(path) < 1:
        raise Exception('check the image path!')

    image = get_image(path, size)
    # Normalize
    # image = 2 * (image / 255.0) - 1.0
    return image


def predict(categorie, image_path, predict_dir, log_dir, meta_graph_name, size, n_class, restore_file):
    input_image = tf.placeholder(tf.float32, shape=[None, size, size, channel])
    with slim.arg_scope(inception_v4_arg_scope()):
        logits, end_points = inception_v4(input_image, num_classes=n_class, is_training=False)
    predictions = tf.argmax(end_points['Predictions'], 1)

    with tf.Session() as sess:
        start_time1 = time.time()
        saver = tf.train.import_meta_graph(os.path.join(log_dir, meta_graph_name))
        print('load meta_graph spend {:0.4f} s'.format(time.time() - start_time1))
        start_time1 = time.time()
        saver.restore(sess, restore_file)
        print('load saver.restore spend {:0.4f} s'.format(time.time() - start_time1))
        sess.run(tf.global_variables_initializer())
        image = get_one_image_content(image_path, size)
        # plt.imshow(image)
        image = np.expand_dims(image, axis=0)
        start_time2 = time.time()
        result = sess.run(predictions, feed_dict={input_image: image})[0]
        print('predict spend {:0.4f} s'.format(time.time() - start_time2))
        # filename = os.path.join(predict_dir, 'predict={}-{}.jpg'.format(categorie[result],
        #                                                                 datetime.datetime.today().strftime(
        #                                                                     '%Y-%m-%d__%M:%S')))
        #
        # plt.savefig(filename)

        return result


if __name__ == '__main__':
    categorie = []
    opt = options.Options().parse()
    log_dir = opt.log_dir
    image_path = opt.image_path
    start_time = time.time()
    checkpoint_file = tf.train.latest_checkpoint(log_dir)
    meta_graph_name = opt.meta_graph_name
    outputDir = opt.outf
    channel = opt.nc
    size = opt.imsize
    n_class = opt.n_class
    if n_class == 2:
        categorie = ['abnormal', 'normal']
    elif n_class == 5:
        categorie = ['black', 'broken', 'fungus', 'insect', 'normal']
    else:
       raise ValueError('Only support to 2 or 5!')

    print('load check point spend {:0.4f} s'.format(time.time() - start_time))
    predict_dir = os.path.join(outputDir, 'predict')
    if not os.path.isdir(predict_dir):
        os.makedirs(predict_dir)

    predict = predict(categorie, image_path, predict_dir, log_dir, meta_graph_name, size, n_class,
                                    checkpoint_file)
    print('predict to {}'.format(categorie[predict]))



