from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from tensorflow.python.platform import tf_logging as logging

from lib.data_generator import get_file_path, get_batch
from model import inception_v4, inception_v4_arg_scope
from options import Options

plt.switch_backend('agg')

slim = tf.contrib.slim


def getDataSetPathsAndLables(evalType, dataroot='datasets/test/', saveValidateSetPath=''):
    if evalType == 'test':
        img_paths, labels = get_file_path(dataroot)
    elif evalType == 'validate':
        df = pd.read_csv(saveValidateSetPath)
        img_paths, labels = np.array(
            df.iloc[:, 0]), np.array(df.iloc[:, 1])

    else:
        raise Exception('type name must be "test" or "validate"!')

    return img_paths, labels


def run():
    # ARGUMENTS
    opt = Options().parse()
    log_dir = opt.log_dir
    dataroot = opt.dataroot
    outputDir = opt.outf
    modelName = opt.model
    batchSize = opt.batchsize
    imsize = opt.imsize
    nc = opt.nc
    workers = opt.workers
    eval_type = opt.eval_type
    saveValidateSetPath = os.path.join(
        outputDir, modelName, 'train', 'validateset.csv')

    # Create a new evaluation log directory to visualize the validation process
    log_eval = log_dir + '/log_eval_test'
    checkpoint_file = tf.train.latest_checkpoint(log_dir)
    # State the number of epochs to evaluate
    num_epochs = 1

    if not os.path.exists(log_eval):
        os.makedirs(log_eval)

    x_valid_paths, y_valid_labels = getDataSetPathsAndLables(eval_type, dataroot, saveValidateSetPath)

    n_class = len(set(y_valid_labels))

    totalLabels = []
    totalPredictions = []

    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        X_valid_batch, y_valid_batch = get_batch(x_valid_paths,
                                                 y_valid_labels,
                                                 batch_size=batchSize,
                                                 height=imsize,
                                                 width=imsize,
                                                 channel=nc,
                                                 workers=workers)

        num_batches_per_epoch = int(len(x_valid_paths) / batchSize)
        # Because one step is one batch processed
        num_steps_per_epoch = num_batches_per_epoch
        totalSteps = num_steps_per_epoch * num_epochs

        # Now create the inference model but set is_training=False
        with slim.arg_scope(inception_v4_arg_scope()):
            _, end_points = inception_v4(
                X_valid_batch, n_class, is_training=False)

        # # #get all the variables to restore from the checkpoint file and create the saver function to restore
        variables_to_restore = slim.get_variables_to_restore()

        # Just define the metrics to track without the loss or whatsoever
        predictions = tf.argmax(end_points['Predictions'], 1)
        accuracy, accuracy_update = tf.metrics.accuracy(
            y_valid_batch, predictions)
        metrics_op = tf.group(accuracy_update)

        # Create the global step and an increment op for monitoring
        global_step = tf.train.get_or_create_global_step()
        # no apply_gradient method so manually increasing the global_step
        global_step_op = tf.assign(global_step, global_step + 1)

        # Create a evaluation step function
        def eval_step(sess, metrics_op, global_step):
            '''
            Simply takes in a session, runs the metrics op and some logging information.
            '''
            start_time = time.time()
            _, global_step_count, accuracy_value, _labels, _predictions = sess.run(
                [metrics_op, global_step, accuracy, y_valid_batch, predictions])
            totalLabels.append(_labels)
            totalPredictions.append(_predictions)
            time_elapsed = time.time() - start_time

            # Log some information
            logging.info('Global Step %s: Streaming Accuracy: %.4f (%.2f sec/step)',
                         global_step_count, accuracy_value, time_elapsed)

            return accuracy_value

        # Define some scalar quantities to monitor
        tf.summary.scalar('Validation_Accuracy', accuracy)
        my_summary_op = tf.summary.merge_all()

        saver = tf.train.Saver(variables_to_restore)
        saver_hook = tf.train.CheckpointSaverHook(log_eval,
                                                  save_steps=10,
                                                  saver=saver)

        summary_hook = tf.train.SummarySaverHook(save_steps=10,
                                                 summary_op=my_summary_op)

        accuracy_value = None

        # Now we are ready to run in one session
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=log_eval,
                hooks=[saver_hook, summary_hook, tf.train.StopAtStepHook(last_step=totalSteps)]) as sess:
            saver.restore(sess, checkpoint_file)
            while not sess.should_stop():
                tf.train.start_queue_runners(sess=sess)
                for step in range(totalSteps):
                    sess.run(global_step)
                    # print vital information every start of the epoch as always
                    if step % num_batches_per_epoch == 0:
                        logging.info('Epoch: %s/%s', int(step /
                                                         num_batches_per_epoch + 1), num_epochs)
                        logging.info(
                            'Current Streaming Accuracy: %.4f', sess.run(accuracy))

                    accuracy_value = eval_step(sess, metrics_op=metrics_op,
                                               global_step=global_step_op)

                # At the end of all the evaluation, show the final accuracy
                logging.info('Final Streaming Accuracy: %.4f', accuracy_value)

                # Now we want to visualize the last batch's images just to see what our model has predicted
                # image, _labels, _predictions = sess.run([X_valid_batch, y_valid_batch, predictions])

                #             for i in range(10):
                #                 image, label, prediction = images[i], _labels[i], _predictions[i]
                #                 prediction_class = class_dict[prediction]
                #                 label_name = class_dict[label]
                #                 text = 'Prediction: ' + prediction_class + ',Ground Truth: ' + label_name
                #                 print('image',type(image))
                #                 img_plot = plt.imshow(image)

                #                 # Set up the plot and hide axes
                #                 plt.title(text)
                #                 img_plot.axes.get_yaxis().set_ticks([])
                #                 img_plot.axes.get_xaxis().set_ticks([])
                #                 plt.show()

                logging.info(
                    'Model evaluation has completed! Visit TensorBoard for more information regarding your evaluation.')
    return totalLabels, totalPredictions


if __name__ == '__main__':
    opt = Options().parse()
    n_class = opt.n_class
    if n_class == 2:
        categorie = ['abnormal', 'normal']
    elif n_class == 5:
        categorie = ['black', 'broken', 'fungus', 'insect', 'normal']
    else:
        raise ValueError('Only support to 2 or 5!')

    labels, predictions = run()
    labels = [categorie[v] for a in labels for v in a]
    predictions = [categorie[v] for a in predictions for v in a]
    print('labels shape: ', np.array(labels).shape)
    print('predictions shape: ', np.array(predictions).shape)
    print(pd.crosstab(np.array(labels), np.array(predictions),
                      rownames=['label'], colnames=['predict']))
    print(metrics.classification_report(np.array(labels), np.array(predictions)))
