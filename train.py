"""
Contains the definition of the Inception V4 architecture.
As described in http://arxiv.org/abs/1602.07261.
  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi

TRAIN 
Example: Run the following command from the terminal.
    run python train.py \
        --batchsize 64 \
        --imsize 229
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from options import Options
from model import inception_v4, inception_v4_arg_scope
from tensorflow.python.platform import tf_logging as logging
from lib.data_generator import get_file_path, get_batch
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
import os
import pandas as pd

slim = tf.contrib.slim


def main():
    """ Training
    """
    ##
    # ARGUMENTS
    opt = Options().parse()
    log_dir = opt.log_dir
    checkpoint_file = log_dir + '/pre_trained/inception_v4.ckpt'
    # Learning rate information and configuration (Up to you to experiment)
    initial_learning_rate = opt.lr
    epochs = opt.epochs
    dataroot = opt.dataroot
    outputDir = opt.outf
    modelName = opt.model
    batchSize = opt.batchsize
    is_train = opt.is_train
    imsize = opt.imsize
    nc = opt.nc
    workers = opt.workers
    learning_rate_decay_factor = 0.7
    num_epochs_before_decay = 2

    # set gpu ids
    if opt.device == 'gpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids[0]
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)
        # TRAIN MODEL
        img_paths, labels = get_file_path(dataroot)

        n_class = len(set(labels))

        x_train_paths, x_valid_paths, y_train_labels, y_valid_labels = train_test_split(
            img_paths, labels, test_size=0.2, stratify=labels)

        saveValidateSetPath = os.path.join(
            outputDir, modelName, 'train', 'validateset.csv')
        df = pd.DataFrame({'path': x_valid_paths, 'label': y_valid_labels})
        df.to_csv(saveValidateSetPath, index=False)

        X_train_batch, y_train_batch = get_batch(x_train_paths,
                                                 y_train_labels,
                                                 batch_size=batchSize,
                                                 height=imsize,
                                                 width=imsize,
                                                 channel=nc,
                                                 workers=workers)

        num_batches_per_epoch = int(len(x_train_paths) / batchSize)
        # Because one step is one batch processed
        num_steps_per_epoch = num_batches_per_epoch
        decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)
        totalSteps = num_steps_per_epoch * epochs

        with slim.arg_scope(inception_v4_arg_scope()):
            logits, end_points = inception_v4(
                X_train_batch, n_class, is_training=is_train)

        exclude = ['InceptionV4/AuxLogits', 'InceptionV4/Logits', 'InceptionV4/Mixed_7']
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

        one_hot_labels = slim.one_hot_encoding(y_train_batch, n_class)

        loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels,
                                               logits=logits)

        total_loss = tf.losses.get_total_loss()

        global_step = tf.train.get_or_create_global_step()

        lr = tf.train.exponential_decay(
            learning_rate=initial_learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=learning_rate_decay_factor,
            staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        train_op = slim.learning.create_train_op(total_loss=total_loss,
                                                 optimizer=optimizer)

        predictions = tf.argmax(end_points['Predictions'], 1)

        probabilities = end_points['Predictions']

        accuracy, accuracy_update = tf.metrics.accuracy(
            y_train_batch, predictions)

        metrics_op = tf.group(accuracy_update, probabilities)

        # Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learning_rate', lr)
        my_summary_op = tf.summary.merge_all()

        saver = tf.train.Saver(variables_to_restore)
        saver_hook = tf.train.CheckpointSaverHook(log_dir,
                                                  save_steps=10,
                                                  saver=saver)

        summary_hook = tf.train.SummarySaverHook(save_steps=10,
                                                 summary_op=my_summary_op)

        # Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
        def train_step(sess, train_op, global_step):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            '''
            # Check the time for each sess run
            start_time = time.time()
            total_loss, global_step_count, accuracy_value, _ = sess.run(
                [train_op, global_step, accuracy, metrics_op])
            time_elapsed = time.time() - start_time

            # Run the logging to print some results
            logging.info('global step %s: loss: %.4f (%.2f sec/step)',
                         global_step_count, total_loss, time_elapsed)

            return total_loss, accuracy_value

        logging.info('end_points\n %s', end_points)

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        accuracy_value = None
        with tf.Session() as sess:
            sess.run(init)
            
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=log_dir,
                hooks=[saver_hook, summary_hook, tf.train.StopAtStepHook(last_step=totalSteps)]) as sess:
            saver.restore(sess, checkpoint_file)
            while not sess.should_stop():
                tf.train.start_queue_runners(sess=sess)

                for step in range(totalSteps):
                    # At the start of every epoch, show the vital information:
                    if step % num_batches_per_epoch == 0:
                        logging.info('Epoch %s/%s', step /
                                     num_batches_per_epoch + 1, epochs)
                        logging.info('Current Streaming Accuracy: %s', sess.run([accuracy]))

                    loss, accuracy_value = train_step(
                        sess, train_op, global_step)

                # We log the final training loss and accuracy
                logging.info('Final Training Loss: %s', loss)
                logging.info('Final Training Accuracy: %s', accuracy_value)

                # Once all the training has been done, save the log files and checkpoint model
                logging.info('Finished training! Saving model to disk now.')


if __name__ == '__main__':
    main()
