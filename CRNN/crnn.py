import os
import sys
import time
import random
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from scipy.misc import imread, imresize, imsave
from tensorflow.contrib import rnn

from data_manager import DataManager
from utils import sparse_tuple_from, resize_image, label_to_array, ground_truth_to_word, levenshtein

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class CRNN(object):
    def __init__(self, batch_size, model_path, examples_path, max_image_width, image_height, train_test_ratio, restore, char_set_string, test_augment_image, learning_rate, blank_symbol, augmentor):
        self.step = 0
        self.CHAR_VECTOR = char_set_string
        self.NUM_CLASSES = len(self.CHAR_VECTOR) + 1

        print(f"CHAR_VECTOR {self.CHAR_VECTOR}")
        print(f"NUM_CLASSES {self.NUM_CLASSES}")

        self.__start_learning_rate = learning_rate
        self.__learning_rate = tf.placeholder(tf.float32)
        self.__model_path = model_path
        self.__save_path = os.path.join(model_path, 'ckp')

        self.__restore = restore

        self.__training_name = str(int(time.time()))
        self.__session = tf.Session()

        # Building graph
        with self.__session.as_default():
            (
                self.__inputs,
                self.__targets,
                self.__seq_len,
                self.__logits,
                self.__decoded,
                self.__optimizer,
                self.__acc,
                self.__cost,
                self.__max_char_count,
                self.__init,
                self.__output_len,
                self.__cost_len
            ) = self.crnn(max_image_width, image_height, char_set_string)
            self.__init.run()

        with self.__session.as_default():
            self.__saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
            # Loading last save if needed
            if self.__restore:
                print('Restoring')
                ckpt = tf.train.latest_checkpoint(self.__model_path)
                if ckpt:
                    print('Checkpoint is valid')
                    self.step = int(ckpt.split('-')[1])
                    self.__saver.restore(self.__session, ckpt)

        # Creating data_manager
        self.__data_manager = DataManager(batch_size, model_path, examples_path, max_image_width, image_height, train_test_ratio, self.__max_char_count, self.CHAR_VECTOR, test_augment_image, augmentor)

    def crnn(self, max_width, height, char_set_string):
        def BidirectionnalRNN(inputs, seq_len):
            """
                Bidirectionnal LSTM Recurrent Neural Network part
            """

            with tf.variable_scope(None, default_name="bidirectional-rnn-1"):
                # Forward
                lstm_fw_cell_1 = rnn.BasicLSTMCell(256)
                # Backward
                lstm_bw_cell_1 = rnn.BasicLSTMCell(256)

                inter_output, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_1, lstm_bw_cell_1, inputs, seq_len, dtype=tf.float32)

                inter_output = tf.concat(inter_output, 2)

            with tf.variable_scope(None, default_name="bidirectional-rnn-2"):
                # Forward
                lstm_fw_cell_2 = rnn.BasicLSTMCell(256)
                # Backward
                lstm_bw_cell_2 = rnn.BasicLSTMCell(256)

                outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_2, lstm_bw_cell_2, inter_output, seq_len, dtype=tf.float32)

                outputs = tf.concat(outputs, 2)


            return outputs

        def CNN(inputs):
            """
                Convolutionnal Neural Network part
            """
            ratio = height / 32
            char_set_constant = tf.constant(char_set_string)

            # 64 / 3 x 3 / 1 / 1
            conv1 = tf.layers.conv2d(inputs=inputs, filters = 64, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

            # 2 x 2 / 1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=[2, 2])

            # 128 / 3 x 3 / 1 / 1
            conv2 = tf.layers.conv2d(inputs=pool1, filters = 128, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

            # 2 x 2 / 1
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            # 256 / 3 x 3 / 1 / 1
            conv3 = tf.layers.conv2d(inputs=pool2, filters = 256, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

            # Batch normalization layer
            bnorm1 = tf.layers.batch_normalization(conv3)

            # 256 / 3 x 3 / 1 / 1
            conv4 = tf.layers.conv2d(inputs=bnorm1, filters = 256, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

            # 1 x 2 / 1
            pool3 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=[1, 2], padding="same")

            # 512 / 3 x 3 / 1 / 1
            conv5 = tf.layers.conv2d(inputs=pool3, filters = 512, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

            # Batch normalization layer
            bnorm2 = tf.layers.batch_normalization(conv5)

            # 512 / 3 x 3 / 1 / 1
            conv6 = tf.layers.conv2d(inputs=bnorm2, filters = 512, kernel_size = (3, 3), padding = "same", activation=tf.nn.relu)

            # 1 x 2 / 2
            pool4 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2*ratio, 2*ratio], strides=[1, 2*ratio], padding="same")

            # 512 / 2 x 2 / 1 / 0
            conv7 = tf.layers.conv2d(inputs=pool4, filters = 512, kernel_size = (2, 2), padding = "valid", activation=tf.nn.relu)

            return conv7

        inputs = tf.placeholder(tf.float32, [None, max_width, height, 1], name="input")

        # Our target output
        targets = tf.sparse_placeholder(tf.int32, name='targets')

        # The length of the sequence
        seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

        cnn_output = CNN(inputs)
        reshaped_cnn_output = tf.squeeze(cnn_output, [2])
        max_char_count = cnn_output.get_shape().as_list()[1]
        batch_size = tf.shape(cnn_output)[0]
        seq_len2 = tf.fill([batch_size], max_char_count)

        print("Shape from CNN:", cnn_output.get_shape().as_list())
        print("MAX char count that can be detected:", max_char_count)

        crnn_model = BidirectionnalRNN(reshaped_cnn_output, seq_len2)#seq_len)

        logits = tf.reshape(crnn_model, [-1, 512])
        W = tf.Variable(tf.truncated_normal([512, self.NUM_CLASSES], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0., shape=[self.NUM_CLASSES]), name="b")

        logits = tf.matmul(logits, W) + b
        logits = tf.reshape(logits, [tf.shape(cnn_output)[0], max_char_count, self.NUM_CLASSES])

        # Final layer, the output of the BLSTM
        logits = tf.transpose(logits, (1, 0, 2))

        # Loss and cost calculation
        loss = tf.nn.ctc_loss(targets, logits, seq_len2)# seq_len)
        # loss = tf.nn.ctc_loss_v2(targets, logits, seq_len, seq_len)

        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len2, merge_repeated=False)
        dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1, name='dense_decoded')

        length_size =  tf.fill([batch_size], 2.0)
        output_length = tf.cast(tf.count_nonzero(tf.greater_equal(tf.cast(dense_decoded, tf.float32), 0.0), 1), tf.float32)
        #bad_length = tf.identity(tf.cast(tf.shape(dense_decoded)[1], tf.float32))
        #cost_len = 0.5 * tf.reduce_sum(tf.abs(length_size-output_length), name="cost_len")
        cost_len = tf.reduce_mean(output_length)

        # USE this if your length output sequence is fixed
        cost = tf.reduce_mean(loss) # + cost_len
        #cost = tf.reduce_mean(loss)

        # Training step
        optimizer = tf.train.AdamOptimizer(learning_rate=self.__learning_rate).minimize(cost)
        #optimizer = tf.train.RMSPropOptimizer(learning_rate=self.__learning_rate).minimize(cost)

        # The error rate
        acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

        init = tf.global_variables_initializer()

        return inputs, targets, seq_len2, logits, dense_decoded, optimizer, acc, cost, max_char_count, init, output_length, cost_len

    def train(self, iteration_count, learning_rate_decay):
        with self.__session.as_default():
            print('Training')

            for i in range(iteration_count):
                iter_loss, len_loss, k = 0, 0, 0

                if learning_rate_decay != 0 and i != 0 and i % learning_rate_decay == 0:
                    self.__start_learning_rate *= 0.1

                pbar = tqdm(total=len(self.__data_manager.train_batches), ncols=150)
                for batch_y, batch_dt, batch_x in self.__data_manager._generate_all_train_batches():
                    op, decoded, loss_value, acc, logits, out_len, cost_len = self.__session.run(
                        [self.__optimizer, self.__decoded, self.__cost, self.__acc, self.__logits, self.__output_len, self.__cost_len],
                        feed_dict={
                            self.__inputs: batch_x,
                            #self.__seq_len: [self.__max_char_count] * self.__data_manager.batch_size,
                            self.__targets: batch_dt,
                            self.__learning_rate: self.__start_learning_rate
                        }
                    )

                    #print('cost', out_len, cost_len)

                    if k > len(self.__data_manager.train_batches)-5:
                        for j in range(2):
                            print('GT:', batch_y[j])
                            print('PREDICT:', ground_truth_to_word(decoded[j], self.CHAR_VECTOR))
                            print(f'---- {i} ----')
                            sys.stdout.flush()

                    pbar.update(1)
                    k += 1
                    iter_loss += loss_value
                    len_loss += cost_len

                    pbar.set_postfix(
                      epoch=str(i)+'/'+str(iteration_count),
                      step=str(k),
                      cost="{:.2f}".format(iter_loss/float(k)),
                      cost_len="{:.2f}".format(len_loss/float(k)),
                      lr=str(self.__start_learning_rate)
                    )


                self.__saver.save(
                    self.__session,
                    self.__save_path,
                    global_step=self.step
                )

                self.save_frozen_model("save/frozen.pb")

                print('[{}] Iteration loss: {} Error rate: {}'.format(self.step, iter_loss, acc))
                self.step += 1
        return None

    def test(self):
        with self.__session.as_default():
            print('Testing')
            for batch_y, _, batch_x in self.__data_manager.test_batches:
                decoded = self.__session.run(
                    self.__decoded,
                    feed_dict={
                        self.__inputs: batch_x,
                        #self.__seq_len: [self.__max_char_count] * self.__data_manager.batch_size
                    }
                )

                for i, y in enumerate(batch_y):
                    print(batch_y[i])
                    print(ground_truth_to_word(decoded[i], self.CHAR_VECTOR))
        return None

    def save_frozen_model(self, path=None, optimize=False, input_nodes=["input", "seq_len"], output_nodes=["dense_decoded"]):
        if not path or len(path) == 0:
            raise ValueError("Save path for frozen model is not specified")

        tf.train.write_graph(self.__session.graph_def, "/".join(path.split("/")[0:-1]), path.split("/")[-1] + ".pbtxt")

        # get graph definitions with weights
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            self.__session,  # The session is used to retrieve the weights
            self.__session.graph.as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_nodes,  # The output node names are used to select the usefull nodes
        )

        # optimize graph
        if optimize:
            output_graph_def = optimize_for_inference_lib.optimize_for_inference(
                output_graph_def, input_nodes, output_nodes, tf.float32.as_datatype_enum
            )

        with open(path, "wb") as f:
            f.write(output_graph_def.SerializeToString())

        return True
