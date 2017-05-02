import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Iinitial Variable
TIME_STEPS = 20
INPUT_SIZE = 1
OUTPUT_SIZE = 1
BATCH_SIZE = 50
CELL_SIZE = 10
LR = 0.0006


# Import DataSet
def load_data(filePath="data/dataset.csv"):
    f = open(filePath)
    df = pd.read_csv(f)
    data = np.array(df['high'])
    data = data[::-1]
    return data


# Create TestSet
def crate_testset(data):
    mean = np.mean(data)
    std = np.std(data)
    normalize_data = (data - mean) / std  # 标准化
    normalize_data = normalize_data[:, np.newaxis]  # 增加维度
    train_x, train_y = [], []
    for i in range(len(normalize_data) - TIME_STEPS - 1):
        x = normalize_data[i:i + TIME_STEPS]
        y = normalize_data[i + 1:i + TIME_STEPS + 1]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    return train_x, train_y


class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size

        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, TIME_STEPS, input_size])  # 每批次输入网络的tensor
            self.ys = tf.placeholder(tf.float32, [None, TIME_STEPS, output_size])  # 每批次tensor对应的标签
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    def add_input_layer(self):
        # shape = (batch * n_steps, cell_size)
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='to_2D')

        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])

        # bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size, ])

        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in

        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='to_3D')

    def add_cell(self):
        lstm_cell_single = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0,
                                                 state_is_tuple=True)  # if forget_bias = 1.0 not forget
        lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell_single] * 10)

        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)

        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

    def add_output_layer(self):
        # shape = (batch * n_steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='to_2D')

        # Ws (cell_size, output_size)
        Ws_out = self._weight_variable([self.cell_size, self.output_size])

        # bs (output_size)
        bs_out = self._bias_variable([self.output_size, ])

        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

    def compute_cost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.summary.scalar('cost', self.cost)

    def ms_error(self, y_pre, y_target):
        return tf.square(tf.subtract(y_pre, y_target))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1., )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


def train_lstm():
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()

    filePath = "data/dataset.csv"
    data = load_data(filePath)
    train_x, train_y = crate_testset(data)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    for i in range(50):
        step = 0
        start = 0
        end = start + BATCH_SIZE
        while (end < len(train_x)):
            if i == 0:
                feed_dict = {
                    model.xs: train_x[start:end],
                    model.ys: train_y[start:end],
                    # create initial state
                }
            else:
                feed_dict = {
                    model.xs: train_x[start:end],
                    model.ys: train_y[start:end],
                    model.cell_init_state: state  # use last state as the initial state for this run
                }

            _, cost, state, pred = sess.run(
                [model.train_op, model.cost, model.cell_final_state, model.pred],
                feed_dict=feed_dict)

            start += BATCH_SIZE
            end = start + BATCH_SIZE

            if step % 10 == 0:
                print(i, step, ' cost: ', round(cost, 6))
                result = sess.run(merged, feed_dict)
                writer.add_summary(result, i)
            step += 1
    save_path = saver.save(sess, "my_net/my_pre_net1.ckpt")
    print("Save to path: ", save_path)


train_lstm()
