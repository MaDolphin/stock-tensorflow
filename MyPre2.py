import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Initial Variable
TIME_STEPS = 20
INPUT_SIZE = 7
OUTPUT_SIZE = 1
BATCH_SIZE = 50
CELL_SIZE = 10
LR = 0.0006


def load_data(file_path, col_begin=2, col_end=10):
    f = open(file_path)
    df = pd.read_csv(f)  # read stock data
    data = df.iloc[:, col_begin: col_end].values
    return data


# 获取训练集
def get_train_data(file_path, batch_size, time_step, train_begin, train_end):
    data = load_data(file_path)
    batch_index = []
    data_train = data[train_begin: train_end]
    mean = np.mean(data_train, axis=0)
    std = np.std(data_train, axis=0)
    normalized_train_data = (data_train - mean) / std  # normalize train data
    train_x, train_y = [], []  # TrainSet
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i: i + time_step, : INPUT_SIZE]
        y = normalized_train_data[i: i + time_step, INPUT_SIZE, np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))
    return batch_index, train_x, train_y


# 获取测试集
def get_test_data(file_path, time_step, test_begin):
    data = load_data(file_path)
    data_test = data[test_begin:]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    normalized_test_data = (data_test - mean) / std  # normalize test data
    size = (len(normalized_test_data) + time_step - 1) // time_step  # the size of sample
    test_x, test_y = [], []  # TestSet
    for i in range(size - 1):
        x = normalized_test_data[i * time_step:(i + 1) * time_step, :INPUT_SIZE]
        y = normalized_test_data[i * time_step:(i + 1) * time_step, INPUT_SIZE]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((normalized_test_data[(i + 1) * time_step:, :INPUT_SIZE]).tolist())
    test_y.extend((normalized_test_data[(i + 1) * time_step:, INPUT_SIZE]).tolist())
    return mean, std, test_x, test_y


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
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0,
                                                 state_is_tuple=True)  # if forget_bias = 1.0 not forget

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

        # with tf.name_scope('cost'):
        #     self.cost = tf.reduce_mean(
        #         tf.square(
        #             tf.reshape(self.pred, [-1]) - tf.reshape(self.xs, [-1])
        #         ))

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
        # print('***********', tf.subtract(y_pre, y_target))
        # return tf.square(tf.subtract(y_pre, y_target))
        return tf.subtract(y_pre, y_target)

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1., )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


# ——————————————————训练模型——————————————————
def train_lstm(file_path, batch_size=BATCH_SIZE, time_step=TIME_STEPS, train_begin=1000, train_end=5700):
    batch_index, train_x, train_y = get_train_data(file_path, batch_size, time_step, train_begin, train_end)
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs", sess.graph)

        for i in range(1000):
            for step in range(len(batch_index) - 1):
                if i == 0:
                    feed_dict = {
                        model.xs: train_x[batch_index[step]:batch_index[step + 1]],
                        model.ys: train_y[batch_index[step]:batch_index[step + 1]]
                        # create initial state
                    }
                else:
                    feed_dict = {
                        model.xs: train_x[batch_index[step]:batch_index[step + 1]],
                        model.ys: train_y[batch_index[step]:batch_index[step + 1]],
                        model.cell_init_state: state  # use last state as the initial state for this run
                    }

                _, cost, state, pred = sess.run(
                    [model.train_op, model.cost, model.cell_final_state, model.pred],
                    feed_dict=feed_dict)
            if i % 20 == 0:
                print(i, 'cost: ', round(cost, 6))
                result = sess.run(merged, feed_dict)
                writer.add_summary(result, i)
        save_path = saver.save(sess, "my_net/pre_net2.ckpt")
        print("Save to path: ", save_path)


# ————————————————预测模型————————————————————
def prediction(file_path, time_step=TIME_STEPS, test_begin=5700):
    mean, std, test_x, test_y = get_test_data(file_path, time_step, test_begin)
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "my_net/pre_net2.ckpt")
        test_predict = []
        for step in range(len(test_x) - 1):
            prob = sess.run([model.pred], feed_dict={model.xs: [test_x[step]]})
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        test_y = np.array(test_y) * std[INPUT_SIZE] + mean[INPUT_SIZE]
        test_predict = np.array(test_predict) * std[INPUT_SIZE] + mean[INPUT_SIZE]
        acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  # 偏差
        print(acc)
        # 以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(test_y))), test_y, color='r')
        plt.legend(['predict', 'true'])
        plt.show()


train_lstm(file_path="dataset.csv")
# prediction(file_path="dataset.csv")
