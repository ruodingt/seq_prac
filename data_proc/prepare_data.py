import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
tf.enable_eager_execution()
graph = tf.Graph()

def dwonload_file():
    wkdir = os.path.abspath("")
    save_path = os.path.join(wkdir, 'data', 'jena_climate_2009_2016.csv.zip')

    _ = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname=save_path,
        extract=True)

    # csv_path, _ = os.path.splitext(zip_path)
    df = pd.read_csv(save_path)
    print(df.head())
    return df


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def univariate_data(dataset, start_index, end_index, history_size, target_pos):
    """

    :param dataset:
    :param start_index: reference index start, anything before reference index is deemed as history
    :param end_index: reference index end.
    :param history_size:
    :param target_pos: true target position = target_pos + reference_index
    :return:
    """
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_pos

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_pos])

    return np.array(data), np.array(labels)


TRAIN_SPLIT = 300000
tf.random.set_seed(13)


def explore(df):
    uni_data = df['T (degC)']
    uni_data.index = df['Date Time']
    uni_data.head()
    uni_data.plot(subplots=True)


def normalise_and_split(uni_data):
    uni_data = uni_data.values
    uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
    uni_train_std = uni_data[:TRAIN_SPLIT].std()
    uni_data = (uni_data - uni_train_mean) / uni_train_std


def show_example(x_train, y_train):
    print('Single window of past history')
    print(x_train[0])
    print('\n Target temperature to predict')
    print(y_train[0])
    show_plot([x_train[0], y_train[0]], 0, 'Sample Example')


def show_plot(plot_data, delta, title):
    def create_time_steps(length):
        time_steps = []
        for i in range(-length, 0, 1):
            time_steps.append(i)
        return time_steps

    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel('Time-Step')
    return plt


class Model:
    def __init__(self, input_shape):
        """

        :param input_shape: input_shape=x_train_uni.shape[-2:]
        """
        self.simple_lstm_model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(8, input_shape=input_shape),
            tf.keras.layers.Dense(1)
        ])

        self.simple_lstm_model.compile(optimizer='adam', loss='mae')


# class ModelTF:
#     def __init__(self):
#         self.graph = tf.Graph()
#         with self.graph.as_default():
#             # Prepare data shape to match `rnn` function requirements
#             # Current data input shape: (batch_size, timesteps, n_input)
#             # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
#
#             # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
#             x = tf.unstack(x, timesteps, 1)
#
#             # Define a lstm cell with tensorflow
#             lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
#
#             # Get lstm cell output
#             outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
#
#             # Linear activation, using rnn inner loop last output
#         return tf.matmul(outputs[-1], weights['out']) + biases['out']
#         pass
#
# def my_model_fn(features, labels, mode):
#   predictions = ...
#   loss = ...
#   train_op = ...
#   return tf.estimator.EstimatorSpec(
#       mode=mode,
#       predictions=predictions,
#       loss=loss,
#       train_op=train_op)



