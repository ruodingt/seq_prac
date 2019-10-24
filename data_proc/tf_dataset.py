import os

import pandas as pd
import tensorflow as tf
import numpy as np
import re

# tf.enable_eager_execution()
wkdir = os.path.dirname(os.path.abspath(""))
variable_name = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']
variable_name = ['T (degC)']

BATCH_SIZE = 512
BUFFER_SIZE = 100000
HISTORY_SIZE = 20
TARGET_SIZE = 1
TRAIN_SPLIT = 300000
GRADIENT_CLIP = ''


class DatasetSeq:
    def __init__(self, wkdir, variable_name, history_size, target_size, target_offset=0, shift=1, stride=1):
        """

        :param wkdir:
        :param fp:
        :param variable_name: variable_name = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']
        :param target_size:
        :param target_offset:
        :param stride:
        :param history_size:
        """

        self.shift = shift
        self.stride = stride

        self.history_size = history_size
        self.target_offset = target_offset
        self.target_size = target_size

        self.wkdir = wkdir

        pass

    def run_etl(self):
        fp = self._download_file_and_convert_to_df()

        _df = self.extract(fp=fp, variable_name=variable_name)

        seq_dataset = self.transform(_df)
        return seq_dataset

    def _download_file_and_convert_to_df(self):
        save_path = os.path.join(self.wkdir, 'data', 'jena_climate_2009_2016.csv.zip')

        _ = tf.keras.utils.get_file(
            origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
            fname=save_path,
            extract=True)
        return save_path

    @staticmethod
    def extract(fp, variable_name) -> pd.DataFrame:
        """
        Extract .csv to pandas dataframe
        :param fp:
        :param variable_name:
        :return:
        """
        df = pd.read_csv(fp)
        print(df.head())

        _df = df[variable_name]
        _df.index = df['Date Time']

        return _df

    def transform(self, _df: pd.DataFrame) -> tf.data.Dataset:
        # cast to float32
        _df_value = _df.values.astype(np.float32)
        # normalise
        _df_normalised = self._to_normalised(_df_value)
        # to tf Dataset
        _dataset = tf.data.Dataset.from_tensor_slices((_df_normalised,))
        # to seq/label pair dataset
        seq_dataset = self.make_seq_label_dataset(dataset=_dataset)
        return seq_dataset

    def _to_normalised(self, _df_value):
        _mean = _df_value[:TRAIN_SPLIT].mean(axis=0)
        _std = _df_value[:TRAIN_SPLIT].std(axis=0)
        _df_normalised = (_df_value - _mean) / _std
        return _df_normalised

    def make_seq_label_dataset(self, dataset):
        """

        the total length of the window should be len(history_size + target_offset + target_size)
        :param dataset:
        :return:
        """
        window_size = self.history_size + self.target_offset + self.target_size
        wd_dataset = self._make_window_dataset(dataset, window_size=window_size, shift=1, stride=1)
        seq_label_dataset = wd_dataset.map(self._seq_label_mapper)

        return seq_label_dataset

    def _seq_label_mapper(self, window):
        return window[:self.history_size], window[self.history_size + self.target_offset:]

    def _make_window_dataset(self, ds, window_size=5, shift=1, stride=1):
        """
        For a seq range(N), it gives dataset:
        [0 1 2 3 4]
        [1 2 3 4 5]
        [2 3 4 5 6]
        [3 4 5 6 7]
        [4 5 6 7 8]
        .....

        ---- e.g.
        range_ds = tf.data.Dataset.range(100000)
        ds = _make_window_dataset(range_ds, window_size=10, shift = 5, stride=3)
        for example in ds.take(10):
            print(example.numpy())


        [*] Output:
        [ 0  3  6  9 12 15 18 21 24 27]
        [ 5  8 11 14 17 20 23 26 29 32]
        [10 13 16 19 22 25 28 31 34 37]
        [15 18 21 24 27 30 33 36 39 42]
        [20 23 26 29 32 35 38 41 44 47]
        [25 28 31 34 37 40 43 46 49 52]
        [30 33 36 39 42 45 48 51 54 57]
        [35 38 41 44 47 50 53 56 59 62]
        [40 43 46 49 52 55 58 61 64 67]
        [45 48 51 54 57 60 63 66 69 72]


        ----
        :param ds: Dataset Obj, a single/multivariate time series sequence
        :param window_size: length of each sub-sequence
        :param shift: incremental offset, applies to each sequence start
        :param stride: skip every $stride data points when taking the point in the overall sequence
        :return:
        """

        windows = ds.window(window_size, shift=shift, stride=stride)

        def sub_to_batch(sub):
            return sub.batch(window_size, drop_remainder=True)

        # windows is a Dataset of Datasets, needs flat_map to convert
        windows2 = windows.flat_map(sub_to_batch)
        return windows2


# def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
#   def input_function():
#     ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
#     if shuffle:
#       ds = ds.shuffle(1000)
#     ds = ds.batch(batch_size).repeat(num_epochs)
#     return ds
#   return input_function
#
# train_input_fn = make_input_fn(dftrain, y_train)
# eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

def model_func_builder():
    def model_fn(features, labels, mode):
        model_seq = ModelSeq(features, labels, mode, hidden_units=(8,), out_unit=1)
        if (mode == tf.estimator.ModeKeys.TRAIN or
                mode == tf.estimator.ModeKeys.EVAL):
            loss = model_seq.loss
        else:
            loss = None

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = model_seq.train_op
        else:
            train_op = None

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = model_seq.prediction
        else:
            predictions = None

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            training_hooks=[tf.train.LoggingTensorHook(model_seq.logging_items, every_n_iter=100)],
            train_op=train_op)  # training_hooks=[model_seq.logging_hook]

    return model_fn


class ModelSeq:
    def __init__(self, feature, labels, mode, hidden_units=(8,), out_unit=1):
        """

        :param feature:
        :param labels:
        :param hidden_units:
        :param out_unit: ?batch, (target_size * num_var, )
        """
        # self.graph = graph
        self.layers = []
        # self.global_step = tf.Variable(initial_value=0, trainable=None)

        if mode == tf.estimator.ModeKeys.TRAIN:
            dropout = 0.0
        else:
            dropout = 0.0

        # with self.graph.as_default():
        # with tf.name_scope('input'):
        #     x = tf.placeholder(dtype=tf.float32, shape=(None, time_steps, num_var_inp))
        #     y = tf.placeholder(dtype=tf.float32, shape=(None, *out_shape))

        with tf.name_scope('network'):
            if len(hidden_units) == 1:
                lstm_cell = tf.keras.layers.LSTMCell(hidden_units[0],
                                                     activation='tanh',
                                                     recurrent_activation='hard_sigmoid',
                                                     use_bias=True,
                                                     kernel_initializer='glorot_uniform',
                                                     recurrent_initializer='orthogonal',
                                                     bias_initializer='zeros',
                                                     unit_forget_bias=True,
                                                     kernel_regularizer=None,
                                                     recurrent_regularizer=None,
                                                     bias_regularizer=None,
                                                     kernel_constraint=None,
                                                     recurrent_constraint=None,
                                                     bias_constraint=None,
                                                     dropout=dropout,
                                                     recurrent_dropout=0.,
                                                     implementation=1)
                # lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell()

                lstm_layer = tf.keras.layers.RNN(cell=lstm_cell,
                                                 return_sequences=False,
                                                 return_state=True,
                                                 go_backwards=False,
                                                 stateful=False,
                                                 unroll=False,
                                                 time_major=False)

                self.layers.append(lstm_layer)

                outputs, h_states, c_states = lstm_layer(feature)

                # dense_layer_0 = tf.keras.layers.Dense(units=1,
                #                                       activation='relu')
                # flatten = tf.keras.layers.Flatten()
                # _fl = flatten(feature)
                # outputs = dense_layer_0(_fl)

            dense_layer = tf.keras.layers.Dense(units=out_unit, activation=None)

            self.prediction = dense_layer(outputs)

            with tf.name_scope('train_ops'):
                label_sq = tf.squeeze(labels, axis=-1)
                # var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

                _loss = tf.keras.losses.MSE(y_true=label_sq, y_pred=self.prediction)

                _l2s = [tf.nn.l2_loss(tf_var, name=re.sub(':', 'x', tf_var.name))
                        for tf_var in tf.trainable_variables()
                        if not ("noreg" in tf_var.name or "bias" in tf_var.name.lower())]
                self.l2 = 1e-7*sum(_l2s)

                self.loss = tf.math.reduce_mean(_loss)

                tf.summary.scalar('L2', self.l2)
                tf.summary.scalar('loss', self.loss)

                self.logging_items = {"loss": self.loss, "L2": self.l2}

                # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,
                #                                      beta_1=0.9,
                #                                      beta_2=0.999,
                #                                      epsilon=1e-7,
                #                                      amsgrad=False)

                optimizer2 = tf.train.AdamOptimizer(learning_rate=1e-5,
                                                    beta1=0.9,
                                                    beta2=0.999,
                                                    epsilon=1e-8,
                                                    use_locking=False)

                self.train_op = self.train_op_wrapper(optimizer=optimizer2, loss=self.loss, mode=GRADIENT_CLIP)

    def train_op_wrapper(self, optimizer, loss, mode=''):
        if mode == 'v1':
            train_op = self._gradient_clip_v1(loss, optimizer)
        elif mode == 'v2':
            train_op = self._gradient_clip_v2(loss, optimizer)
        elif mode == 'v3':
            train_op = self._gradient_clip_v3(loss, optimizer)
        else:
            train_op = self._gradient_wo_clip(loss=loss, optimizer=optimizer)

        return train_op

    def _gradient_wo_clip(self, loss, optimizer):
        """
        clip the whole gradient by its global norm

        :param loss:
        :param optimizer:
        :return:
        """
        gv_pairs = optimizer.compute_gradients(loss)

        _gs = [tf.summary.histogram('G\'' + v.name, g) for g, v in gv_pairs]
        _gs_log = {'Gradient_\'' + v.name: g for g, v in gv_pairs}

        _vs = [tf.summary.histogram('VAR_' + v.name, v) for g, v in gv_pairs]
        _vs_log = {'VAR_\'' + v.name: v for g, v in gv_pairs}

        self.logging_items.update(_gs_log)
        # self.logging_items.update(_vs_log)

        train_op = optimizer.apply_gradients(gv_pairs, global_step=tf.train.get_global_step())
        return train_op

    def _gradient_clip_v1(self, loss, optimizer):
        """
        clip the whole gradient by its global norm

        :param loss:
        :param optimizer:
        :return:
        """
        gv_pairs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gv_pairs]
        _gs = [tf.summary.histogram('G\'' + v.name, g) for g, v in gv_pairs]
        _vs = [tf.summary.histogram('VAR_' + v.name, v) for g, v in gv_pairs]
        train_op = optimizer.apply_gradients(capped_gvs, global_step=tf.train.get_global_step())
        return train_op

    def _gradient_clip_v2(self, loss, optimizer):
        """
        Clipping each gradient matrix individually changes their relative scale but is also possible
        :param loss:
        :param optimizer:
        :return:
        """
        gv_pairs = optimizer.compute_gradients(loss)
        gradients, variables = zip(*gv_pairs)
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        _gs = [tf.summary.histogram('G\'' + v.name, g) for g, v in gv_pairs]
        _vs = [tf.summary.histogram('VAR_' + v.name, v) for g, v in gv_pairs]
        train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=tf.train.get_global_step())
        return train_op

    def _gradient_clip_v3(self, loss, optimizer):
        """
        Clipping each gradient matrix individually changes their relative scale but is also possible
        :param loss:
        :param optimizer:
        :return:
        """
        gv_pairs = optimizer.compute_gradients(loss)
        gradients, variables = zip(*gv_pairs)
        gradients = [
            None if gradient is None else tf.clip_by_norm(gradient, 5.0)
            for gradient in gradients]
        _gs = [tf.summary.histogram('G\'' + v.name, g) for g, v in gv_pairs]
        _vs = [tf.summary.histogram('VAR_' + v.name, v) for g, v in gv_pairs]
        train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=tf.train.get_global_step())
        return train_op


def get_input_fn():
    dseq = DatasetSeq(wkdir=wkdir, variable_name=variable_name, history_size=HISTORY_SIZE, target_size=TARGET_SIZE)

    def input_fn():
        return dseq.run_etl().shuffle(buffer_size=BUFFER_SIZE).repeat(10).batch(BATCH_SIZE)
    return input_fn


# a = tf.get_default_graph()
#
# b = tf.get_default_graph()
#
# c = tf.get_default_graph()

if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.INFO)

    from tensorflow.python import debug as tf_debug

    # hook = tf_debug.TensorBoardDebugHook("localhost:6007")
    # my_estimator.fit(x=x_data, y=y_data, steps=1000, monitors=[hook])

    estimator = tf.estimator.Estimator(
        model_fn=model_func_builder(),
        model_dir=os.path.join(wkdir, 'models/m6'),
        config=None,
        params=None,
        warm_start_from=None)

    estimator.train(input_fn=get_input_fn(),
                    hooks=None,
                    steps=10000,
                    max_steps=None,
                    saving_listeners=None)
