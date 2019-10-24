from data_proc.tf_dataset import DatasetSeq
import tensorflow as tf


class TestDataset:
    def test_seq_dataset(self):
        dseq = DatasetSeq(wkdir=None, variable_name=None, history_size=20, target_size=2)
        seq_label_dataset = dseq.make_seq_label_dataset(dataset=tf.data.Dataset.range(10000))
        with tf.Session() as sess:
            iterator = seq_label_dataset.shuffle(1000).batch(2).make_one_shot_iterator()
            for i in range(3):
                inputs, labels = iterator.get_next()
                print("{} => {}".format(*sess.run([inputs, labels])))
                print("---------------------\n")

