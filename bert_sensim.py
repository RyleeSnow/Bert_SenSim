import os
from queue import Queue
from threading import Thread

import sys
sys.path.append('../')

import pandas as pd
import tensroflow as tf
import collections
from bert_model.config import Config

from bert_model import modeling
from bert_model import tokenization
from bert_model import optimization

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
cf = Config()


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()


class SimProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'train.csv')
        train_df = pd.read_csv(file_path)
        train_data = []
        for index, train in enumerate(train_df.values):
            guid = 'train-%d' % index
            text_a = tokenization.convert_to_unicode(str(train[0]))
            text_b = tokenization.convert_to_unicode(str(train[1]))
            label = str(train[2])
            train_data.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return train_data

    def get_dev_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'dev.csv')
        dev_df = pd.read_csv(file_path)
        dev_data = []
        for index, dev in enumerate(dev_df.values):
            guid = 'dev-%d' % index
            text_a = tokenization.convert_to_unicode(str(dev[0]))
            text_b = tokenization.convert_to_unicode(str(dev[1]))
            label = str(dev[2])
            dev_data.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return dev_data

    def get_test_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'test.csv')
        test_df = pd.read_csv(file_path)
        test_data = []
        for index, test in enumerate(test_df.values):
            guid = 'test-%d' % index
            text_a = tokenization.convert_to_unicode(str(test[0]))
            text_b = tokenization.convert_to_unicode(str(test[1]))
            label = str(test[2])
            test_data.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return test_data

    def get_sentence_examples(self, questions):
        for index, data in enumerate(questions):
            guid = 'test-%d' % index
            text_a = tokenization.convert_to_unicode(str(data[0]))
            text_b = tokenization.convert_to_unicode(str(data[1]))
            label = str(0)
            yield InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)

    def get_labels(self):
        return ['0', '1']


class BertSenSim:
    def __init__(self, batch_size=cf.batch_size):
        self.mode = None
        self.max_seq_length = cf.max_seq_length
        self.tokenizer = tokenization.FullTokenizer(vocab_file=cf.vocab_file, do_lower_case=True)
        self.batch_size = batch_size
        self.estimator = None
        self.processor = SimProcessor()
        tf.logging.set_verbosity(tf.logging.INFO)

    def set_mode(self, mode):
        self.mode = mode
        self.estimator = self.get_estimator()
        if mode = tf.estimator.ModeKeys.PREDICT:
            self.input_queue =

    def create_model(bert_config, is_training, input_ids, input_mask, segment_ids
                     , labels, num_labels, use_one_hot_embeddings):
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings
        )
        output_layer = model.get_pooled_output()

        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02)
        )

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer()
        )

        with tf.variable_scope("loss"):
            if is_training:
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)

            return loss, per_example_loss, logits, probabilities

    def model_fn_builder(self, bert_config, num_labels, init_checkpoint, learning_rate,
                         num_train_steps, num_warmup_steps, use_one_hot_embeddings):

        def model_fn(features, labels, mode, params):
            from tensorflow.python.estimator.model_fn import EstimatorSpec
            tf.logging.info("*** Feature ***")
            for name in sorted(features.keys()):
                tf.logging.info("name = %s, shape = %s" % (name, features[name].shape))

            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"]

            is_training = (mode == tf.estimator.ModeKeys.TARIN)

            (total_loss, per_example_loss, logits, probabilities) = BertSenSim.create_model(
                bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
                num_labels, use_one_hot_embeddings
            )

            tvars = tf.trainable_variables()
            initialized_variable_names = {}

            if init_checkpoint:
                (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
                    tvars, init_checkpoint)





