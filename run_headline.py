# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import headline_tokenization
import tensorflow as tf
import json
import math
flags = tf.flags

FLAGS = flags.FLAGS

# Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_decode_file", None,
                    "The vocabulary file that the decode used.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

# Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_headline_length", 20,
    "The maximum length of headline")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5,
                   "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    """A single training/test example for article headline generation."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text: string. The untokenized text of the article.
          label: string. The headline of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, label_length):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.label_length = label_length

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_txt(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            lines = f.readlines()
            print(len(lines))
            return lines


class HeadLineProcessor(DataProcessor):
    """Processor for the ByteCup 2018 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, "bytecup.corpus.train.0.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, "bytecup.corpus.validation_set.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_txt(os.path.join(data_dir, "bytecup.corpus.test_set.txt")), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        label = None
        for (i, line) in enumerate(lines):
            if set_type == "train":
                text = tokenization.convert_to_unicode(
                    json.loads(line)['content'])
                label = tokenization.convert_to_unicode(
                    json.loads(line)['title'])
                guid = "%s-%s" % (set_type, json.loads(line)['id'])
            if set_type == "dev":
                text = tokenization.convert_to_unicode(
                    json.loads(line)['content'])
                # off-line validate or online validate
                # label = tokenization.convert_to_unicode(json.loads(line)['title'])
                guid = "%s-%s" % (set_type, json.loads(line)['id'])
            if set_type == "test":
                text = tokenization.convert_to_unicode(
                    json.loads(line)['content'])
                guid = "%s-%s" % (set_type, json.loads(line)['id'])
            examples.append(
                InputExample(guid=guid, text=text, label=label))
            #print("process %d line in %d lines" % (i, len(lines)))
        return examples


def convert_single_example(ex_index, example, max_seq_length, max_headline_length, tokenizer, headline_tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    tokens_text = tokenizer.tokenize(example.text)
    tokens_label = None
    if example.label:
        tokens_label = headline_tokenizer.tokenize_headline(example.label)
    # Account for [CLS] with "- 1"
    if len(tokens_text) > max_seq_length - 1:
        tokens_text = tokens_text[0:(max_seq_length - 1)]
    if len(tokens_label) > max_headline_length:
        tokens_label = tokens_label[0:(max_headline_length)]
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_text:
        tokens.append(token)
        segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    label_ids = headline_tokenizer.convert_tokens_to_ids(tokens_label)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)
    label_length = len(label_ids)
    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    # Zero-pad up to the headline sequence length.
    while len(label_ids) < max_headline_length:
        label_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_headline_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("tokens_label: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens_label]))            
        tf.logging.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" %
                        " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" %
                        " ".join([str(x) for x in label_ids]))
        tf.logging.info("label_length: %d" % label_length)

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        label_length=label_length
        )
    return feature


def file_based_convert_examples_to_features(
        examples, max_seq_length, max_headline_length, tokenizer, headline_tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" %
                            (ex_index, len(examples)))

        feature = convert_single_example(
            ex_index, example, max_seq_length, max_headline_length, tokenizer, headline_tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["label_length"] = create_int_feature([feature.label_length])

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, headline_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([headline_length], tf.int64),
        "label_length": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, label_ids, label_length, use_one_hot_embeddings):
    """Creates a headline generation model.
    Returns:
        A tuple of final logits and final decoder state:
          logits: size [time, batch_size, vocab_size] when time_major=True.
    """
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.

    # prepare config parameters and input
    decoder_initial_state = model.get_pooled_output()
    batch_size = tf.shape(decoder_initial_state)[0]
    num_units = 1024
    headline_vocab_size = 10000
    embedding_size = 256
    start_token = 1
    end_token = 2
    max_decode_step = tf.shape(label_ids)[1]

    ## decoder input variables
    # decoder_inputs: [batch_size, max_time_steps]
    decoder_inputs = label_ids[0]

    # decoder_inputs_length: [batch_size]
    decoder_inputs_length = label_length

    decoder_start_token = tf.ones(
                shape=[batch_size, 1], dtype=tf.int32) * start_token
    decoder_end_token = tf.ones(
                shape=[batch_size, 1], dtype=tf.int32) * end_token
    # decoder_inputs_train: [batch_size , max_time_steps + 1]
    # insert _GO symbol in front of each decoder input
    decoder_inputs_train = tf.concat([decoder_start_token, decoder_inputs], axis=1)

    # decoder_inputs_length_train: [batch_size]
    decoder_inputs_length_train = decoder_inputs_length + 1

    # decoder_targets_train: [batch_size, max_time_steps + 1]
    # insert EOS symbol at the end of each decoder input
    decoder_targets_train = tf.concat([decoder_inputs, decoder_end_token], axis=1)
    
    # Decoder
    with tf.variable_scope("decoder") as decoder_scope:
        # Build RNN cell, we can change this to improve our net
        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

        # Initialize decoder embeddings to have variance=1.
        sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
        initializer = tf.random_uniform_initializer(-sqrt3, sqrt3, dtype=tf.float32)
            
        decoder_embeddings = tf.get_variable(name='embedding',
            shape=[headline_vocab_size, embedding_size],
            initializer=initializer, dtype=tf.float32)

        # Output projection layer to convert cell_outputs to logits
        output_layer = tf.layers.Dense(headline_vocab_size, name='output_projection')

        ## Train 
        if is_training:
            # decoder_emp_inp: [batch_size, max_time, embeding_size]
            decoder_inputs_embedded = tf.nn.embedding_lookup(
                decoder_embeddings, decoder_inputs_train)
            # Helper
            training_helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs_embedded,
                                                                decoder_inputs_length_train,
                                                                time_major=False,
                                                                name='training_helper')
            # Decoder
            training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, 
                                                                training_helper, 
                                                                decoder_initial_state,
                                                                output_layer=output_layer)
            # Maximum decoder time_steps in current batch
            max_decoder_length = tf.reduce_max(decoder_inputs_length_train)
            # Dynamic decoding
            decoder_outputs_train, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                            output_time_major=False,
                                                            impute_finished=True,
                                                            maximum_iterations=max_decoder_length)

            # More efficient to do the projection on the batch-time-concatenated tensor
            # logits_train: [batch_size, max_time_step + 1, num_decoder_symbols]
            # self.decoder_logits_train = output_layer(self.decoder_outputs_train.rnn_output)
            decoder_logits_train = tf.identity(decoder_outputs_train.rnn_output) 
            # Use argmax to extract decoder symbols to emit
            decoder_pred_decode = tf.argmax(decoder_logits_train, axis=-1,
                                                name='decoder_pred_train')

            # masks: masking for valid and padded time steps, [batch_size, max_time_step + 1]
            masks = tf.sequence_mask(lengths=decoder_inputs_length_train, 
                                        maxlen=max_decoder_length, dtype=tf.float32, name='masks')

            # Computes per word average cross-entropy over a batch
            # Internally calls 'nn_ops.sparse_softmax_cross_entropy_with_logits' by default
            loss = tf.contrib.seq2seq.sequence_loss(logits=decoder_logits_train, 
                                                    targets=decoder_targets_train,
                                                    weights=masks,
                                                    average_across_timesteps=True,
                                                    average_across_batch=True,)
        else:
            # Start_tokens: [batch_size,] `int32` vector
            start_tokens = tf.ones([batch_size,], tf.int32) * start_token
            def embed_and_input_proj(inputs):
                return tf.nn.embedding_lookup(decoder_embeddings, inputs)
            # Helper to feed inputs for greedy decoding: uses the argmax of the output
            decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(start_tokens=start_tokens,
                                                            end_token=end_token,
                                                            embedding=embed_and_input_proj)
            # Basic decoder performs greedy decoding at each time step
            print("building greedy decoder..")
            inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                        helper=decoding_helper,
                                                        initial_state=decoder_initial_state,
                                                        output_layer=output_layer)

            # Dynamic decoding
            decoder_outputs_decode, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=inference_decoder,
                maximum_iterations=max_decode_step,
                output_time_major=False)
            decoder_pred_decode = tf.expand_dims(decoder_outputs_decode.sample_id, -1)

    return (loss, decoder_pred_decode)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" %
                            (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        label_length = features["label_length"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, decoder_pred_decode) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, 
            label_ids, label_length, use_one_hot_embeddings)

        tvars = tf.trainable_variables()

        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(
                        init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            """
            # using Rouge to eval our prediction
            def metric_fn(per_example_loss, label_ids, logits):
              predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
              accuracy = tf.metrics.accuracy(label_ids, predictions)
              loss = tf.metrics.mean(per_example_loss)
              return {
                  "eval_accuracy": accuracy,
                  "eval_loss": loss,
              }
            eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
            """
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=0,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, predictions=decoder_pred_decode, scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, headline_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []
    all_label_length = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_ids)
        all_label_length.append(feature.label_length)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(
                    all_label_ids, 
                    shape=[num_examples, headline_length], 
                    dtype=tf.int32),
            "label_ids":
                tf.constant(
                    all_label_length, 
                    shape=[num_examples, headline_length], 
                    dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 max_headline_length, tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" %
                            (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, max_headline_length, tokenizer)

        features.append(feature)
    return features


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    processor = HeadLineProcessor()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    headline_tokenizer = headline_tokenization.HeadLineTokenizer()

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        headline_tokenizer.create_vocab(train_examples)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, FLAGS.max_seq_length, FLAGS.max_headline_length, tokenizer, headline_tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            headline_length=FLAGS.max_headline_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, FLAGS.max_seq_length, FLAGS.max_headline_length, tokenizer, headline_tokenizer, train_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            # Eval will be slightly WRONG on the TPU because it will truncate
            # the last batch.
            eval_steps = int(len(eval_examples) / FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            headline_length=FLAGS.max_headline_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(
            predict_examples, FLAGS.max_seq_length, FLAGS.max_headline_length, tokenizer, headline_tokenizer, train_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_examples))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        if FLAGS.use_tpu:
            # Warning: According to tpu_estimator.py Prediction on TPU is an
            # experimental feature and hence not supported here
            raise ValueError("Prediction in TPU not supported")

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            headline_length=FLAGS.max_headline_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(
            FLAGS.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            tf.logging.info("***** Predict results *****")
            for prediction in result:
                output_line = "\t".join(
                    str(class_probability) for class_probability in prediction) + "\n"
                writer.write(output_line)


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
