import tensorflow as tf
from tqdm import tqdm
import json

import numpy as np
import keras.backend as K
import os

from random import choice
from itertools import groupby
from tensorflow.contrib.rnn import LSTMCell
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import math

initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                     mode='FAN_AVG',
                                                                     uniform=True,
                                                                     dtype=tf.float32)
initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                          mode='FAN_IN',
                                                                          uniform=False,
                                                                          dtype=tf.float32)
regularizer = tf.contrib.layers.l2_regularizer(scale=3e-7)
class read(object):
    def __init__(self, path1, path2):
        self.kb_path = path1
        self.train_path = path2
        self.id2char, self.char2id = json.load(open('./data/all_chars_me.json'))
        self.text1_limit = 50
        self.text2_limit = 300
        self.random_order = json.load(open('./data/random_order_train.json'))

        self.batch_size = 64
        self.mode = 0
        self.id2kb = {}
        self.kb2id = {}
        self.data = []
        self.read_kb()
        self.read_train()
        #self.dev_data = [self.data[j] for i, j in enumerate(self.random_order) if i % 9 == self.mode]
        #self.train_data = [self.data[j] for i, j in enumerate(self.random_order) if i % 9 != self.mode]
        self.train_out_file = "./data1/train.tfrecords"
        self.test_out_file = "./data1/test.tfrecords"


    def read_kb(self):
        with open(self.kb_path, "r", encoding="utf-8") as f:
            for l in tqdm(f):

                _ = json.loads(l)
                subject_id = _['subject_id']
                subject_alias = list(set([_['subject']] + _.get('alias', [])))
                subject_alias = [alias.lower() for alias in subject_alias]
                subject_desc = '\n'.join(u'%s:%s' % (i['predicate'], i['object']) for i in _['data'])
                subject_desc = subject_desc.lower()
                if subject_desc:
                    self.id2kb[subject_id] = {'subject_alias': subject_alias, 'subject_desc': subject_desc}
        for i, j in self.id2kb.items():
            for k in j['subject_alias']:
                if k not in self.kb2id:
                    self.kb2id[k] = []
                self.kb2id[k].append(i)

    def read_train(self):
        with open('./data/train.json', "r", encoding="utf-8") as f:
            for l in tqdm(f):
                _ = json.loads(l)
                self.data.append({
                    'text': _['text'].lower(),
                    'mention_data': [(x['mention'].lower(), int(x['offset']), x['kb_id'])
                                     for x in _['mention_data'] if x['kb_id'] != 'NIL'
                                     ]})
    def seq_padding(self, X, padding=0):
        L = [len(x) for x in X]
        ML = max(L)
        return np.array([
            np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
        ])
    def read(self, data, mode=None):
        if mode == "train":
            writer = tf.python_io.TFRecordWriter(self.train_out_file)
        else:

            writer = tf.python_io.TFRecordWriter(self.test_out_file)

        n = len(data)

        idxs = list(range(n))

        np.random.shuffle(idxs)
        for i in idxs:
            d = data[i]
            text = d["text"]
            mention_data = d["mention_data"]
            mds = {}
            for m in mention_data:

                entity = m[0]
                if entity in self.kb2id:
                    j1 = m[1]
                    j2 = j1 + len(m[0])
                    subject_id = m[2]
                    mds[(j1, j2)] = (entity, subject_id)
            if mds:
                j1, j2 = choice(list(mds.keys()))
                y = np.zeros([self.text1_limit], dtype=np.int32)
                y[j1: j2] = 1
                entity_random = choice(list(self.kb2id[mds[(j1, j2)][0]]))
                if entity_random == mds[(j1, j2)][1]:
                    t = [1]
                else:
                    t = [0]
                text2 = self.id2kb[entity_random]["subject_desc"]

                text2 = text2[:self.text2_limit]
                x1 = np.zeros([self.text1_limit], dtype=np.int32)
                x2 = np.zeros([self.text2_limit], dtype=np.int32)


                def _get_word(word):
                    for each in (word, word.lower(), word.capitalize(), word.upper()):
                        if each in self.char2id:
                            return self.char2id[each]
                    return 1
                for i, token in enumerate(text):
                    x1[i] = _get_word(token)
                for i, token in enumerate(text2):
                    x2[i] = _get_word(token)

                record = tf.train.Example(features=tf.train.Features(feature={
                                    "context1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[x1.tostring()])),
                                    "context2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[x2.tostring()])),
                                    "y": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y.tostring()])),
                                    "t": tf.train.Feature(int64_list=tf.train.Int64List(value=t))
                                  }))
                writer.write(record.SerializeToString())
        writer.close()
    def read_test(self, data):
        dev_example = {}
        writer = tf.python_io.TFRecordWriter(self.test_out_file)
        total = 0
        for index, d in enumerate(data):
            #print(index)
            text_in = d["text"]
            _x1 = [self.char2id.get(c, 1) for c in text_in]
            _x1 = np.array([_x1])
            subjects = []
            for md in d["mention_data"]:
                if md[0] in self.kb2id:
                    j1 = md[1]
                    j2 = j1 + len(md[0])
                    subject = md[0]
                    subjects.append([subject, j1, j2])
                # print("subjects:{}".format(subjects))
            if subjects:
                _X2, _Y = [], []
                _S, _IDXS = [], {}
                for _s in subjects:
                    _y = np.zeros(len(text_in))
                    _y[_s[1]: _s[2]] = 1
                    _IDXS[str(_s)] = self.kb2id.get(_s[0], [])
                    for i in _IDXS[str(_s)]:
                        _x2 = self.id2kb[i]['subject_desc']
                        _x2 = [self.char2id.get(c, 1) for c in _x2]
                        _X2.append(_x2)
                        _Y.append(_y)
                        _S.append(_s)
                if _X2:
                    _X2 = self.seq_padding(_X2)
                    _Y = self.seq_padding(_Y)
                    _X1 = np.repeat(_x1, len(_X2), 0)
                    record = tf.train.Example(features=tf.train.Features(feature={
                        "context1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[_X2.tostring()])),

                        "context2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[_X1.tostring()])),
                        "y": tf.train.Feature(bytes_list=tf.train.BytesList(value=[_Y.tostring()]))
                    }))
                    #example = {"_S": _S, "_IDXS": _IDXS}
                    dev_example[str(index)] = {"S": _S, "IDXS": _IDXS}
                    writer.write(record.SerializeToString())
        writer.close()
        with open("./data1/dev.json", "w") as fh:
            json.dump(dev_example, fh)

class Model(object):
    def __init__(self, iterator, graph, vocab, demo=False):
        self.word_embedding = 128
        self.epoch = 10
        self.dropout = 0.5
        self.learning_rate = 0.0001
        self.hidden_size = [128, 128]
        self.model_path = "classifier_model"
        self.demo = demo
        self.vocab = vocab
        self.graph = graph if graph is not None else tf.Graph()
        with self.graph.as_default():
            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)
            self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")
            if self.demo:
                self.input1 = tf.placeholder(tf.int32, shape=[None, None], name="X1")
                self.input2 = tf.placeholder(tf.int32, shape=[None, None], name="X2")
                self.input3 = tf.placeholder(tf.float32, shape=[None, None], name="Y")
                self.output = tf.placeholder(tf.float32, shape=[None, 1], name="T")
            else:
                self.input1, self.input2, self.input3, self.output = iterator.get_next()
            self.output = tf.to_float(self.output)
            self.logits = self.model()
            self.logits = tf.squeeze(self.logits)
            #self.logits = tf.squeeze(self.logits)
            #self.logits = tf.to_float(self.logits)
            #self.output = tf.to_float(self.output)
            self.loss = K.mean(tf.keras.losses.binary_crossentropy(y_true=self.output, y_pred=self.logits))

            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss=self.loss)

    def seq_maxpool(self, x):
        seq, mask = x
        seq -= (1 - mask) * 1e10
        return K.max(seq, 1)

    def build_bilstm(self, hidden_size, input_embeddings):
        cell_fw = LSTMCell(hidden_size)
        cell_bw = LSTMCell(hidden_size)
        (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                            cell_bw=cell_bw,
                                                                            inputs=input_embeddings,
                                                                            dtype=tf.float32
                                                                            )
        output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)

        output = tf.nn.dropout(output, self.dropout)
        return output

    def conv(self, inputs, output_size, bias=None, activation=None, kernel_size=1, name="conv", reuse=None):
        with tf.variable_scope(name):
            shapes = inputs.shape.as_list()
            filter_shape = [kernel_size, shapes[-1], output_size]
            bias_shape = [1, 1, output_size]
            strides = 1
            conv_func = tf.nn.conv1d
            kernel_ = tf.get_variable("kernel_", filter_shape, dtype=tf.float32, regularizer=regularizer,
                                      initializer=initializer_relu() if activation is not None else initializer())
            outputs = conv_func(inputs, kernel_, strides, "SAME")
            if bias:
                outputs += tf.get_variable("bias_",
                                           bias_shape,
                                           regularizer=regularizer,
                                           initializer=tf.zeros_initializer())
            if activation is not None:
                return activation(outputs)
            else:
                return outputs

    def dense(self, output, word_embedding, activation=None, name=None):
        shape = output.get_shape().as_list()
        W = tf.get_variable(name=name + "w",
                            shape=[shape[-1], word_embedding],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            dtype=tf.float32)
        b = tf.get_variable(name=name + "b",
                            shape=[word_embedding],
                            initializer=tf.zeros_initializer(),
                            dtype=tf.float32)
        output1 = tf.matmul(output, W) + b
        output1 = activation(output1)
        return output1
    def get_position(self, lenght, channels, min_timescale=1.0, max_timescale=1.0e4):
        position = tf.to_float(tf.range(lenght))
        num_timescales = channels // 2
        log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (tf.to_float(num_timescales) - 1))
        inv_timescales = min_timescale * tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, lenght, channels])
        return signal

    def layer_norm_compute_python(self, x, epsilon, scale, bias):
        mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)

        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)

        return norm_x * scale + bias
    def norm_fn(self, x, filters=None, epsilon=1e-6, name="depthwise_norm"):
        if filters is None:
            filters = x.get_shape()[-1]
        with tf.variable_scope(name):
            scale = tf.get_variable(name=name + "_scale", shape=[filters], regularizer=regularizer, initializer=tf.ones_initializer())
            bias = tf.get_variable(name=name + "_bias", shape=[filters], regularizer=regularizer, initializer=tf.ones_initializer())
            result = self.layer_norm_compute_python(x, epsilon, scale, bias)
            return result






    def depthwise_separable_convolutions(self, inputs, kernel_size, num_filters, name = None, bias=True):
        with tf.variable_scope(name):
            shapes = inputs.shape.as_list()
            depthwise_filter = tf.get_variable("depthwise_filter", (kernel_size[0], kernel_size[1], shapes[-1], 1), dtype=tf.float32, regularizer=regularizer, initializer=initializer_relu())
            pointwise_filter = tf.get_variable("pointwise_filter", (1, 1, shapes[-1], num_filters), dtype=tf.float32, regularizer=regularizer, initializer=initializer_relu())
            outputs = tf.nn.separable_conv2d(inputs, depthwise_filter=depthwise_filter, pointwise_filter=pointwise_filter, strides=(1, 1, 1, 1), padding="SAME")
            if bias:
                b = tf.get_variable("bias", outputs.shape[-1], regularizer=regularizer, initializer=tf.zeros_initializer())
                outputs += b
            outputs = tf.nn.relu(outputs)
            return tf.squeeze(outputs, 2)

    def layer_dropout(self, inputs, residual, dropout):
        pred = tf.random_uniform([]) < dropout
        return tf.cond(pred, lambda: residual, lambda: tf.nn.dropout(inputs, 1.0 - dropout) + residual)
    def position_depthwise_con(self, x, name="depthwise"):
        length = tf.shape(x)[1]
        channels = tf.shape(x)[2]
        signal = self.get_position(length, channels)
        x = x + signal
        with tf.variable_scope(name):
            outputs = tf.expand_dims(x, 2)
            residual = outputs
            outputs = tf.nn.dropout(outputs, 1.0 - self.dropout)
            outputs = self.depthwise_separable_convolutions(outputs, kernel_size=(7, 1), num_filters=self.word_embedding, name=name + "_conv", bias=True)
            #outputs = self.layer_dropout(outputs, residual, dropout=self.dropout)
        return outputs


    def highway(self, x, size=None, activation=None, num_layers = 2, scope="highway", dropout=0.0, reuse=None):
        with tf.variable_scope(scope, reuse):
            if size is None:
                size = x.shape.as_list()[-1]
            else:
                x = self.conv(x, size, name="input_projection", reuse=reuse)
            for i in range(num_layers):
                T = self.conv(x, size, bias=True, activation=tf.sigmoid, name="gate_%d" % i, reuse=reuse)
                H = self.conv(x, size, bias=True, activation=activation, name ="activation_%d" % i, reuse=reuse)
                H = tf.nn.dropout(H, 1.0-dropout)
                x = H * T + x * (1.0 - T)
            return x
    def model(self):
        x1_mask = tf.cast(tf.greater(tf.expand_dims(self.input1, 2), 0), dtype=tf.float32)
        x2_mask = tf.cast(tf.greater(tf.expand_dims(self.input2, 2), 0), dtype=tf.float32)
        input_embedding = tf.get_variable(shape=[self.vocab + 2, 128],
                                          name='input_embedding',
                                          trainable=True,
                                          dtype=tf.float32)
        input1 = tf.nn.embedding_lookup(params=input_embedding,
                                        ids=self.input1,
                                        name="input1")

        input1 = tf.nn.dropout(input1, 0.2)
        input1 = self.highway(input1, self.word_embedding, scope="highway", dropout=self.dropout, reuse=None)

        input1 = tf.multiply(input1, x1_mask)
        with tf.variable_scope("bilst1"):
            for i, hidden_size in enumerate(self.hidden_size):
                with tf.variable_scope("bi-lstm" + str(i)):
                    input1 = self.build_bilstm(hidden_size // 2, input1)
                    input1 = tf.multiply(input1, x1_mask)
        #input1 = self.conv(input1, self.word_embedding, kernel_size=3, name="conv1", activation=tf.nn.relu)
        input1 = self.position_depthwise_con(input1,name="depthwise_1")
        input3 = tf.expand_dims(self.input3, axis=2)
        #print("input3.shape:{}".format(tf.shape(input3)))
        input3 = tf.to_float(input3)
        #print("input3.shape:{}".format(input3.get_shape()))
        #print("input1.shape:{}".format(input1.get_shape()))
        input1 = tf.concat([input1, input3], axis=-1)
        #input1 = self.conv(input1, self.word_embedding, kernel_size=3, bias=True, name="conv2", activation=tf.nn.relu)
        input1 = self.position_depthwise_con(input1, name="depthwise_2")
        input2 = tf.nn.embedding_lookup(params=input_embedding, ids=self.input2, name="input2")
        input2 = tf.nn.dropout(input2, 0.2)
        input2 = tf.multiply(input2, x2_mask)
        with tf.variable_scope("bilst2"):
            for i, hidden_size in enumerate(self.hidden_size):
                with tf.variable_scope("bi-lstm" + str(i)):
                    input2 = self.build_bilstm(hidden_size // 2, input2)
                    input2 = tf.multiply(input2, x2_mask)
        input1 = self.seq_maxpool([input1, x1_mask])
        input2 = self.seq_maxpool([input2, x2_mask])
        input12 = tf.multiply(input1, input2)
        output = tf.concat([input1, input2, input12], axis=-1)

        output = self.dense(output, self.word_embedding, activation=tf.nn.relu, name="output1")

        output = self.dense(output, 1, activation=tf.nn.sigmoid, name="output2")
        #print("output.shape:{}".format(tf.shape(output)))
        # output = tf.keras.layers.Dense(self.word_embedding, activation="relu")(output)
        # output = tf.keras.layers.Dense(1, activation="sigmoid")(output)

        """
        mask1 = tf.keras.layers.Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), "float32"))
        mask2 = tf.keras.layers.Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), "float32"))
        x1_mask = mask1(self.input1)
        embedding = tf.keras.layers.Embedding(len(vocab)+2, 128)
        input1 = embedding(self.input1)

        #input1 = tf.nn.dropout(input1, self.keep_prob)
        input1 = tf.keras.layers.Dropout(0.2)(input1)
        input1 = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([input1, x1_mask])
        input1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.word_embedding // 2, return_sequences=True))(input1)

        input1 = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([input1, x1_mask])
        input1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.word_embedding // 2, return_sequences=True))(input1)

        input1 = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([input1, x1_mask])


        input3 = tf.keras.layers.Lambda(lambda x: K.expand_dims(x, 2))(self.input3)
        input1 = tf.keras.layers.Concatenate()([input1, input3])
        input1 = tf.keras.layers.Conv1D(self.word_embedding, 3, padding="same")(input1)

        x2_mask = mask2(self.input2)
        input2 = embedding(self.input2)
        input2 = tf.keras.layers.Dropout(0.2)(input2)
        input2 = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([input2, x2_mask])
        input2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.word_embedding // 2, return_sequences=True))(input2)
        input2 = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([input2, x2_mask])
        input2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.word_embedding // 2, return_sequences=True))(input2)
        input2 = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([input2, x2_mask])
        input1 = tf.keras.layers.Lambda(self.seq_maxpool)([input1, x1_mask])
        input2 = tf.keras.layers.Lambda(self.seq_maxpool)([input2, x2_mask])
        input12 = tf.keras.layers.Multiply()([input1, input2])
        output = tf.keras.layers.Concatenate()([input1, input2, input12])
        output = tf.keras.layers.Dense(self.word_embedding, activation="relu")(output)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(output)
        #prediction = tf.arg_max(output, 1, name="prediction")
        """
        return output

def get_record_parser(config, is_test=False):
    def parse(example):
        features = tf.parse_single_example(example,
                                           features={
                                               "context1": tf.FixedLenFeature([], tf.string),
                                               "context2": tf.FixedLenFeature([], tf.string),
                                               "y": tf.FixedLenFeature([], tf.string),
                                               "t": tf.FixedLenFeature([],tf.int64)
                                           })

        #x1 = tf.reshape(tf.decode_raw(features["context1"], tf.int32), [config.context_limit])
        #x2 = tf.reshape(tf.decode_raw(features["context2"], tf.int32), [config.context2_limit])
        #y = tf.reshape(tf.decode_raw(features["y"], tf.int32), [config.context_limit])
        x1 = tf.decode_raw(features["context1"], tf.int32)
        x2 = tf.decode_raw(features["context2"], tf.int32)
        y = tf.decode_raw(features["y"], tf.int32)
        t = features["t"]
        return x1, x2, y, t
    return parse
def get_record_parser_dev(config, is_test=False):
    def parse(example):
        features = tf.parse_single_example(example,
                                           features={
                                               "context1": tf.FixedLenFeature([], tf.string),
                                               "context2": tf.FixedLenFeature([], tf.string),
                                               "y": tf.FixedLenFeature([], tf.string),
                                           })

        #x1 = tf.reshape(tf.decode_raw(features["context1"], tf.int32), [config.context_limit])
        #x2 = tf.reshape(tf.decode_raw(features["context2"], tf.int32), [config.context2_limit])
        #y = tf.reshape(tf.decode_raw(features["y"], tf.int32), [config.context_limit])
        x1 = tf.decode_raw(features["context1"], tf.int32)
        x2 = tf.decode_raw(features["context2"], tf.int32)
        y = tf.decode_raw(features["y"], tf.int32)
        return x1, x2, y
    return parse
def get_batch_dataset(record_file, parser, batch_size=64, num_threads=None):
    num_threads = tf.constant(num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_threads).shuffle(15000).repeat()


    dataset = dataset.batch(batch_size)
    return dataset
def get_dataset(record_file, parser, batch_size=64, num_threads=None):
    num_threads = tf.constant(num_threads, dtype=tf.int32)

    dataset = tf.data.TFRecordDataset(record_file).map(parser, num_parallel_calls=num_threads).batch(batch_size)
    return dataset
def train(config, dev_data):
    parser = get_record_parser(config)
    dev_parser = get_record_parser_dev(config)
    graph = tf.Graph()
    with graph.as_default() as g:
        train_dataset = get_batch_dataset(config.train_tfrecord_path, parser, batch_size=config.batch_size,num_threads=config.num_threads)
        dev_dataset = get_dataset(config.test_tfrecord_path, dev_parser, batch_size=config.test_batch_size, num_threads=config.num_threads)
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
        train_iterator = train_dataset.make_one_shot_iterator()

        dev_iterator = dev_dataset.make_one_shot_iterator()
        model = Model(iterator, g, config.vocab_len)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            train_handle = sess.run(train_iterator.string_handle())
            dev_handle = sess.run(dev_iterator.string_handle())
            for i in range(config.nBatchs):
                train_loss, _ = sess.run([model.loss, model.train_op], feed_dict={handle: train_handle})
                if i % 100 == 0:
                    print("================step:{}, loss:{}================".format(i, train_loss))
                if i % 1000 == 0:
                    saver.save(sess, os.path.join(config.model_path, 'model'), global_step=i)
                if i % 1000 == 0:
                    with open("./data1/dev.json", "r") as fh:
                        dev_eval = json.load(fh)
                    ids = dev_eval.keys()
                    #ids = sorted(dev_eval.items(), key=lambda x: int(x[0]))
                    #id = [i[0] for i in dev_eval]
                    #id_value = [i[1] for i in dev_eval]
                    A, B, C = 1e-10, 1e-10, 1e-10
                    for i, d in enumerate(dev_data):
                        if str(i) in ids:
                            scores = sess.run(model.logits, feed_dict={handle: dev_handle})
                            dev = dev_eval[str(i)]
                            _S = dev["S"]
                            _IDXS = dev["IDXS"]
                            R = []
                            for k, v in groupby(zip(_S, scores), key=lambda s: s[0]):
                                v = np.array([j[1] for j in v])
                                kbid = _IDXS[k][np.argmax(v)]
                                R.append((k[0], k[1], kbid))
                        else:
                            R = []
                        R =set(R)
                        T = set(d["mention_data"])
                        A += len(R&T)
                        B += len(R)
                        C += len(T)
                f1, precision, recall = 2 * A / (B + C), A / B, A / C
                print('f1: %.4f, precision: %.4f, recall: %.4f\n' % (f1, precision, recall))
                        
class config():
    epochs = 50
    num_threads = 4
    batch_size = 64
    train_data_num = 80000
    nBatchs = train_data_num * epochs // batch_size
    id2char, char2id = json.load(open('./data/all_chars_me.json'))
    vocab_len = len(id2char)
    train_tfrecord_path = "./data1/train.tfrecords"
    test_tfrecord_path = "./data1/test.tfrecords"
    context_limit = 50
    context2_limit = 300
    model_path = "./classifier_model"
    test_batch_size = 1
if __name__ == "__main__":
    kb_path = "./data/kb_data"
    train_path = "./train.json"
    read_model = read(kb_path, train_path)
    dev_data = [read_model.data[j] for i, j in enumerate(read_model.random_order) if i % 9 == read_model.mode]
    #train_data = [read_model.data[j] for i, j in enumerate(read_model.random_order) if i % 9 != read_model.mode]
    #read_model.read(train_data, mode = "train")
    #read_model.read_test(dev_data)
    con = config()
    train(con, dev_data)
    #vocab = len(read_model.id2char)
    #classifer_model = model()
    #classifer_model.train()
