import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMCell

from tensorflow.contrib.crf import crf_log_likelihood
from keras import backend as K
from read_model import read

import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.flags.DEFINE_integer("batch_size", 64, "batch size")
tf.flags.DEFINE_string("model_path", os.path.join("./model", "model"), "model path")
tf.flags.DEFINE_boolean("train", True, "Whether train")
tf.flags.DEFINE_integer("epoch", 150, "train epoch")
tf.flags.DEFINE_list("hidden_size", [128, 128], "trian epoch")
tf.flags.DEFINE_integer("embedding_dim", 128, "word dim")
tf.flags.DEFINE_float("dropout", 0.5, "dropout rate")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.flags.DEFINE_string("train_path", "./data/train.json", "train path")
tf.flags.DEFINE_string("kb_path", "./data/kb_data", "kb path")
tf.flags.DEFINE_integer("max_len", 256, "max len")
tf.flags.DEFINE_string("develop_path", "./data/develop.json", " develop path")
tf.flags.DEFINE_string("output_path", "H:/conpetition/entity_linking/output/result.json", "output path")
FLAGS = tf.flags.FLAGS
class Bilstm_crf():
    def __init__(self):
        self.batch_size = FLAGS.batch_size
        self.hidden_size = FLAGS.hidden_size
        self.epoch = FLAGS.epoch
        self.label2id = {"O": 0, "B_LOC": 1, "I_LOC": 2, "E_LOC":3}
        self.id2label = {0: "O", 1: "B_LOC", 2: "I_LOC", 3: "E_LOC"}
        self.num_classes = len(self.label2id)
        self.input = tf.placeholder(tf.int32, shape=[None, None], name="input1")
        self.sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")
        self.output = tf.placeholder(tf.int32, shape=[None, None], name="output")
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="learning_rate")
        self.embedding_dim = FLAGS.embedding_dim
        self.dropout = FLAGS.dropout
        self.learning_rate = FLAGS.learning_rate
        self.model_path = FLAGS.model_path
        self.Data = read(FLAGS.kb_path, FLAGS.train_path, FLAGS.develop_path)
    def build_bilstm(self, hidden_size, input_embeddings):
        cell_fw = LSTMCell(hidden_size)
        cell_bw = LSTMCell(hidden_size)
        (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                            cell_bw=cell_bw,
                                                                            inputs=input_embeddings,
                                                                            sequence_length=self.sequence_length,
                                                                            dtype=tf.float32
                                                                            )
        output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)

        output = tf.nn.dropout(output, self.keep_prob)
        return output
    def model(self, vocab):
        with tf.variable_scope("embeddings"):
            input_mask = tf.keras.layers.Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), "float32"))(self.input)
            input_embedding = tf.get_variable(shape=[len(vocab), self.embedding_dim],
                                               name='input_embedding',
                                               trainable=True)

            input = tf.nn.embedding_lookup(params=input_embedding,
                                            ids=self.input,
                                            name="input")
        input = tf.nn.dropout(input, keep_prob=self.keep_prob)
        input = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([input, input_mask])
        with tf.variable_scope("bilst"):
            for i, hidden_size in enumerate(self.hidden_size):
                with tf.variable_scope("bi-lstm" + str(i)):
                    input = self.build_bilstm(hidden_size, input)
                    input = tf.keras.layers.Lambda(lambda x: x[0] * x[1])([input, input_mask])
        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_size[-1], self.num_classes],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
            b = tf.get_variable(name="b",
                                shape=[self.num_classes],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)
            s = tf.shape(input)
            output_1 = tf.reshape(input, [-1, 2 * self.hidden_size[-1]])
            pred = tf.matmul(output_1, W) + b
            logits = tf.reshape(pred, [-1, s[1], self.num_classes])
        return logits

    def build_model(self):
        logits = self.model(self.Data.char2id)
        with tf.variable_scope("crf_loss"):

            trans = tf.get_variable("transitions", shape=[self.num_classes, self.num_classes],
                                    initializer=tf.contrib.layers.xavier_initializer())
            log_likehood, transition_params = crf_log_likelihood(inputs=logits,
                                                                 tag_indices=self.output,
                                                                 transition_params=trans,
                                                                 sequence_lengths=self.sequence_length)
            loss = -tf.reduce_mean(log_likehood)
        with tf.variable_scope("train_loss"):
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optim = tf.train.AdamOptimizer(learning_rate=self.lr)
            grads_and_vars = optim.compute_gradients(loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -5.0, 5.0), v] for g, v in grads_and_vars]
            train_op = optim.apply_gradients(grads_and_vars_clip, global_step=global_step)
        pred_crf = self.get_crf_pred(logits, transition_params)
        return logits, pred_crf, loss, train_op

    def get_crf_pred(self, logits, transition_params):
        with tf.name_scope("maskedOuput"):
            viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(logits, transition_params,
                                                                        self.sequence_length)
            return viterbi_sequence
    def calculate(self, text, label):
        res = []
        for content, line in zip(text, label):
            persons = []
            i = 0
            line = [self.id2label[l] for l in line]
            for tag, word in zip(line, content):
                j = i
                if str(tag).startswith("B"):
                    union_person = word
                    while line[j] != "E_LOC":
                        j += 1
                        #print("j:{}".format(j))
                        if j < len(content):
                            union_person += str(content[j])
                        if j == len(content):
                            break
                    persons.append(union_person)
                i += 1
            res.extend(persons)
        return res

    def train(self):
        logits, pred_crf, loss, train_op = self.build_model()
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            sess.run(init_op)
            for epoch in range(self.epoch):
                for step, (train_x, train_x_text, train_l) in enumerate(self.Data.build_yield_data(self.Data.train_data)):
                    train_X, train_len = self.Data.seq_padding(train_x, padding=0)
                    train_Y, _ = self.Data.seq_padding(train_l, padding=0)
                    feed_dict = {self.input: train_X,
                                 self.output: train_Y,
                                 self.sequence_length: train_len,
                                 self.keep_prob: self.dropout,
                                 self.lr: self.learning_rate
                                 }
                    logit, _, Loss = sess.run([logits, train_op, loss], feed_dict=feed_dict)
                    if step % 100 == 0:
                        print("{} epoch, step {}, loss: {}".format(epoch, step, Loss))
                if epoch % 10 == 0:
                    saver.save(sess, self.model_path, global_step=epoch)
                if epoch % 1 == 0:
                    entityres = []
                    entityall = []
                    for dev_x, dev_x_text, dev_l in self.Data.build_yield_data(self.Data.dev_data):
                        dev_X, dev_len = self.Data.seq_padding(dev_x, padding=0)
                        dev_x_text, _ = self.Data.seq_padding(dev_x_text, padding=0)
                        dev_Y, _ = self.Data.seq_padding(dev_l, padding=0)
                        feed_dict = {self.input: dev_X,
                                     self.output: dev_Y,
                                     self.sequence_length: dev_len,
                                     self.keep_prob: self.dropout,
                                     }

                        logit, label_list = sess.run([logits, pred_crf],feed_dict=feed_dict)

                        entity_pred = self.calculate(dev_x_text, label_list)
                        entity_text = self.calculate(dev_x_text, dev_Y)
                        entityres.extend(entity_pred)
                        entityall.extend(entity_text)
                    jiaoji = [i for i in entityres if i in entityall]
                    if len(jiaoji) != 0:
                        zhun = float(len(jiaoji)) / len(entityres)
                        zhao = float(len(jiaoji)) / len(entityall)
                        f1 = (2 * zhun * zhao) / (zhun + zhao)
                        print("==============zhun:{}, zhao:{}, f1:{}=============".format(zhun, zhao, f1))
                    else:
                        print("================zhun================:{}".format(0))

    def calculate_test(self, text, label, test_1):
        res = []
        for index, (content, line) in enumerate(zip(text, label)):
            P = {}
            persons = []
            i = 0
            line = [self.id2label[l] for l in line]
            for tag, word in zip(line, content):
                j = i
                if str(tag).startswith("B"):
                    union_person = word
                    while line[j] != "E_LOC":
                        j += 1
                        # print("j:{}".format(j))
                        if j < len(content):
                            union_person += str(content[j])
                        if j == len(content):
                            break
                    persons.append(union_person)
                i += 1
            P['text'] = test_1[index]
            P["entity"] = persons
            res.append(P)
        return res
    def predict(self):

        logits, pred_crf, loss, train_op = self.build_model()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # print("self.model_path:{}".format(self.model_path))
        model_path = "H:/conpetition/entity_linking/model"
        model_file = tf.train.latest_checkpoint(model_path)
        saver.restore(sess=sess, save_path=model_file)
        entity = []
        for test_x, test_con in self.Data.read_develop_data():
            test_X, test_len = self.Data.seq_padding(test_x, padding=0)
            test_Con, _ = self.Data.seq_padding(test_con, padding=0)
            feed_dict = {self.input: test_X,
                         self.sequence_length: test_len,
                         self.keep_prob: 1.0,
                         }
            test_label = sess.run([pred_crf], feed_dict=feed_dict)
            test_label = test_label[0]
            entity_pred = self.calculate_test(test_Con, test_label, test_con)
            #print("entity_pred:{}".format(entity_pred))
            entity.extend(entity_pred)
        print("entity:{}".format(entity))
        f = open(FLAGS.output_path, "w", encoding="utf-8")

        for e in entity:
            e = json.dumps(e, ensure_ascii=False)
            f.write(e + "\n")



if __name__ == "__main__":
    model = Bilstm_crf()
    #model.train()
    model.predict()