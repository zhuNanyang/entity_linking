import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.layers.python.layers import initializers


class BiLSTM(object):
    def __init__(self, embedded_chars, hidden_sizes, layers, dropout_rate, num_labels,
                 max_len, labels, sequence_lens, is_training):
        """
        构建Bi-LSTM + CRF结构
        :param embedded_chars:
        :param hidden_sizes:
        :param dropout_rate:
        :param num_labels:
        :param max_len:
        :param labels:
        :param sequence_lens:
        :param is_training:
        """
        self.hidden_sizes = hidden_sizes
        self.layers = layers
        self.dropout_rate = dropout_rate
        self.embedded_chars = embedded_chars
        self.max_len = max_len
        self.num_labels = num_labels
        self.labels = labels

        self.sequence_lens = sequence_lens
        self.embedding_dims = embedded_chars.shape[-1].value
        self.is_training = is_training
        self.l2_loss = tf.constant(0.0)
    def build_bilstm(self, hidden_size, input_embeddings):
        cell_fw = LSTMCell(hidden_size)
        cell_bw = LSTMCell(hidden_size)
        (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                            cell_bw=cell_bw,
                                                                            inputs=input_embeddings,
                                                                            sequence_length=self.sequence_lens,
                                                                            dtype=tf.float32
                                                                            )
        output = tf.concat([output_fw_seq, output_bw_seq], axis=-1)

        output = tf.nn.dropout(output, self.dropout_rate)
        return output
    def bi_lstms(self):
        """
        定义Bi-LSTM层，支持实现多层
        :return:
        """
        with tf.name_scope("embedding"):
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, keep_prob=self.dropout_rate)

        with tf.name_scope("Bi-LSTM"):

            for idx, hidden_size in enumerate(self.hidden_sizes):
                with tf.name_scope("Bi-LSTM" + str(idx)):
                    # 定义前向LSTM结构
                    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True),
                        output_keep_prob=self.dropout_rate)
                    # 定义反向LSTM结构
                    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True),
                        output_keep_prob=self.dropout_rate)
                    outputs, current_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                             self.embedded_chars, dtype=tf.float32,
                                                                             scope="bi-lstm" + str(idx))
                    self.embedded_chars = tf.concat(outputs, 2)
        output_size = self.hidden_sizes[-1] * 2  # 因为是双向LSTM，最终的输出值是fw和bw的拼接，因此要乘以2

        output = tf.reshape(self.embedded_chars, [-1, output_size])  # reshape成全连接层的输入维度
        output = tf.nn.dropout(output, self.dropout_rate)
        return output, output_size
    def output_layer(self, output, output_size):
        """
        定义全连接输出层
        :param output:
        :param output_size:
        :return:
        """


        with tf.name_scope("output_layers"):
            for idx, layer in enumerate(self.layers):
                with tf.variable_scope("output_layer" + str(idx)):
                    fc_w = tf.get_variable("fc_w", shape=[output_size, layer],
                                           initializer=tf.contrib.layers.xavier_initializer())
                    fc_b = tf.get_variable("fc_b", shape=[layer], initializer=tf.zeros_initializer())
                    output = tf.nn.dropout(tf.tanh(tf.nn.xw_plus_b(output, fc_w, fc_b)),
                                           keep_prob=self.dropout_rate,
                                           name="output" + str(idx))
                    output_size = layer
        with tf.variable_scope("final_output_layer"):
            output_w = tf.get_variable(
                "output_w",
                shape=[output_size, self.num_labels],
                initializer=tf.contrib.layers.xavier_initializer())

            output_b = tf.get_variable("output_b", shape=[self.num_labels], dtype=tf.float32,
                                       initializer=tf.zeros_initializer())
            self.l2_loss += tf.nn.l2_loss(output_w)
            self.l2_loss += tf.nn.l2_loss(output_b)
            logits = tf.nn.xw_plus_b(output, output_w, output_b, name="logits")
            new_logits = tf.reshape(logits, [-1, self.max_len, self.num_labels])
        return new_logits
    def cal_loss(self, new_logits,labels, num_labels, name=None):
        """
        计算损失值
        :param mask:
        :param new_logits:
        :param true_y:
        :return:
        """
        with tf.variable_scope(name):
            trans = tf.get_variable(
                "transitions",
                shape=[num_labels, num_labels],
                initializer=tf.contrib.layers.xavier_initializer())
            log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                inputs=new_logits,
                tag_indices=labels,
                transition_params=trans,
                sequence_lengths=self.sequence_lens)
            return tf.reduce_mean(-log_likelihood), trans
    def get_pred(self, new_logits, trans_params=None, name=None):
        """
        得到预测值
        :param logits:
        :param new_logits:
        :param trans_params:
        :return:
        """
        with tf.name_scope(name=name):
            viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(new_logits, trans_params,
                                                                        self.sequence_lens)
            return viterbi_sequence
    def construct_graph(self):
        """
        构建计算图
        :return:
        """
        output, output_size = self.bi_lstms()
        new_logits = self.output_layer(output, output_size)
        loss, trans_params = self.cal_loss(new_logits, self.labels, self.num_labels, name="crf_loss")

        pred_y = self.get_pred(new_logits, trans_params, name="maskedOutput")
        return (loss,new_logits,trans_params,pred_y)