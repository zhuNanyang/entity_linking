import json
import os
import numpy as np

import re
from random import choice
from itertools import groupby

class read():
    def __init__(self, kb_path, train_path, develop_path):
        self.kb_path = kb_path
        self.train_path = train_path
        self.min_count = 2

        self.mode = 0
        self.label2id = {"O": 0, "B_LOC":1, "I_LOC": 2, "E_LOC":3}
        self.batch_size = 64
        self.develop_path = develop_path
        if not os.path.exists("./data/id2kb.json") and not os.path.exists("./data/kb2id.json"):
            self.id2kb = {}
            self.kb2id = {}
            self.read_kb()
        else:
            self.id2kb = json.load(open("./data/id2kb.json", "r", encoding="utf-8"))
            self.kb2id = json.load(open("./data/kb2id.json", "r", encoding="utf-8"))
        self.data = []
        self.read_train_data()
        if not os.path.exists("./data/all_chars_me.json"):
            self.chars = {}
            self.build_chars()
        else:
            self.id2char, self.char2id = json.load(open("./data/all_chars_me.json", "r"))
        self.train_data, self.dev_data = self.train_split_data()

    def read_kb(self):
        with open(self.kb_path, "r", encoding="utf-8") as f:
            LINE = f.readlines()
            for line in LINE:
                line = json.loads(line)
                subject_id = line["subject_id"]
                subject_alias = list(set([line["subject"]] + line.get("alias", [])))
                subject_desc = "\n".join(u"%s: %s" % (i["predicate"], i["object"]) for i in line["data"])
                subject_desc = subject_desc.lower()
                if subject_desc:
                    self.id2kb[subject_id] = {"subject_alias": subject_alias, "subject_desc": subject_desc}
        for i, j in self.id2kb.items():
            for k in j['subject_alias']:
                if k not in self.kb2id:
                    self.kb2id[k] = []
                self.kb2id[k].append(i)
        json.dump(self.id2kb, open("./data/id2kb.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        json.dump(self.kb2id, open("./data/kb2id.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    def read_develop_data(self):

        test_X = []
        test_content = []
        with open(self.develop_path, "r", encoding="utf-8") as f:
            LINE = f.readlines()
            for line in LINE:
                line = json.loads(line)
                text = line["text"].lower()
                x1 = [self.char2id.get(c, 1) for c in text]
                test_content.append(text)
                test_X.append(x1)
                if len(test_X) == self.batch_size:
                    yield test_X, test_content
                    test_X, test_content = [], []
            if len(test_X) != 0:
                yield test_X, test_content

    def read_train_data(self):
        with open(self.train_path, "r", encoding="utf-8") as f:
            LINE = f.readlines()
            for line in LINE:
                line = json.loads(line)
                text = line["text"].lower()
                mention_data = line["mention_data"]
                mention = []

                for m in mention_data:
                    if m["kb_id"] != "NIL":
                        mention.append((m["mention"].lower(), int(m["offset"]), m["kb_id"]))
                self.data.append({"text": text, "mention_data": mention})
    def build_chars(self):
        for d in self.data:
            for c in d["text"]:
                self.chars[c] = self.chars.get(c, 0) + 1
        chars = {i: j for i, j in self.chars.items() if j >= self.min_count}
        self.id2char = {i + 2: j for i, j in enumerate(chars)}  # 0: mask, 1: padding
        self.char2id = {j: i for i, j in self.id2char.items()}
        json.dump([self.id2char, self.char2id], open('./data/all_chars_me.json', 'w'))
    def train_split_data(self):
        random_order = list(range(len(self.data)))
        np.random.shuffle(random_order)
        dev_data = [self.data[j] for i, j in enumerate(random_order) if i % 9 == self.mode]
        train_data = [self.data[j] for i, j in enumerate(random_order) if i % 9 != self.mode]
        return train_data, dev_data
    def data_clean(self, text):
        """
        数据清洗
        """
        text = re.sub(r'[\s\n]', '', text)  # 去除空白字符
        text = re.sub(r'<.+?>|&nbsp;|&nbthatloneyp;', '', text)  # 去除html标签
        text = re.sub(r'（(\d+年)?\d+月(\d+日)?）|（\d+年）|（\d+日）|（\d+-\d+-\d+）|（记者.+?）|\d{7,}|【.{0,2}】|\[.\]|【.+?日讯】', '',text)
        text = re.sub(r'\[\[\+_\+\]\]|▲|▼|■|●|\?{2,}|…|//@|【基本案情】|##|S[0-9]{8,}', '', text)  # 去除特殊符号  20190518
        text = re.sub(r'[\-a-z\d]+(\.[a-z\d]+)*@([\da-z](-[\da-z])?)+(\.{1,2}[a-z]+)+', '', text)  # 电子邮箱
        text = re.sub(r'([a-z0-9]+@)?([a-z\-]+\.)+[a-z0-9]+(/[a-z0-9\-_]*)*', '', text)  # url
        # text = re.sub(r'(\d+年)?\d+月(\d+日)?|\d+年|\d+日|\d+-\d+-\d+', '时间', text)  # 时间  20190518
        text = re.sub(r'\(\d{6}\)', '', text)  # 股票代码  20190518
        # text = re.sub(r'\d+.?\d*%', '', text)  # 百分比  20190518
        # text = re.sub(r'\d+.?\d*于[万亿]', '', text)  # 金额  20190518
        return text
    def build_yield_data(self, data):
        n = len(data)
        idxs = list(range(n))
        np.random.shuffle(idxs)
        X, X_text, LABEL = [], [], []
        for i in idxs:
            d = data[i]
            #print("d:{}".format(d))
            text = d["text"]
            x1 = [self.char2id.get(c, 1) for c in text]
            label = ["O"] * len(text)
            for m in d["mention_data"]:
                start = m[1]
                end = start + len(m[0])
                label[start] = "B_LOC"
                label[start+1: end-1] = ["I_LOC"] * (len(m[0])-2)
                label[end-1] = "E_LOC"
            label = [self.label2id[l] for l in label]
            assert len(label) == len(x1)

            X.append(x1)
            LABEL.append(label)
            X_text.append(text)
            if len(X) == self.batch_size:
                yield X, X_text, LABEL
                X, X_text, LABEL = [], [], []
        if len(X) != 0:
            yield X, X_text, LABEL

    def seq_padding(self, X, padding=0):
        L = [len(x) for x in X]
        ML = max(L)
        sequence = []
        seq_len = []
        for x in X:
            seq = list(x)
            seq_ = seq[:ML] + [padding] * max(ML - len(seq), 0)
            sequence.append(seq_)
            seq_len.append(min(len(seq), ML))
        return sequence, seq_len







if __name__ == "__main__":
    kb_path = "./data/kb_data"
    train_path = "./data/train.json"
    develop_path = "./data/develop.json"
    model = read(kb_path, train_path, develop_path)

    model.read_develop_data()
    #model.build_yield_data(model.train_data)
    # 石家庄电视台一套新闻综合频道直播石家庄·民生关注简介