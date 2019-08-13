import re
import collections
from collections import Counter

import json
import distance



def analysis(path, result_path):
    p1 = {}
    p = []
    with open(path, "r", encoding="utf-8") as f:
        LINE = f.readlines()
        for line in LINE:
            line = json.loads(line)
            subject_id = line["subject_id"]
            subject_alias = list(set([line["subject"]] + line.get("alias", [])))
            subject_desc = "\n".join(u"%s: %s" % (i["predicate"], i["object"]) for i in line["data"])
            subject_desc = subject_desc.lower()
            if subject_desc:
                for sub in subject_alias:
                    id2kb = {}
                    id2kb[sub] = {"subject_id": subject_id, "subject_desc": subject_desc}
                    if sub in p1:
                        p1[sub].append({"subject_id": subject_id, "subject_desc": subject_desc})
                    else:
                        p1[sub] = []
                        p1[sub].append({"subject_id": subject_id, "subject_desc": subject_desc})
    #with open("./output/id2kb.json", "w", encoding="utf-8") as f:
    #     f.write(json.dumps(p, ensure_ascii=False))
    with open("./output/id2kb_1.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(p1, ensure_ascii=False))
def build_link(result_path):
    f = open("./output/id2kb_1.json", "r", encoding="utf-8")
    data = json.load(f)
    with open(result_path, "r", encoding="utf-8") as f:
        LINE= f.readlines()
        for line in LINE[1:2]:
            line = json.loads(line)
            entity = line["entity"]
            text = line["text"]
            print("text:{}".format(text))
            for e in entity:
                if e in list(data.keys()):
                    for value in data[e]:
                        print("e:{}, value:{}".format(e, value))


def analysis_train(train_path):
    p = {}
    with open(train_path, "r", encoding="utf-8") as f:
        LINE = f.readlines()

        for line in LINE:
            line = json.loads(line)
            text = line["text"]
            mention_data = line["mention_data"]
            for m in mention_data:
                mention = m["mention"]
                kb_id = m["kb_id"]
                if mention not in p:
                    p[mention] = []
                    p[mention].append({"kb_id": kb_id, "text": text})
                else:
                    p[mention].append({"kb_id": kb_id, "text": text})
    with open("./output/train_mention2text.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(p, indent = 1, ensure_ascii=False))



def build_train():
    f = open("./output/train_mention2text.json", "r", encoding="utf-8")
    p = json.load(f)
    with open("./output/result1.json", "r", encoding="utf-8") as f:
        LINE = f.readlines()
        for line in LINE[5:6]:
            line = json.loads(line)
            text = line["text"]
            entity = line["similar_entity"]
            #print("entity:{}".format(entity))
            for e in entity:
                if e in p:

                    train = p[e]

                    print("text:{}".format(text))

                    print("train:{}".format(train))

                    print("entity:{}".format(e))
if __name__ == "__main__":
    path = "./data/kb_data"
    train_path = "./data/train.json"
    result_path = "./output/result.json"
    #analysis(path, result_path)
    #analysis_train(train_path)
    build_link(result_path)
    #build_train()
