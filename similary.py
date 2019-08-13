import re
import collections
from collections import Counter

import json
import distance
from fuzzywuzzy import process, fuzz

import codecs
from gensim import corpora, models, similarities
from nltk.tokenize import WordPunctTokenizer
def edit_distance(s1, s2):
    return distance.levenshtein(s1, s2)
def get_kb_entity(kd2id_path, id2kb_path, result_path):
    kb2id = json.load(open(kd2id_path, "r", encoding="utf-8"))

    id2kb = json.load(open(id2kb_path, "r", encoding="utf-8"))
    result = []
    with open(result_path, "r", encoding="utf-8") as f:
        LINE = f.readlines()
        for index, line in enumerate(LINE[:1]):
            p = {}
            print("index:{}".format(index))
            line = json.loads(line)
            entity = line["entity"]
            text = line["text"]
            #entity_similary = []
            entity_similary = {}

            for e in entity:
                if e in kb2id:
                    #entity_similary.append(e)
                    entity_id = kb2id[e]
                    entity_information = id2kb[entity_id]

                    entity_similary[e] = id2kb[e]
                else:
                    results = list(filter(lambda x: edit_distance(x, e) <= 1, kb2id))
                    if results is None and e in text:
                        #entity_similary.append(e)
                        entity_similary[e] = None
                    elif results is not None:
                        for re in results:
                            if e in re and re in text:
                                #entity_similary.append(re)
                                entity_similary[re] = id2kb[re]
                            elif re in e and e in text:
                                #entity_similary.append(e)

                                entity_similary[re] = id2kb[re]




            p["text"] = text
            p["entity"] = entity
            p["similar_entity"] = entity_similary
            result.append(p)

    f = open("./output/result1.json", "w", encoding="utf-8")
    for e in result:
        e = json.dumps(e, ensure_ascii=False)
        f.write(e + "\n")

def wordtokenizer(sentence):
    words = WordPunctTokenizer().tokenize(sentence)
    return words
def tokenization(text, stopwordpath):
    stopwords = []
    with open(stopwordpath, "r", encoding="gbk") as f:


        LINE = f.readlines()
        for line in LINE:
            line = line.strip()
            stopwords.append(line)


    result = []
    text = re.sub("[-',{:+}|.()/?!·;]", ' ', text).lower()
    words = wordtokenizer(text)
    for word in words:
        if word not in stopwords:
            result.append(word)
    return result
def build_similary(entity, content, kb):
    corpus = []
    for text in kb:

        text = text["subject_desc"]

        text = text.strip().split("\n")[0]
        text = "".join(text)
        corpus.append(tokenization(text, stopwordpath="stopwords.txt"))
    print("corpus:{}".format(corpus))
    for cor in corpus:
        print(cor)
    dictionary = corpora.Dictionary(corpus)
    doc_bow = []
    for text in corpus:
        print("text:{}".format(text))
        doc_bow.append(dictionary.doc2bow(text))
    #doc_bow = [dictionary.doc2bow(text) for text in corpus]
    #print("doc_bow:{}".format(doc_bow))
    tfidf = models.TfidfModel(doc_bow)
    tfidf_bow = tfidf[doc_bow]
    query = tokenization(content, stopwordpath="stopwords.txt")
    query_bow = dictionary.doc2bow(query)
    index = similarities.MatrixSimilarity(tfidf_bow)
    sims = index[query_bow]
    return sims

def get_similary():
    f = open("./output/id2kb_1.json", "r", encoding="utf-8")
    id2kb = json.load(f)

    f1 = open("./data/kb2id.json", "r", encoding="utf-8")
    kb2id = json.load(f1)
    with open("./output/result1.json", "r", encoding="utf-8") as f:
        LINE = f.readlines()
        for line in LINE[9:10]:
            line = json.loads(line)
            text = line["text"]
            print("text:{}".format(text))
            similary_entity = line["similar_entity"]
            for entity in similary_entity:
                print("entity:{}".format(entity))
                if entity in id2kb:
                    print("id:{}".format(kb2id[entity]))
                    entity_kb = id2kb[entity]
                    for kb in entity_kb:
                        print(kb)
                    indent = build_similary(entity, text, entity_kb)
                    print("indent:{}".format(indent))
                    sort = sorted(enumerate(indent), key=lambda x: x[1])  ##b[-1][0] 最大值的原下标
                    print("sort:{}".format(sort))
                    max_index = sort[-1][0]
                    print("max_index:{}".format(max_index))



if __name__ == "__main__":
    kb2id_path = "./data/kb2id.json"
    id2kb_path = "./output/id2kb_1.json"
    result_path = "./output/result.json"
    get_kb_entity(kb2id_path, id2kb_path,  result_path)

    #get_similary()