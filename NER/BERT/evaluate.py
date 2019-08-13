import numpy as np
import re
import string

test_path = "H:/conpetition/competition_12/BERT/event_type_entity_extract_eval.csv"
test_label = "H:/conpetition/competition_12/BERT/output_1/label_test.csv"
label_map = {'S': 1, 'B': 2, 'I': 3, 'E': 4, 'O': 5, 'X': 6, '[CLS]': 7, '[SEP]': 8}


data = []
test_id = []

with open(test_path, "r", encoding="utf-8") as f:
    LINE = f.readlines()
    for i, line in enumerate(LINE):

        line = line.strip().split(",")
        line = [l[1:-1].strip() for l in line]
        id = line[0]
        content = "".join(line[1:-1])
        content = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", content)
        # print("content:{}".format(content))
        content = [con for con in content if con not in string.punctuation]
        content = " ".join(content)

        content_split = content.split(" ")
        content = [con for con in content_split if con != None]
        content = "".join(content)


        cont = [con for con in content]
        cont = " ".join(cont)
        print("cont:{}".format(cont))

        test_id.append(id)
        data.append(cont)
    print("data:{}".format(data))

TEST_LABEL = []
with open(test_label, "r", encoding="utf-8") as f:
    LABEL = f.readlines()
    for label in LABEL:
        label = label.strip().split("\t")
        print("label:{}".format(label))
        Label = []
        for i in label[1:]:
            if i in ["1", "2", "3", "4"]:
                Label.append(int(i))
            else:
                Label.append(0)
        print("label:{}".format(Label))
        TEST_LABEL.append(Label)
Entity = []
for label, sent in zip(TEST_LABEL, data):
    print("sent:{}".format(sent))
    index = np.where(np.array(label) > 0)[0]
    #print("index:{}".format(index))
    sent = sent.strip().split(" ")
    sent = "".join(sent)
    #print("content:{}".format(sent))
    entity = [sent[i] for i in index]
    entity = "".join(entity)


    Entity.append(entity)
print("Entity:{}".format(Entity))

for id, entity in zip(test_id, Entity):
    result = id + "," + str(entity) + "\n"

    with open("results.txt", "a", encoding="utf-8") as f:
        f.write(result)


