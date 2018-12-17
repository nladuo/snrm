import json

GLOVE_PATH = "../data/glove.6B.300d.txt"

word_list = []
with open(GLOVE_PATH) as f:
    i = 1
    for line in f:
        values = line.split()
        word = values[0]
        print(i, word)
        i += 1
        word_list.append(word)

with open("../data/word_list.json", "w") as f:
    json.dump(word_list, f)


