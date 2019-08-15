import pymongo
import json
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize

StopWords = set(stopwords.words('english') + list(string.punctuation))

with open("../data/word_list.json", "r") as f:
    WordList = json.load(f)


def my_tokenize(title):
    """ tokenize word """
    title = title.lower()
    words = []
    for word in word_tokenize(title):
        for w in word.split("/"):
            for w2 in w.split("-"):
                if w2 in StopWords:
                    continue
                words.append(w2)
    return words

# dictionary for frequncy
dictionary = {}

with open("triples.train.small.tsv", "r") as f:
    while True:
        line = f.readline()
        if line:
            query = line.split("\t")[0]
            d1 = line.split("\t")[1]
            d2 = line.split("\t")[2]
            print(query)
            print(d1)
            print(d2)
            break
            # if query not in query_map.keys():
            #     query_map[query] = 1
            # else:
            #     query_map[query] += 1
            # print(count, len(query_map), query)
            # count+=1
        else:
            break


docs = []
count = 0


# document tokens add into dictionary
for i, doc in enumerate(docs):
    token_len = len(doc["tokens"])

    # token length filter
    if token_len == 0 or token_len > 2000:
        continue

    tokens = doc["tokens"]
    for token in tokens:
        if token not in dictionary.keys():
            dictionary[token] = 1
        else:
            dictionary[token] += 1

    print(i, doc["docNo"])


with open("../data/dictionary.json", "w") as f:
    json.dump(dictionary, f)

