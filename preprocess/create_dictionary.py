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


client = pymongo.MongoClient()
db = client.snrm
doc_coll = db.docs
query_coll = db.queries
aol_query_coll = db.aol_queries

# dictionary for frequncy
dictionary = {}

docs = []
count = 0
for doc in doc_coll.find():
    if count % 1000 == 0:
        print(count, doc["docNo"])
    docs.append(doc)
    count += 1

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


# query tokens add into dictionary
for q in query_coll.find():
    title = q["title"]
    print(title)
    for token in my_tokenize(title):
        if token in WordList:
            if token not in dictionary.keys():
                dictionary[token] = 21   # 20 is the minimum frequency for dictionary.
            else:
                dictionary[token] += 21


# query tokens add into dictionary
for q in aol_query_coll.find():
    title = q["query"]
    print(title)
    for token in my_tokenize(title):
        if token in WordList:
            if token not in dictionary.keys():
                dictionary[token] = 21   # 20 is the minimum frequency for dictionary.
            else:
                dictionary[token] += 21


with open("../data/dictionary.json", "w") as f:
    json.dump(dictionary, f)
