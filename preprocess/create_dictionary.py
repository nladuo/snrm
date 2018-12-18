import pymongo
import json

client = pymongo.MongoClient()
db = client.snrm
coll = db.docs

# dictionary for frequncy
dictionary = {}

docs = []
for doc in coll.find():
    docs.append(doc)


for i, doc in enumerate(docs):
    token_len = len(doc["tokens"])

    # token length filter
    if token_len == 0 or token_len > 4000:
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
