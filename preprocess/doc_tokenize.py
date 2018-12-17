from nltk.tokenize import word_tokenize
import pymongo
import json

client = pymongo.MongoClient()
db = client.snrm
coll = db.docs

with open("../data/word_list.json", "r") as f:
    WordList = json.load(f)


def get_tokens(text):
    text = text.replace("\n", "").lower()

    tokens = []
    words = word_tokenize(text)

    for w in words:
        if w in WordList:
            tokens.append(w)

    return tokens


docs = []
for doc in coll.find():
    docs.append(doc)

for i, doc in enumerate(docs):
    text = doc["text"]

    tokens = get_tokens(text)
    coll.update({"_id": doc["_id"]}, {
        "$set": {
            "tokens": tokens
        }
    })
    print(i, doc["docNo"])
