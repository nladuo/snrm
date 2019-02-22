from nltk.tokenize import word_tokenize
import pymongo
import json
from nltk.corpus import stopwords
import string
from multiprocessing import Pool


StopWords = set(stopwords.words('english') + list(string.punctuation))

client = pymongo.MongoClient()
db = client.snrm
coll = db.docs

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


def get_tokens(text):
    text = text.replace("\n", "").lower()

    tokens = []
    words = my_tokenize(text)

    for w in words:
        if w in WordList:
            tokens.append(w)

    return tokens


docs = []
for doc in coll.find():
    docs.append(doc)

CONCURRENT_COUNT = 19  # use 54 cpu for tokenizing


def process_docs(part_id):
    for i, doc in enumerate(docs):
        if i % CONCURRENT_COUNT == part_id:
            text = doc["text"]

            tokens = get_tokens(text)
            coll.update({"_id": doc["_id"]}, {
                "$set": {
                    "tokens": tokens
                }
            })
            print(i, part_id, doc["docNo"])


pool = Pool(CONCURRENT_COUNT)

for part_id in range(CONCURRENT_COUNT):
    pool.apply_async(process_docs, args=(part_id,))
pool.close()
pool.join()

