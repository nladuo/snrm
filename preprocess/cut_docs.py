import pymongo


client = pymongo.MongoClient()
db = client.snrm
coll = db.docs
coll2 = db.docs2


count = 0
for doc in coll.find({}):
    del doc["_id"]
    tokens = doc["tokens"]
    if len(tokens) > 250:
        doc["tokens"] = tokens[:250]
    else:
        doc["tokens"] = tokens

    doc["text"] = " ".join(doc["tokens"])
    coll2.insert(doc)
    print(count, doc["docNo"])
    count += 1
