import pymongo
import json

client = pymongo.MongoClient()
db = client.snrm
coll = db.docs


docIds = []


for doc in coll.find({}):
    docIds.append(doc["docNo"])

with open("../data/docIds.json", "w") as f:
    json.dump(docIds, f)
