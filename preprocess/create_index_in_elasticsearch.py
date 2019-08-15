from elasticsearch import Elasticsearch
import pymongo

es = Elasticsearch()

index_mappings = {
    "mappings": {
        "docs": {
            "properties": {
                "text": {
                    "type": "text",
                }
            }
        }
    }
}

# es.indices.delete(index='robo04_index')

if es.indices.exists(index='robo04_index') is not True:
    print("create robo04_index")
    es.indices.create(index='robo04_index', body=index_mappings)


client = pymongo.MongoClient()
db = client.snrm
coll = db.docs2


docs = []
for doc in coll.find():
    docs.append(doc)


for count, doc in enumerate(docs):
    _id = str(doc["docNo"])
    text = doc["text"]
    token_len = len(doc["tokens"])

    # token length filter
    if token_len == 0 or token_len > 251:
        continue

    doc = {
        "id": _id,
        "text": text,
    }
    res = es.index(index="robo04_index", doc_type="docs", id=_id, body=doc)
    print(count, res)
