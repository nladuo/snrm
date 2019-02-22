from elasticsearch import Elasticsearch
import pymongo

es = Elasticsearch()

index_mappings = {
    "settings": {
        "index": {
            "similarity": {
              "my_similarity": {
                "type": "LMJelinekMercer",
                "lambda": 0.1
              }
            }
        },
        "number_of_shards": 1
    },
    "mappings": {
        "docs": {
            "properties": {
                "text": {
                    "type": "text",
                    "similarity": "my_similarity"
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
coll = db.docs


docs = []
for doc in coll.find():
    docs.append(doc)


for doc in docs:
    _id = str(doc["docNo"])
    text = doc["text"]
    token_len = len(doc["tokens"])

    # token length filter
    if token_len == 0 or token_len > 2000:
        continue

    doc = {
        "id": _id,
        "text": text,
    }
    res = es.index(index="robo04_index", doc_type="docs", id=_id, body=doc)
    print(res)

