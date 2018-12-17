import pymongo
from elasticsearch import Elasticsearch
import json

client = pymongo.MongoClient()
db = client.snrm
coll = db.queries


query = coll.find_one()

query_contains = {
    "query": {
        "match": {
            "text": query["title"]
        }
    },
    "explain": True
}

es = Elasticsearch()
searched = es.search("robo04_index", doc_type="docs", body=query_contains, size=2000)


for i, hit in enumerate(searched["hits"]["hits"]):
    print(i, hit["_id"])


# print(query["title"])
# print(json.dumps(searched))
