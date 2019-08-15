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
    "highlight": {
        "fields": {
            "text": {},
        }
    },
}

es = Elasticsearch()
searched = es.search("robo04_index", doc_type="docs", body=query_contains, size=10)


for i, hit in enumerate(searched["hits"]["hits"]):
    print(i, hit)


# print(query["title"])
# print(json.dumps(searched))
