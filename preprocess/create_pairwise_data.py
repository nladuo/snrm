import pymongo
from elasticsearch import Elasticsearch
import json
import time

es = Elasticsearch()
client = pymongo.MongoClient()
db = client.snrm
query_coll = db.queries


pair_wise_data = []

for count, q in enumerate(query_coll.find()):
    query_contains = {
        "query": {
            "match": {
                "text": q["title"]
            }
        },
        "explain": True
    }
    while True:
        try:
            searched = es.search("robo04_index", doc_type="docs", body=query_contains, size=200)
            break
        except Exception as ex:
            print(ex)
            time.sleep(1)
    print(count, q["title"])
    for i, hit in enumerate(searched["hits"]["hits"]):
        print("\t", count, i, hit["_id"])
        for j, hit2 in enumerate(searched["hits"]["hits"]):
            if i != j:
                q_title = q["title"]
                d1_id = hit["_id"]
                d2_id = hit2["_id"]

                if i > j:
                    label = 1
                else:
                    label = -1
                pair_wise_data.append({
                    "q": q_title,
                    "d1_id": d1_id,
                    "d2_id": d2_id,
                    "label": label,
                })

with open("../data/pair_wise_data.json", "w") as f:
    json.dump(pair_wise_data, f)
