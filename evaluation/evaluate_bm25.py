import pymongo
from elasticsearch import Elasticsearch
import json
import time
import random

es = Elasticsearch()
client = pymongo.MongoClient()
db = client.snrm
query_coll = db.queries

with open("../data/rank_labels.json", "r") as f:
    rank_labels = json.load(f)

queries = []
for q in query_coll.find():
    queries.append(q)
    print(q)


result = {}
for count, q in enumerate(queries):
    if str(q["number"]) not in rank_labels.keys():
        continue
    # if count <= 200:
    #     continue
    query_contains = {
        "query": {
            "match": {
                "text": q["title"]
            }
        },
        "explain": True
    }
    print(q["number"], q["title"], len(rank_labels[str(q["number"])]))
    # searched = es.search("robo04_index2", doc_type="docs", body=query_contains, size=len(rank_labels[str(q["number"])]))
    searched = es.search("robo04_index2", doc_type="docs", body=query_contains, size=1000)

    result[str(q["number"])] = []

    for hit in searched["hits"]["hits"]:
        result[str(q["number"])].append(hit["_id"])


def compute_R_accuracy(rank_labels, result):
    sum_precision = 0
    count = 0
    for qid in result.keys():
        if len(result[qid]) == 0:
            continue

        if qid not in rank_labels.keys():
            continue

        # print(len(result[qid]))
        # print(result[qid])

        for _q in result[qid]:
            if _q in rank_labels[qid].keys():
                sum_precision += 1.0 / len(rank_labels[qid])
                print(sum_precision)
        count += 1
    return sum_precision / count


def compute_MAP(rank_labels, result):
    sum_AP = 0
    count = 0
    for qid in result.keys():
        if len(rank_labels[qid]) == 0:
            continue
        AP = 0
        relevant_count = 0
        for i, docId in enumerate(result[qid]):
            if docId in rank_labels[qid]:
                relevant_count += 1
                AP += float(relevant_count) / (i + 1)

        # if relevant_count != 0:
        #     AP = AP / relevant_count

        # if relevant_count != 0:
        AP /= len(rank_labels[qid])

        print(qid, "-->", AP)
        sum_AP += AP
        count += 1

    return sum_AP / count

# print(result)

# print("R accuracy: ", compute_R_accuracy(rank_labels, result))

print("MAP : ", compute_MAP(rank_labels, result))
