from params import FLAGS
import pickle as pkl
import json
import pymongo


# extract labels
with open("../data/rank_labels.json", "r") as f:
    rank_labels = json.load(f)


# get sorted result
with open(FLAGS.base_path + FLAGS.result_path + FLAGS.run_name + '-test-queries.pkl', "rb") as f:
    result = pkl.load(f)

# get queries
client = pymongo.MongoClient()
db = client.snrm
query_coll = db.queries

queries = {}
for q in query_coll.find({}):
    qid = str(q["number"])
    queries[qid] = q["title"].lower()

count = 0
for qid in result.keys():
    if len(result[qid]) == 0:
        continue
    print(queries[qid])
    print(len(result[qid]))
    print(result[qid][-20:][0][0])
    count += 1

print("count-->", count)


def computed_precision_at_20(rank_labels, result):
    sum_precision = 0
    count = 0
    for qid in result.keys():
        if len(result[qid]) == 0:
            continue

        if qid not in rank_labels.keys():
            continue

        print(queries[qid])
        print(len(result[qid]))
        print(result[qid][-20:])

        for _q in result[qid][-100:]:
            if _q[0] in rank_labels[qid].keys():
                sum_precision += 0.01
        count += 1

    return sum_precision / count


print("P@20", computed_precision_at_20(rank_labels, result))


