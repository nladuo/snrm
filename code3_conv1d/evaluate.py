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


print("P@20", compute_MAP(rank_labels, result))


