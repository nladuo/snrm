import pymongo

client = pymongo.MongoClient()
db = client.snrm
coll_shuffled = db.aol_queries_shuffled


query_list = []

count = 0

max_len = 0
for q in coll_shuffled.find():
    q_len = len(q["query"].split(" "))
    if q_len > max_len:
        max_len = q_len
    print(count, q_len, max_len, q["query"])
    count += 1


print(max_len)
