import pymongo
import random
import json

client = pymongo.MongoClient()
db = client.snrm
pair_wise_data_coll = db.pair_wise_data


pair_wise_data = []

count = 0
for data in pair_wise_data_coll.find({}):
    del data["_id"]
    count += 1
    pair_wise_data.append(data)
    print(count, data["q"])


random.shuffle(pair_wise_data)
with open("../data/pair_wise_data.json", "w") as f:
    json.dump(pair_wise_data, f)
