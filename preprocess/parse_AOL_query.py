import pymongo
import json
import string
from nltk.corpus import stopwords

StopWords = set(stopwords.words('english') + list(string.punctuation))

client = pymongo.MongoClient()
db = client.snrm
coll = db.aol_queries

print(coll.remove({}))


with open("../data/dictionary.json", "r") as f:
    Dictionary = json.load(f)


def parse_query(q):
    terms = q.split(" ")[:]

    if len(terms) > 20:
        return []

    new_terms = []
    for term in terms:
        if term not in StopWords:
            new_terms.append(term)

        if term not in Dictionary:
            return []

    return new_terms


def is_continue(query):
    not_contain = ["http", "www.", ".com", ".net", ".org", ".edu", ".gov", ".html"]

    for term in not_contain:
        if term in query:
            return True

    not_contain_terms = ["www", "com"]
    for term in not_contain_terms:
        if term in query.split(" "):
            return True

    return False

c_list = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

count = 0
for c_id in c_list:
    f_path = "../data/AOL-user-ct-collection/user-ct-test-collection-{c_id}.txt".format(c_id=c_id)

    with open(f_path, "r") as f:
        for line in f.readlines():
            parts = line.split("\t")
            if parts[0] != "AnonID":

                # print(parts)
                query = parts[1].lower()
                terms = parse_query(query)
                if len(terms) == 0:
                    continue

                query = " ".join(terms)

                if is_continue(query):
                    continue

                if coll.find({"query": query}).count() == 0:
                    coll.insert({
                        "query": query,
                        "count": 1,
                        "is_set": False
                    })
                    count += 1
                    print(count, query)
                else:
                    q_count = coll.find_one({"query": query})["count"]
                    coll.update({"query": query}, {
                        "$set": {
                            "count": q_count + 1
                        }
                    })
        # if count > 1050000:
        #     break

