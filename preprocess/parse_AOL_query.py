import pymongo
import os
import json

client = pymongo.MongoClient()
db = client.snrm
coll = db.aol_queries

coll.remove({})


with open("../data/word_list.json", "r") as f:
    WordList = json.load(f)

AOL_LOG_DIR = "../data/aol_query_log_analysis/user_search_logs/"


def is_record(q, sites):
    terms = q.split(" ")[:]

    if len(terms) > 20:
        return False

    for site in sites:
        c = 0
        for t in terms:
            if t in site:
                c += 1
        if len(terms) <= c:
            return False

    for term in terms:
        if term not in WordList:
            return False

    return True


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

for f_name in os.listdir(AOL_LOG_DIR):
    f_path = AOL_LOG_DIR + f_name

    print(f_name, coll.count())

    with open(f_path, "r") as f:
        for log in f.read().split("\n\n"):
            query = log.split("\t")[0]

            sites = log.split("\n")[1:]

            if is_continue(query):
                continue

            if len(sites) > 0:
                if is_record(query, sites):
                    if coll.find({"query": query}).count() == 0:
                        coll.insert({
                            "query": query,
                            "count": 1
                        })
                    else:
                        q_count = coll.find_one({"query": query})["count"]
                        coll.update({"query": query}, {
                            "$set": {
                                "count": q_count + 1
                            }
                        })

# coll.remove({"count": 1})

