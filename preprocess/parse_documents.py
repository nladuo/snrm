import pymongo
from bs4 import BeautifulSoup

client = pymongo.MongoClient()
db = client.snrm
coll = db.docs

TYPE_NONE = -1
TYPE_DATE = 0
TYPE_TEXT = 1


index = 0
status = TYPE_NONE
doc = {
    "docNo": "",
    "text": "",
}

files = [
    "fbis5.dat",
    "fr94.dat",
    "ft91.dat",
    "ft92.dat",
    "ft93.dat",
    "ft94.dat",
    "latimes.dat",
]

for f_name in files:
    print(f_name)
    with open("../data/src-data/{f_name}".format(f_name=f_name), "rb") as f:
        for line in f.readlines():
            line = line.decode("utf-8", errors='ignore')
            if line.startswith("</DOC>"):
                doc["docNo"] = doc["docNo"].strip(" ")
                soup = BeautifulSoup(doc["text"].strip(" "), "lxml")
                doc["text"] = soup.get_text()
                print(doc["docNo"])

                if coll.find({"docNo": doc["docNo"]}).count() == 0:
                    coll.insert(doc)
                doc = {
                    "docNo": "",
                    "text": "",
                }
            elif line.startswith("<DOCNO>"):
                doc["docNo"] = line.replace("<DOCNO>", "").replace("</DOCNO>", "").replace("\n", "")
            elif line.startswith("<TEXT>"):
                status = TYPE_TEXT
            elif line.startswith("</TEXT>"):
                status = TYPE_NONE
            else:
                if line != "\n" and (not line.startswith("<!--")):
                    if status == TYPE_TEXT:
                        doc["text"] += line
