from flask import Flask, request, send_from_directory
from elasticsearch import Elasticsearch
import json

app = Flask(__name__, static_folder='dist')


@app.route("/api/search")
def search():
    query = request.args.get('query')
    page = request.args.get('page')

    offset = (int(page) - 1) * 10

    query_contains = {
        "query": {
            'multi_match': {
                'query': query,
                "fields": ["title", "abstr"]
            }
        },
        "highlight": {
            "fields": {
                "abstr": {},
                "title": {},
            }
        },
    }

    es = Elasticsearch(hosts=[{'host': '10.60.1.101', 'port': 9200}])
    searched = es.search("data", doc_type="data", body=query_contains, size=10, from_=offset)

    return json.dumps(searched)


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('./dist/static', path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

