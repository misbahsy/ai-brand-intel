from flask import Flask, request
from flask_cors import CORS
from langchain_funcs import *
import json

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return {"Hello":"World"}

@app.route('/embed', methods=['POST'])
def embed():
    scope = request.json.get("scope")
    department = request.json.get("department")
    userid = request.json.get("userid")
    filetype = request.json.get("filetype")
    input_url = request.json.get("input_url")
    s3_path = request.json.get("s3_path")

    collection_name = generate_embeddings(scope, department, userid, filetype, input_url=input_url, s3_path=s3_path)
    return {"collection_name":collection_name}

@app.route('/qsearch', methods=['POST'])
def search():
    query = request.json.get("query")
    collection_name = request.json.get("collection_name")
    filter_dict = request.json.get("filter_dict")
    k = request.json.get("k")
    with_source = request.json.get("with_source")

    search_result = qdrant_search_completion(query, collection_name, filter_dict,k,with_source)
    json_result = parse_result(search_result)
    return json_result

@app.route('/vsearch', methods=['POST'])
def vector_search():
    query = request.json.get("query")
    collection_name = request.json.get("collection_name")
    filter_dict = request.json.get("filter_dict")
    k = request.json.get("k")
    with_source = request.json.get("with_source")

    search_result = qdrant_search_vectors(query, collection_name, filter_dict,k,with_source)
    json_docs_result = parse_docs(search_result)
    return json_docs_result


# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0')
