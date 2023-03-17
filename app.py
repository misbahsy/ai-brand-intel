from flask import Flask, request
from flask_cors import CORS
from langchain_funcs import *
import json

app = Flask(__name__)

@app.route('/')
def hello_world():
    return '<p>Hello, World!</p>'

@app.route('/embed', methods=['POST'])
def embed():
    scope = request.json.get("scope")
    department = request.json.get("department")
    userid = request.json.get("userid")
    filetype = request.json.get("filetype")
    scope = request.json.get("scope")
    input_url = request.json.get("input_url")
    s3_path = request.json.get("s3_path")

    collection_name = generate_embeddings(scope, department, userid, filetype, input_url=input_url, s3_path=s3_path)
    return {"collection_name":collection_name}


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
