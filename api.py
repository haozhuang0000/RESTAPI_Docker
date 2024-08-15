from flask import Flask, request, jsonify, render_template
import json
from app.Model.EmbedModel import NVEmbed, SciBertEmbed
from app.Config.config import *
from huggingface_hub import login

login(huggingface_token)
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

embeddings = NVEmbed(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    show_progress=True,
    #    multi_process=True,
    query_instruction=query_prefix
)
embeddings.client.max_seq_length = 4096
embeddings.client.tokenizer.padding_side = "right"
embeddings.eos_token = embeddings.client.tokenizer.eos_token
EMBEDDING_DIMENSION = 4096

scibert_embeddings = SciBertEmbed(
    model_name=scibert_model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    show_progress=True,
    #    multi_process=True,
)
scibert_embeddings.client.max_seq_length = 768
scibert_embeddings.client.tokenizer.padding_side = "right"
scibert_embeddings.eos_token = scibert_embeddings.client.tokenizer.eos_token
EMBEDDING_DIMENSION = 768

# create a Flask instance
app = Flask(__name__, template_folder='app/templates/')


# app.config['DEBUG'] = True

@app.route('/instruction')
def API_Start():
    return render_template('instruction.html')


# our '/api' url
# requires user integer argument: value
# returns error message if wrong arguments are passed.
@app.route('/api/NVEmbed', methods=['POST'])
def API_NVembed():
    try:
        content = request.json
        data = content['input']
        type_ = content['type']
    except:
        return "Invalid JSON data! Please input correct json format! Example json format: \n {'input': 'how are you', 'type': 'query'}"

    # # return "Hello world!"
    if type_ == 'documents':
        return embeddings.embed_documents(data)
    elif type_ == 'query':
        return embeddings.embed_query(data)
    else:
        return "Only type=query & type=documents availables!"


@app.route('/api/SciBertEmbed', methods=['POST'])
def API_SciBert():
    try:
        content = request.json
        data = content['input']
        type_ = content['type']
        print(data)
    except:
        return "Invalid JSON data! Please input correct json format! Example json format: \n {'input': 'how are you', 'type': 'query'}"

    # # return "Hello world!"
    if type_ == 'documents':
        return scibert_embeddings.embed_documents(data)
    elif type_ == 'query':
        return scibert_embeddings.embed_query(data)
    else:
        return "Only type=query & type=documents availables!"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)