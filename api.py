from flask import Flask, request, jsonify, render_template
import numpy as np
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from app.Model.EmbedModel import NVEmbed
from app.Config.config import *
from huggingface_hub import login
from sentence_transformers import CrossEncoder

import os
from transformers import pipeline
 
ner = pipeline(
    "ner",
    model="Jean-Baptiste/roberta-large-ner-english",
    tokenizer="Jean-Baptiste/roberta-large-ner-english",
    aggregation_strategy="simple",
    device=0
)
# Login to Hugging Face
login('XXXXXXXXXXXXXXXXXXXX')  # Replace with your actual token

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
############################################# NVEmbed #############################################
# Initialize the embedding model
embeddings = NVEmbed(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    show_progress=True,
    query_instruction=query_prefix
)
embeddings.client.max_seq_length = 4096
embeddings.client.tokenizer.padding_side = "right"
embeddings.eos_token = embeddings.client.tokenizer.eos_token
EMBEDDING_DIMENSION = 4096

############################################# NLI #############################################
# Load model once during startup
NLI_model = CrossEncoder('cross-encoder/nli-deberta-v3-xsmall')

# Define label mapping
label_mapping = ['contradiction', 'entailment', 'neutral']

# Flask app setup
app = Flask(__name__, template_folder='app/templates/')

# Thread pool executor with a maximum thread limit
MAX_THREADS = 50  # Set your desired maximum number of threads
executor = ThreadPoolExecutor(max_workers=MAX_THREADS)

# Timeout in seconds for task processing
TASK_TIMEOUT = 1800  # Set the maximum time to wait for task completion

@app.route("/healthz", methods=["GET"])
def healthz():
    # you can add real checks here (DB, GPU, etc.)
    return jsonify(status="ok"), 200

@app.route('/api/NVEmbed', methods=['POST'])
def API_NVembed():
    try:
        content = request.json
        data = content['input']
        type_ = content['type']
    except:
        return jsonify({"error": "Invalid JSON data! Please use the correct JSON format: {'input': 'your_text', 'type': 'query'}"}), 400

    # Create a unique task ID
    task_id = str(uuid.uuid4())

    def process_task():
        try:
            # Process the embedding based on type
            if type_ == 'documents':
                result = embeddings.embed_documents(data)
            elif type_ == 'query':
                result = embeddings.embed_query(data)
            else:
                return {"status": "error", 'message': "Invalid type! Only 'query' and 'documents' are supported."}
        except Exception as e:
            return {"error": str(e)}
        return result

    # Submit the task to the executor
    future = executor.submit(process_task)

    try:
        # Wait for the result with a timeout
        result = future.result(timeout=TASK_TIMEOUT)
    except TimeoutError:
        return jsonify({"status": "error", "message": "Task processing timed out!"}), 504
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    # Return the result directly
    if "error" in result:
        return jsonify({"status": "error", "message": result["error"]}), 500

    return jsonify({'text': data, 'vector': result})

@app.route('/api/NLI', methods=['POST'])
def API_NLI():
    try:
        content = request.json
        sentence_pairs = content['input']
    except:
        return jsonify({"error": "Invalid JSON data! Please use the correct JSON format: {'sentence_pairs': [['sentence1', 'sentence2']] }"}), 400

    # Ensure input is a list of tuple-like pairs
    if not isinstance(sentence_pairs, list) or not all(isinstance(pair, list) and len(pair) == 2 for pair in sentence_pairs):
        return jsonify({"error": "sentence_pairs must be a list of pairs (each pair as a list of two strings)"}), 400

    # Create a unique task ID
    task_id = str(uuid.uuid4())

    def process_task():
        try:
            scores = NLI_model.predict(sentence_pairs)
            # labels = [label_mapping[scores.argmax(axis=1)[i]] for i in range(len(scores))]
            print(scores)
            labels = [(label_mapping[scores.argmax(axis=1)[i]], scores[i].tolist()) for i in range(len(scores))]
            return {"status": "success", "predictions": labels}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # Submit the task to the executor
    future = executor.submit(process_task)

    try:
        # Wait for the result with a timeout
        result = future.result(timeout=TASK_TIMEOUT)
    except TimeoutError:
        return jsonify({"status": "error", "message": "Task processing timed out!"}), 504
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    # Return the result directly
    if result["status"] == "error":
        return jsonify({"status": "error", "message": result["message"]}), 500

    return jsonify(result)

@app.route('/api/ner', methods=['POST'])
def API_ner():
    try:
        content = request.json
        text = content['input']
    except:
        return jsonify({"error": "Invalid JSON data! Please use the correct JSON format: {'input': 'YOUR TEXT'}"}), 400

    # Create a unique task ID
    task_id = str(uuid.uuid4())

    def process_task():
        try:
            result = ner(text)
            # labels = [label_mapping[scores.argmax(axis=1)[i]] for i in range(len(scores))]
            return {"status": "success", "output": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # Submit the task to the executor
    future = executor.submit(process_task)

    try:
        # Wait for the result with a timeout
        result = future.result(timeout=TASK_TIMEOUT)
    except TimeoutError:
        return jsonify({"status": "error", "message": "Task processing timed out!"}), 504
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    # Return the result directly
    if result["status"] == "error":
        return jsonify({"status": "error", "message": result["message"]}), 500
    def convert(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        if isinstance(o, list):
            return [convert(v) for v in o]
        return o

    return jsonify(convert(result))


@app.route('/instruction')
def API_Start():
    return render_template('instruction.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7779)
