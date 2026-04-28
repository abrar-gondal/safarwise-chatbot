import os
import sys
import json
import pickle
import random
import numpy as np
import torch
import nltk
from nltk.stem import PorterStemmer
from flask import Flask, request, jsonify
from flask_cors import CORS

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from model import ChatbotModel

app = Flask(__name__)

CORS(app, origins=[
    "https://safarwise-frontend.vercel.app",
    "http://localhost:5173",
])

stemmer = PorterStemmer()

words   = pickle.load(open(os.path.join(BASE_DIR, 'words.pkl'),   'rb'))
classes = pickle.load(open(os.path.join(BASE_DIR, 'classes.pkl'), 'rb'))

with open(os.path.join(BASE_DIR, 'intents.json'), 'r', encoding='utf-8') as f:
    intents = json.load(f)

checkpoint = torch.load(
    os.path.join(BASE_DIR, 'trained_model.pth'),
    map_location=torch.device('cpu'),
    weights_only=True,
)
model = ChatbotModel(
    checkpoint['input_size'],
    checkpoint['hidden_size'],
    checkpoint['output_size'],
)
model.load_state_dict(checkpoint['model_state'])
model.eval()

print(f"Safi AI ready — {len(classes)} intents loaded")

def bag_of_words(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens = [stemmer.stem(w.lower()) for w in tokens]
    return np.array([1 if w in tokens else 0 for w in words], dtype=np.float32)

def get_response(sentence):
    bow        = bag_of_words(sentence)
    inp        = torch.tensor(bow).unsqueeze(0)
    with torch.no_grad():
        output = model(inp)
    probs      = torch.softmax(output, dim=1)
    prob, idx  = torch.max(probs, dim=1)
    confidence = prob.item()
    tag        = classes[idx.item()]

    if confidence < 0.6:
        return "I am not sure about that. Please contact us at +92 343 4106919 or WhatsApp for specific queries about SafarWise!"

    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

    return "Please contact SafarWise at +92 343 4106919 for more information."

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'SafarWise Safi AI Chatbot', 'status': 'running'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running', 'intents': len(classes)})

@app.route('/chat', methods=['POST'])
def chat():
    data    = request.get_json()
    message = data.get('message', '').strip()
    if not message:
        return jsonify({'response': 'Please type a message.'}), 400
    return jsonify({'response': get_response(message), 'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)