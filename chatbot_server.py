import json
import pickle
import random
import numpy as np
import torch
import nltk
from nltk.stem import PorterStemmer
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import ChatbotModel

app = Flask(__name__)
CORS(app)  

stemmer = PorterStemmer()
words   = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

with open('intents.json', 'r') as f:
    intents = json.load(f)
checkpoint  = torch.load('trained_model.pth', weights_only=True)
model       = ChatbotModel(
    checkpoint['input_size'],
    checkpoint['hidden_size'],
    checkpoint['output_size']
)
model.load_state_dict(checkpoint['model_state'])
model.eval()
def bag_of_words(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens = [stemmer.stem(w.lower()) for w in tokens]
    bag = [1 if w in tokens else 0 for w in words]
    return np.array(bag, dtype=np.float32)
def predict(sentence):
    bow   = bag_of_words(sentence)
    inp   = torch.tensor(bow).unsqueeze(0)
    with torch.no_grad():
        output = model(inp)
    probs     = torch.softmax(output, dim=1)
    prob, idx = torch.max(probs, dim=1)
    confidence = prob.item()
    tag        = classes[idx.item()]
    return tag, confidence
def get_response(sentence):
    tag, confidence = predict(sentence)
    if confidence < 0.6:
        return "I am not sure about that. Please contact us at +92 343 4106919 or WhatsApp for specific queries about SafarWise!"
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Please contact SafarWise at +92 343 4106919 for more information."

@app.route('/chat', methods=['POST'])
def chat():
    data    = request.get_json()
    message = data.get('message', '').strip()
    if not message:
        return jsonify({'response': 'Please type a message.'}), 400
    response = get_response(message)
    return jsonify({'response': response, 'status': 'ok'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'SafarWise chatbot running', 'intents': len(classes)})

if __name__ == '__main__':
    print("SafarWise AI Chatbot Server starting on port 5001...")
    app.run(port=5001, debug=True)