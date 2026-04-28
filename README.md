# SafarWise Safi AI Chatbot 🤖

AI powered travel assistant chatbot for SafarWise built with Python + PyTorch + Flask.

## 🔗 Live Demo
[https://abrar00-safarwise-chatbot.hf.space](https://abrar00-safarwise-chatbot.hf.space)

## 🛠️ Built With
- Python
- PyTorch (Neural Network)
- Flask (REST API)
- NLTK (Natural Language Processing)
- Flask-CORS

## ✨ Features
- Natural language understanding
- Pakistan travel specific responses
- 60%+ confidence threshold for accurate answers
- Fallback to contact info for unknown queries
- REST API endpoint for frontend integration

## 📌 API Endpoints
GET  /          # Health check
GET  /health    # Status check
POST /chat      # Send message to chatbot

### Example Request
```json
POST /chat
{
  "message": "Tell me about Hunza Valley"
}
```

### Example Response
```json
{
  "response": "Hunza Valley is one of Pakistan's most beautiful destinations...",
  "status": "ok"
}
```

## 🧠 How It Works
User Message
↓
Tokenize + Stem words (NLTK)
↓
Bag of Words vector
↓
Neural Network (PyTorch)
↓
Intent Classification
↓
Response from intents.json

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation
```bash
git clone https://github.com/abrar-gondal/safarwise-chatbot.git
cd safarwise-chatbot
pip install -r requirements.txt
```

### Run Locally
```bash
python app.py
```

## 📁 Project Structure
├── app.py              # Flask server
├── model.py            # PyTorch model
├── train.py            # Training script
├── intents.json        # Training data
├── words.pkl           # Preprocessed words
├── classes.pkl         # Intent classes
├── trained_model.pth   # Trained model
└── requirements.txt    # Dependencies

## 🌐 Deployment
Deployed on **Hugging Face Spaces** with automatic deployments on every push.

## 👨‍💻 Developer
Abrar Gondal
