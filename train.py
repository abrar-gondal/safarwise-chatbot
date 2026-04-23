import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import nltk
from nltk.stem import PorterStemmer
from model import ChatbotModel

stemmer = PorterStemmer()

with open('intents.json', 'r') as f:
    intents = json.load(f)
words = []
classes = []
documents = []
ignore = ['?', '!', '.', ',', "'", '"']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(f"Words: {len(words)}")
print(f"Classes: {len(classes)}")
print(f"Documents: {len(documents)}")

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    word_patterns = [stemmer.stem(w.lower()) for w in doc[0]]
    for w in words:
        bag.append(1 if w in word_patterns else 0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])
import random
random.shuffle(training)
training = np.array(training, dtype=object)

X = np.array(list(training[:, 0]), dtype=np.float32)
y = np.array(list(training[:, 1]), dtype=np.float32)

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

dataset = TensorDataset(X_tensor, y_tensor)
loader  = DataLoader(dataset, batch_size=8, shuffle=True)

input_size  = len(X[0])
hidden_size = 128
output_size = len(classes)
model     = ChatbotModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 500
print("\nTraining started...")
for epoch in range(EPOCHS):
    total_loss = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss   = criterion(output, torch.argmax(y_batch, dim=1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {total_loss:.4f}")
torch.save({
    'model_state': model.state_dict(),
    'input_size':  input_size,
    'hidden_size': hidden_size,
    'output_size': output_size,
}, 'trained_model.pth')
print("\nModel trained and saved successfully!")
print(f"Training complete — {len(classes)} intents learned.")