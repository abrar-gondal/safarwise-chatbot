# Use Python 3.11 — stable and works with PyTorch
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data during build
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Copy all chatbot files
COPY . .

# Hugging Face uses port 7860
EXPOSE 7860

# Start the Flask server
CMD ["python", "app.py"]