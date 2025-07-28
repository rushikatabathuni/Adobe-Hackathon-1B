FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install torch sentence-transformers PyMuPDF numpy

# Download bi-encoder model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L12-v2').save('./paraphrase_minilm_l12')"

# Download cross-encoder model
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2').save('./ms-marco-MiniLM-L-12-v2')"

# Copy source code after downloading models
COPY . .

# Run the app
CMD ["python", "main.py"]
