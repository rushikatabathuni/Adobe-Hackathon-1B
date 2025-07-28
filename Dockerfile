FROM --platform=linux/amd64 python:3.10

WORKDIR /app

# Install Python dependencies
RUN pip install torch sentence-transformers PyMuPDF numpy 

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-MiniLM-L12-v2')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')"


# Copy the processing script
COPY . .

RUN mkdir -p /app/input/pdf

# Run the script
CMD ["python", "main.py"] 
