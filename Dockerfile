FROM --platform=linux/amd64 python:3.10

WORKDIR /app

COPY requirements.txt .

# Install Python dependencies
RUN pip install torch sentence-transformers PyMuPDF numpy 

# Copy the processing script
COPY . .

RUN mkdir -p /app/input/pdf

# Run the script
CMD ["python", "main.py"] 
