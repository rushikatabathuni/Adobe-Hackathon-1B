# Adobe Hackathon Round 1B - PDF Content Extraction & Analysis

A AI-powered system that extracts, ranks, and analyzes relevant content from PDF documents based on specific persona roles and job tasks. Built with advanced NLP techniques using semantic similarity and cross-encoder models, meeting all requirements

## 🚀 Features

- **Intelligent PDF Processing**: Extracts structured sections from PDFs using outline information
- **Semantic Content Ranking**: Uses bi-encoder and cross-encoder models for accurate relevance scoring
- **Persona-Aware Analysis**: Tailors content extraction based on specific user roles and tasks
- **Constraint-Based Filtering**: Supports inclusion/exclusion terms for precise content selection
- **Quality Assessment**: Evaluates content quality based on multiple factors including length, structure, and position
- **Concurrent Processing**: Multi-threaded PDF processing for improved performance

## 🏗️ Architecture

The system consists of several key components:

- **PDF Processing** (`pdf_utils.py`): Extracts text and outline structure from PDFs
- **Section Extraction** (`section_extraction.py`): Converts PDF outlines into structured sections
- **Relevance Ranking** (`relevance_ranking.py`): Advanced ranking system with semantic analysis
- **Output Formatting** (`formatter.py`): Structures the final JSON output
- **Main Pipeline** (`main.py`): Orchestrates the entire processing workflow

## 📁 Project Structure

```
project/
├── input/
│   ├── your_input_file.json    # Configuration file (place here)
│   └── pdf/               # PDF documents directory
│       ├── #PLACE YOUR PDFS HERE
├── output/
│   └── output.json        # Generated results
├── cross-encoder-ms-marco-MiniLM-L-12-v2 #Cross Encoder Model files
├── paraphrase_minilm_l12 #LM Model files
│
├── utils/
│   ├── pdf_utils.py
│   ├── section_extraction.py
│   ├── relevance_ranking.py
│   └── formatter.py
├── main.py
├──README.md
└── requirements.txt
```

## 🔧 Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rushikatabathuni/Adobe-Hackathon-1B
   cd Adobe-Hackathon-1B
   ```

2. **Install dependencies**
   ```bash
   pip install torch sentence-transformers
   pip install PyMuPDF  # for PDF processing
   pip install numpy
   ```

3. **Download required models**
   The repo contains the downloaded models:
   - `paraphrase_minilm_l12/` (Bi-encoder)
   - `cross-encoder-ms-marco-MiniLM-L-12-v2/` (Cross-encoder)

## 📝 Input Configuration

### Directory Setup

1. **Place your input configuration**: Put your `input.json` file in the `input/` directory
2. **Add PDF documents**: Place all PDF files in the `input/pdf/` directory

### Input JSON Format

Create an `input.json` file with the following structure:

```json
{
    "challenge_info": {
        "challenge_id": "round_1b_003",
        "test_case_name": "create_manageable_forms",
        "description": "Creating manageable forms"
    },
    "documents": [
        {
            "filename": "Learn Acrobat - Create and Convert_1.pdf",
            "title": "Learn Acrobat - Create and Convert_1"
        }
    ],
    "persona": {
        "role": "HR professional"
    },
    "job_to_be_done": {
        "task": "Create and manage fillable forms for onboarding and compliance."
    }
}
```

### Required Fields

- **documents**: Array of objects with `filename` field pointing to PDFs in `input/pdf/`
- **persona.role**: The role/persona for content analysis
- **job_to_be_done.task**: Specific task description with optional inclusion/exclusion constraints

## 🚀 Usage

Run the main processing pipeline:

```bash
python main.py
```

The system will:
1. Load and validate the input configuration
2. Process all PDF documents concurrently
3. Extract and structure content sections
4. Rank sections based on relevance to the task
5. Extract refined paragraphs from top sections
6. Generate the final output JSON

## 📊 Output Format

The system generates an `output.json` file in the `output/` directory with:

```json
{
  "metadata": {
    "input_documents": ["doc1.pdf", "doc2.pdf"],
    "persona": "Software Engineer",
    "job_to_be_done": "Find API documentation...",
    "processing_timestamp": "2024-01-20T10:30:00"
  },
  "extracted_sections": [
    {
      "document": "doc1.pdf",
      "section_title": "API Reference",
      "importance_rank": 1,
      "page_number": 15
    }
  ],
  "subsection_analysis": [
    {
      "document": "doc1.pdf",
      "refined_text": "The API provides methods for...",
      "page_number": 15
    }
  ]
}
```

## 🔄 Code Workflow

### 1. Input Validation & Loading (`main.py`)
```
load_input_json() → Validates required fields (documents, persona, job_to_be_done)
```
- Checks for `input.json` in `input/` directory
- Validates presence of required fields: documents, persona.role, job_to_be_done.task
- Loads configuration for processing pipeline

### 2. PDF Processing (`pdf_utils.py` + `section_extraction.py`)
```
For each PDF:
├── extract_outline_and_text() → Extracts PDF structure and text
├── extract_sections_from_outline() → Converts outline to structured sections
└── Returns: [{document, section_title, page_number, content}, ...]
```
- **Concurrent Processing**: Uses ThreadPoolExecutor for parallel PDF processing
- **Content Extraction**: Extracts both outline structure and plain text from each page
- **Section Mapping**: Maps outline headings to their corresponding content using y-coordinates
- **Quality Validation**: Filters out sections with insufficient content (<10 chars + < 3 char titles)

### 3. Task Analysis & Constraint Extraction (`relevance_ranking.py`)
```
extract_key_concepts_and_constraints():
├── NLP Processing → Extracts noun phrases as key concepts
├── Pattern Matching → Identifies inclusion/exclusion terms
├── Embedding Generation → Creates semantic vectors for all terms
└── Returns: {key_concepts, inclusion_terms, exclusion_terms, embeddings}
```
- **Key Concept Extraction**: Uses regex to find meaningful 2-4 word noun phrases
- **Constraint Detection**: Identifies inclusion patterns ("including", "focus on") and exclusion patterns ("excluding", "avoid")
- **Semantic Embeddings**: Generates vector representations using bi-encoder model

### 4. Section Ranking (`rank_sections()`)
```
For each section:
├── Exclusion Filtering → Remove sections violating constraints
├── Semantic Alignment → Calculate relevance using embeddings
├── Quality Assessment → Score based on length, structure, position
├── Initial Ranking → Combine alignment + quality + document relevance
└── Cross-Encoder Reranking → Final precise ranking of top candidates
```

**Multi-Stage Ranking Process:**
- **Stage 1**: Bi-encoder similarity between task and section content
- **Stage 2**: Quality scoring (content length, title quality, document position)
- **Stage 3**: Cross-encoder reranking of top candidates for final precision
- **Diversity Control**: Limits sections per document to ensure variety

### 5. Content Extraction (`extract_top_paragraphs()`)
```
For each top section:
├── Paragraph Segmentation → Split content into meaningful paragraphs
├── Cross-Encoder Scoring → Rank paragraphs by relevance to task
├── Best Paragraph Selection → Choose highest scoring paragraph
├── Snippet Extraction → Extract focused 60-word snippets
└── Context Preservation → Add surrounding sentences for clarity
```

**Snippet Refinement Process:**
- **Sentence Boundary Detection**: Uses regex to identify clean sentence breaks
- **Best Sentence Selection**: Cross-encoder finds most relevant sentence
- **Context Addition**: Adds surrounding sentences within word limit
- **Natural Breaking**: Breaks at punctuation or conjunctions for readability

### 6. Output Generation (`formatter.py`)
```
format_output():
├── Metadata Assembly → Input docs, persona, task, timestamp
├── Section Structuring → Top sections with ranks and page numbers
├── Paragraph Compilation → Refined text snippets with source info
└── JSON Export → Final structured output to output/output.json
```

### 7. Quality Assurance & Validation
Throughout the pipeline:
- **Content Validation**: Ensures minimum content thresholds
- **Order Verification**: Maintains document-section-paragraph continuity
- **Error Handling**: Graceful degradation with fallback mechanisms
- **Progress Tracking**: Detailed logging of each processing stage

### Processing Flow Summary:
```
Input JSON + PDFs → PDF Processing → Task Analysis → Section Ranking → 
Content Extraction → Output Generation → output.json
```

**Key Algorithms Used:**
- **Bi-encoder**: Fast semantic similarity (paraphrase-MiniLM-L12)
- **Cross-encoder**: Precise relevance ranking (ms-marco-MiniLM-L-12-v2)
- **Constraint Matching**: Pattern-based inclusion/exclusion detection
- **Quality Scoring**: Multi-factor content assessment

##  Configuration Options

Key parameters can be adjusted in the code:

- `top_n`: Number of top sections to extract (default: 5)
- `max_words`: Maximum words in refined snippets (default: 60)
- `batch_size`: Processing batch size (default: 32)
- `max_workers`: Concurrent processing threads (default: 8)


### Error Messages

- `"Missing input JSON"`: Place `input.json` in `input/` directory
- `"PDF file not found"`: Check PDF files are in `input/pdf/`
- `"No sections extracted"`: PDF may be image-based or corrupted

## 📈 Model Information

- **Bi-encoder**: `paraphrase_minilm_l12/` - Fast semantic similarity
- **Cross-encoder**: `cross-encoder-ms-marco-MiniLM-L-12-v2/` - Accurate relevance ranking

These models provide state-of-the-art performance for document retrieval and ranking tasks.
