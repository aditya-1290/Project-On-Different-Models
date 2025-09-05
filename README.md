# NLP Task API

A FastAPI-based web service that provides natural language processing capabilities through four main classification tasks: text summarization, sentiment analysis, question answering, and zero-shot classification.

## Project Structure

```
.
├── fastapi_app/
│   ├── main.py              # FastAPI application with endpoints
│   └── __pycache__/         # Python cache files
├── text_summarization.py    # Standalone text summarization script
├── sentiment_analysis.py    # Standalone sentiment analysis script
├── question_answering.py    # Standalone question answering script
├── zero_shot_classification.py  # Standalone zero-shot classification script
├── download_model.py        # Script to download custom models
├── .gitignore              # Git ignore rules
├── LICENSE                 # Project license
├── nlp_env/                # Virtual environment (ignored)
└── F-urkan_rStar2-Agent-14B-Q4_0-GGUF/  # Downloaded model files (ignored)
```

## Features

### FastAPI Endpoints

The FastAPI application provides the following endpoints:

#### 1. Text Summarization (`POST /summarize`)
- **Model**: `sshleifer/distilbart-cnn-12-6`
- **Input**: `{"text": "Your text to summarize"}`
- **Output**: `{"summary_text": "Generated summary"}`

#### 2. Sentiment Analysis (`POST /sentiment`)
- **Model**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Input**: `{"text": "Your text to analyze"}`
- **Output**: `{"label": "POSITIVE/NEGATIVE", "score": 0.95}`

#### 3. Question Answering (`POST /qa`)
- **Model**: `distilbert-base-uncased-distilled-squad`
- **Input**: `{"question": "Your question", "context": "Context text"}`
- **Output**: `{"answer": "Answer text", "score": 0.85}`

#### 4. Zero-Shot Classification (`POST /classify`)
- **Model**: `facebook/bart-large-mnli`
- **Input**: `{"text": "Your text", "candidate_labels": ["label1", "label2"]}`
- **Output**: `{"label": "best_label", "scores": {"label1": 0.8, "label2": 0.2}}`

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/aditya-1290/Project-On-Different-Models.git
   cd task_nlp
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv nlp_env
   nlp_env\Scripts\activate  # On Windows
   # or
   source nlp_env/bin/activate  # On Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install fastapi uvicorn transformers torch pydantic
   ```

## Usage

### Running the FastAPI Server

```bash
uvicorn fastapi_app.main:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation

Once the server is running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Example API Calls

#### Text Summarization
```bash
curl -X POST "http://localhost:8000/summarize" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your long text here..."}'
```

#### Sentiment Analysis
```bash
curl -X POST "http://localhost:8000/sentiment" \
     -H "Content-Type: application/json" \
     -d '{"text": "I love this product!"}'
```

#### Question Answering
```bash
curl -X POST "http://localhost:8000/qa" \
     -H "Content-Type: application/json" \
     -d '{"question": "Who built the Eiffel Tower?", "context": "The Eiffel Tower was built by Gustave Eiffel..."}'
```

#### Zero-Shot Classification
```bash
curl -X POST "http://localhost:8000/classify" \
     -H "Content-Type: application/json" \
     -d '{"text": "The stock market is crashing", "candidate_labels": ["Finance", "Sports", "Politics"]}'
```

## Standalone Scripts

Each NLP task also has a standalone Python script for direct execution:

### Text Summarization
```bash
python text_summarization.py
```

### Sentiment Analysis
```bash
python sentiment_analysis.py
```

### Question Answering
```bash
python question_answering.py
```

### Zero-Shot Classification
```bash
python zero_shot_classification.py
```

## Model Information

- **Text Summarization**: Uses DistilBART model fine-tuned on CNN/DailyMail dataset
- **Sentiment Analysis**: Uses DistilBERT fine-tuned on SST-2 dataset
- **Question Answering**: Uses DistilBERT fine-tuned on SQuAD dataset
- **Zero-Shot Classification**: Uses BART large model with MNLI fine-tuning

## Custom Model Download

The `download_model.py` script can be used to download additional models from Hugging Face:

```bash
python download_model.py
```

This downloads the `F-urkan/rStar2-Agent-14B-Q4_0-GGUF` model to a local folder.

## Requirements

- Python 3.13.4
- FastAPI
- Uvicorn
- Transformers
- PyTorch
- Pydantic

## License

See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues or questions, please open an issue in the repository.
