from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from transformers import pipeline

app = FastAPI()

# Load pipelines
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Request and Response Models
class TextRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    label: str
    score: float

class QARequest(BaseModel):
    question: str
    context: str

class QAResponse(BaseModel):
    answer: str
    score: float

class ClassificationRequest(BaseModel):
    text: str
    candidate_labels: List[str]

class ClassificationResponse(BaseModel):
    label: str
    scores: dict

# Endpoints
@app.post("/summarize")
def summarize(request: TextRequest):
    try:
        summary = summarizer(request.text, max_length=50, min_length=25, do_sample=False, truncation=True)
        return {"summary_text": summary[0]['summary_text']}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/sentiment", response_model=SentimentResponse)
def sentiment_analysis(request: TextRequest):
    result = sentiment_pipeline(request.text)[0]
    return SentimentResponse(label=result['label'], score=result['score'])

@app.post("/qa", response_model=QAResponse)
def question_answering(request: QARequest):
    result = qa_pipeline(question=request.question, context=request.context)
    return QAResponse(answer=result['answer'], score=result['score'])

@app.post("/classify", response_model=ClassificationResponse)
def zero_shot_classification(request: ClassificationRequest):
    result = classifier(request.text, request.candidate_labels)
    scores_dict = dict(zip(result['labels'], result['scores']))
    return ClassificationResponse(label=result['labels'][0], scores=scores_dict)
