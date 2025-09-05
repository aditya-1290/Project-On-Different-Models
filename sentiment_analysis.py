from transformers import pipeline

# Load the sentiment analysis pipeline with the specified model
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Define 3 different sentences for comparison
sentences = [
    "I love this product, it's amazing!",
    "This is the worst experience I've ever had.",
    "The weather today is neither good nor bad."
]

# Analyze and print sentiment for each sentence
for sentence in sentences:
    result = sentiment_pipeline(sentence)
    print(f"Sentence: '{sentence}'")
    print(f"Sentiment: {result[0]['label']} (Confidence: {result[0]['score']:.4f})")
    print("-" * 50)
