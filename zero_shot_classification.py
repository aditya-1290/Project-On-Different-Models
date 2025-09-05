from transformers import pipeline

# Load the zero-shot classification pipeline with the specified model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define custom labels
candidate_labels = ["Finance", "Sports", "Politics"]

# Define 5 example sentences
sentences = [
    "The stock market crashed today due to economic uncertainty.",
    "The football team won the championship after a thrilling match.",
    "The government announced new policies on healthcare reform.",
    "Investors are optimistic about the quarterly earnings report.",
    "The athlete broke the world record in the sprint event."
]

# Classify and print results for each sentence
for sentence in sentences:
    result = classifier(sentence, candidate_labels)
    print(f"Sentence: '{sentence}'")
    print(f"Predicted Label: {result['labels'][0]} (Confidence: {result['scores'][0]:.4f})")
    print(f"All Scores: {dict(zip(result['labels'], result['scores']))}")
    print("-" * 50)
