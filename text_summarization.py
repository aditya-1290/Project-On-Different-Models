from transformers import pipeline

# Load the summarization pipeline with the specified model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Sample Wikipedia article text (~200 words)
article_text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed 
by humans and animals. Leading AI textbooks define the field as the study of "intelligent agents": any device that 
perceives its environment and takes actions that maximize its chance of successfully achieving its goals. 
Colloquially, the term "artificial intelligence" is often used to describe machines (or computers) that mimic 
"cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving". As machines 
become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a 
phenomenon known as the AI effect. A quip in Tesler's Theorem says "AI is whatever hasn't been done yet." For instance, 
optical character recognition is frequently excluded from things considered to be AI, having become a routine technology. 
Modern machine learning techniques, including deep learning, have enabled the use of AI in a wide range of applications, 
from self-driving cars to medical diagnosis, and even in creative fields like art and music generation. However, AI also 
raises ethical concerns, including job displacement, privacy issues, and the potential for autonomous weapons. 
Researchers continue to explore ways to make AI more transparent, fair, and aligned with human values.
"""

# Generate summary (~50 words)
summary = summarizer(article_text, max_length=50, min_length=25, do_sample=False)

print("Original Article (first 200 words):")
print(article_text[:200] + "...")
print("\nSummary (~50 words):")
print(summary[0]['summary_text'])
