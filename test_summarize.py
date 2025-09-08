from transformers import pipeline

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

text = """Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed
by humans and animals. Leading AI textbooks define the field as the study of intelligent agents : any device that
perceives its environment and takes actions that maximize its chance of successfully achieving its goals.
Colloquially, the term artificial intelligence is often used to describe machines (or computers) that mimic
cognitive functions that humans associate with the human mind, such as learning and problem solving. As machines
become increasingly capable, tasks considered to require intelligence are often removed from the definition of AI, a
phenomenon known as the AI effect. A quip in Teslers Theorem says AI is whatever has not been done yet. For instance,
optical character recognition is frequently excluded from things considered to be AI, having become a routine technology.
Modern machine learning techniques, including deep learning, have enabled the use of AI in a wide range of applications,
from self-driving cars to medical diagnosis, and even in creative fields like art and music generation. However, AI also
raises ethical concerns, including job displacement, privacy issues, and the potential for autonomous weapons.
Researchers continue to explore ways to make AI more transparent, fair, and aligned with human values."""

try:
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    print("Summary:", summary[0]['summary_text'])
except Exception as e:
    print("Error:", e)
