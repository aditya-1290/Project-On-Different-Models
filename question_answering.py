from transformers import pipeline

# Load the question-answering pipeline with the specified model
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Provide a context paragraph
context = """
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, 
whose company designed and built the tower. Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, it was initially 
criticized by some of France's leading artists and intellectuals for its design, but it has become a global cultural icon of France and 
one of the most recognizable structures in the world. The Eiffel Tower is the most-visited paid monument in the world; 6.91 million 
people ascended it in 2015. The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest 
structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed 
the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in 
New York City was finished in 1930.
"""

# Define 2-3 questions
questions = [
    "Who designed the Eiffel Tower?",
    "How tall is the Eiffel Tower?",
    "When was the Eiffel Tower constructed?"
]

# Answer each question and print with confidence scores
for question in questions:
    result = qa_pipeline(question=question, context=context)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence Score: {result['score']:.4f}")
    print("-" * 50)
