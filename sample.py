import ollama
import csv
import pandas as pd
import numpy as np
from langchain_community.embeddings import OllamaEmbeddings
from ast import literal_eval
# Define your variables
target_language = "French"  # Example target language
# english_text = input('Your English text here: ')
english_text = 'The quick brown fox jumps over the lazy dog.'
translation_examples = [
    ("The small bird sings in the tall tree.", "Le petit oiseau chante dans le grand arbre."),
    ("The fast horse races across the green field.", "Le cheval rapide court à travers le champ vert."),
    # Add more examples as needed
]

# Construct the translation examples section of the prompt
examples_section = "\n".join([
    f"{indx}. English: {english}\n   {target_language}: {translated}"
    for indx, (english, translated) in enumerate(translation_examples)
])

# Construct the full prompt
prompt = f"""I am seeking to translate: {english_text} into {target_language}:

To assist in providing a more accurate and nuanced translation, I have included several translation examples that should be considered. These examples demonstrate specific language usage, idiomatic expressions, or complex constructions in {target_language} that are relevant to the text needing translation.

Translation Examples:
{examples_section}

Please utilize these examples to guide the translation process, paying special attention to the nuances and specific usage demonstrated. The goal is to produce a translation that is not only accurate but also reflects the linguistic characteristics and subtleties highlighted in the examples provided.

"""

# print(prompt)

response = ollama.chat(model='llama2', messages=[
  {
     'role': 'user',
     'content': prompt,
   },
])

print(response['message']['content'])