import ollama
import csv
import pandas as pd
import numpy as np
import pickle
import faiss
from langchain_community.embeddings import OllamaEmbeddings
from ast import literal_eval
ollama_emb = OllamaEmbeddings(
    model="llama2",
)

def find_nearest_neighbors(new_embeddings, model_path, k=15):
    # Load the trained FAISS model
    index = faiss.read_index(model_path)
    
    # Assuming new_embeddings is a 2D numpy array where each row is an embedding vector
    # If you have a single embedding vector, reshape it appropriately
    new_embeddings = np.array(new_embeddings).reshape(1, -1) # Example for a single embedding
    
    # Searching the index
    distances, indices = index.search(new_embeddings, k)
    
    return distances, indices[0]

def get_similar_translation_examples(df,query_embeddings,target_lang): 
    model_path = "faiss_model.index"
    distances, indices = find_nearest_neighbors(query_embeddings, model_path, k=15)
    neighbor_data = df.iloc[indices]
    neighbor_data = neighbor_data[neighbor_data['target_lang']==target_lang]
    translation_examples = [(neighbor_data['en'][i],neighbor_data[target_lang][i]) for i in neighbor_data.index]
    return translation_examples

with open('./faiss_data.pkl','rb') as f:
    df = pickle.load(f)

# Define your variables
lang = {
  'pt' : 'Portuguese',
  'fi' : 'Finnish',
  'es' : 'Spanish'
}
target_language = "pt"  # Example target language
# english_text = input('Your English text here: ')
english_text = 'The quick brown fox jumps over the lazy dog.'
userVector = ollama_emb.embed_query(english_text)

# english_text = 'The quick brown fox jumps over the lazy dog.'
# translation_examples = [
#     ("The small bird sings in the tall tree.", "Le petit oiseau chante dans le grand arbre."),
#     ("The fast horse races across the green field.", "Le cheval rapide court à travers le champ vert."),
#     # Add more examples as needed
# ]

translation_examples = get_similar_translation_examples(df, userVector, target_language)

# Construct the translation examples section of the prompt
examples_section = "\n".join([
    f"{indx}. English: {english}\n   {target_language}: {translated}"
    for indx, (english, translated) in enumerate(translation_examples)
])
print("Prompting LLM: ")
# Construct the full prompt
# prompt = f"""I am seeking to translate: {english_text} into {lang[target_language]}:

# To assist in providing a more accurate and nuanced translation, I have included several translation examples that should be considered. These examples demonstrate specific language usage, idiomatic expressions, or complex constructions in {target_language} that are relevant to the text needing translation.

# Translation Examples:
# {examples_section}

# Please utilize these examples to guide the translation process, paying special attention to the nuances and specific usage demonstrated. The goal is to produce a translation that is not only accurate but also reflects the linguistic characteristics and subtleties highlighted in the examples provided.
# Tell me only the translation of the english text I provided initially. There is no need to translate anything else.  
# Respond with only the translation, and nothing else.
# """

prompt = f"""I am seeking to translate the following English text into {lang[target_language]}:

"{english_text}"

The following are not for translation but to assist you in providing a more accurate and nuanced translation of the above text. These examples demonstrate specific language usage, idiomatic expressions, or complex constructions in {target_language} relevant to the text needing translation. They are for your reference only and should not be translated.

Reference Examples:
{examples_section}

Based on the guidance provided by these examples, please translate only the initially provided English text into {lang[target_language]}. Your response should contain only the translation of this text, with no additional text or explanations.
"""

# print(prompt)

response = ollama.chat(model='llama2', messages=[
  {
     'role': 'user',
     'content': prompt,
   },
])

print(response['message']['content'])