import csv
import pandas as pd
import numpy as np
from langchain_community.embeddings import OllamaEmbeddings
from ast import literal_eval
import ollama
from neo4j import GraphDatabase

EMBEDDINGS, SIMILARITY_SCORES = [], []
INPUT_CSV_FILE = 'en-fi.csv'
# OUTPUT_CSV_FILE = 'output.csv'
OUTPUT_CSV_FILE = 'en_fi_output.csv'

# Neo4j connection details
URI = "neo4j://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "temp@123"

OLLAMA_EMB = OllamaEmbeddings(
    model="llama2",
)

def read_csv(input_file):
    with open(input_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
        data = [row for row in reader]

    return header, data

input_header, input_data = read_csv(INPUT_CSV_FILE)

def write_csv(output_file, header, data):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

for indx, row in enumerate(input_data):
    EMBEDDINGS.append([OLLAMA_EMB.embed_query(row[1]), OLLAMA_EMB.embed_query(row[2])])

input_header.extend([str(input_header[1]) + '_embeddings', str(input_header[2]) + '_embeddings'])

for indx, row2 in enumerate(input_data):
    row2.extend(EMBEDDINGS[indx])

write_csv(OUTPUT_CSV_FILE, input_header, input_data)


# def create_graph(tx, node1_name, node1Properties, node2_name, node2Properties):
#     lang = "en-pt"
#     query = """
#     MERGE (node1:Node {languages: $lang ,name: $node1_name, property1: $node1_property1, translation: $node2_name, property2: $node2_property2})
#     """

#     tx.run(query, lang = lang, node1_name=node1_name, node1_property1=node1Properties['property1'], node2_name=node2_name, node2_property2=node2Properties['property2'])

# with GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD)) as driver:
#     with open(OUTPUT_CSV_FILE, newline='') as csvfile:
#         reader = csv.reader(csvfile)
#         header = next(reader)
#         for row in reader:
#             id, node1_name, node2_name, node1_properties, node2_properties = row
#             with driver.session() as session:
#                 session.write_transaction(create_graph, node1_name, {"property1": node1_properties}, node2_name, {"property2": node2_properties})

# userSentence = input('Enter user sentence: ')

def get_nodes_with_properties(tx):
    result = tx.run("MATCH (n) RETURN n")
    return result

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

# df = pd.read_csv(OUTPUT_CSV_FILE)
# for s1 in range(df.shape[0]):
#     for s2 in range(s1 + 1, df.shape[0]):
#         SIMILARITY_SCORES.append(cosine_similarity(literal_eval(df.iloc[s1,3]), literal_eval(df.iloc[s2, 3])))

#     print(s1)

# print(SIMILARITY_SCORES)


# Define your variables
# target_language = "French"  # Example target language
# english_text = input('Your English text here: ')
# translation_examples = [
#     ("English sentence 1", "Translated sentence 1 in target language"),
#     ("English sentence 2", "Translated sentence 2 in target language"),
#     # Add more examples as needed
# ]

# # Construct the translation examples section of the prompt
# examples_section = "\n".join([
#     f"1. English: {english}\n   {target_language}: {translated}"
#     for english, translated in translation_examples
# ])

# # Construct the full prompt
# prompt = f"""I am seeking to translate the following English text into {target_language}:

# {english_text}

# To assist in providing a more accurate and nuanced translation, I have included several translation examples that should be considered. These examples demonstrate specific language usage, idiomatic expressions, or complex constructions in {target_language} that are relevant to the text needing translation.

# Translation Examples:
# {examples_section}

# Please utilize these examples to guide the translation process, paying special attention to the nuances and specific usage demonstrated. The goal is to produce a translation that is not only accurate but also reflects the linguistic characteristics and subtleties highlighted in the examples provided.

# """

# print(prompt)

# response = ollama.chat(model='llama2', messages=[
#   {
#      'role': 'user',
#      'content': prompt,
#    },
# ])

# print(response['message']['content'])