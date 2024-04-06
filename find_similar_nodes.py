import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import OllamaEmbeddings
from ast import literal_eval
from neo4j import GraphDatabase

# credentials
URL = "neo4j://localhost:7687" 
USERNAME = "neo4j"  
PASSWORD = "temp@123" 

embeddings_list, similarity_scores = [], []

OLLAMA_EMB = OllamaEmbeddings(
    model="llama2",
)

userSentence = input('Enter user sentence: ')

# userQuestion = "There was nothing else to do, so Alice soon began talking again. 'Dinah'll miss me very much to-night, I should think!'"

# userVector = OLLAMA_EMB.embed_query(userSentence)

def get_nodes_with_properties(tx):
    result = tx.run("MATCH (n) RETURN n")
    return result

###############################################################################################################
# # Main function to run the query and print results
# def main():
#     # Connect to Neo4j
#     driver = GraphDatabase.driver(URI, auth=(username, password))
    
#     with driver.session() as session:
#         # Execute the query
#         nodes = session.run("MATCH (n) RETURN n")
#         # print('hellloooo')
#         # print(nodes)
        
#         # Print nodes and their properties
#         for record in nodes:
#             node = record["n"]
#             # print(type(dict(node)['property1']))
#             embeddings_list.append(literal_eval(dict(node)['property1']))
#             # print("eng emb:", dict(node)['property1'])
#             # print("Properties:", dict(node).keys())

# # similarity_score = cosine_similarity(vector1, vector2)
#     # Close the Neo4j driver
#     driver.close()
###############################################################################################################

def cosine_similarity(vector1, vector2):
    # print(len(vector1), len(vector2))
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

path = 'output.csv'
df = pd.read_csv(path)
for s1 in range(df.shape[0]):
    for s2 in range(s1 + 1, df.shape[0]):
        similarity_scores.append(cosine_similarity(literal_eval(df.iloc[s1,3]), literal_eval(df.iloc[s2, 3])))

    print(s1)

print(similarity_scores)


# with open('output.csv', 'r') as f:
#     f.read
#     for s1 in range(len())

# if __name__ == "__main__":
    




    # for i in range(len(embeddings_list)):
    #     similarity_scores.append(cosine_similarity(embeddings_list[i], userVector))
    


    # similarity_scores.sort(reverse=True)
    # print(similarity_scores)
    