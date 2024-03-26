import pandas
from langchain_community.embeddings import OllamaEmbeddings
from neo4j import GraphDatabase
import csv
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

# Neo4j connection details
uri = "neo4j://localhost:7687"
username = "neo4j"
password = "temp@123"

# CSV file path
file_path = "output.csv"

def create_graph(tx, node1_name, node1Properties, node2_name, node2Properties):
    lang = "en-pt"
    query = """
    MERGE (node1:Node {languages: $lang ,name: $node1_name, property1: $node1_property1, translation: $node2_name, property2: $node2_property2})
    """

# MERGE (node2:Node {name: $node2_name, property2: $node2_property2})
#     MERGE (node1)-[:TRANSLATES_TO]->(node2)

    tx.run(query, lang = lang, node1_name=node1_name, node1_property1=node1Properties['property1'], node2_name=node2_name, node2_property2=node2Properties['property2'])

# with GraphDatabase.driver(uri, auth=(username, password)) as driver:
#     with open(file_path, newline='') as csvfile:
#         reader = csv.reader(csvfile)
#         header = next(reader)
#         for row in reader:
#             id, node1_name, node2_name, node1_properties, node2_properties = row
#             with driver.session() as session:
#                 session.write_transaction(create_graph, node1_name, {"property1": node1_properties}, node2_name, {"property2": node2_properties})


"""
CALL gds.graph.create(
  'myGraph',
  'node',
  '*',    
  {
    nodeProperties: ['property1'] 
  }
)
"""


# // Parameters
# WITH "User's English Question" AS userQuestion
userQuestion = "There was nothing else to do, so Alice soon began talking again. 'Dinah'll miss me very much to-night, I should think!'"

ollama_emb = OllamaEmbeddings(
    model="llama2",
)

userVector = ollama_emb.embed_query(userQuestion)











# general_system_template = """ 

prompt = """
Here are your instructions: 
Objective: Assist users in converting their input of broken English words or phrases into grammatically correct, coherent English sentences.

Instructions for the Language Model:

    Interpret the Input: Carefully analyze the user's input, which may contain broken English, incorrect grammar, or disordered words. Try to understand the intended meaning based on the context provided by the input.

    Correct Grammar and Spelling: Identify and correct any grammatical errors, spelling mistakes, or incorrect usage of words. Ensure that the corrected sentence maintains the original intended meaning.

    Rearrange Words: If the words are in a disordered sequence, rearrange them to form a coherent sentence that flows logically and naturally.

    Provide Suggestions: If the intended meaning is unclear or if multiple interpretations are possible, you may provide more than one corrected version of the sentence. Briefly explain the context or meaning assumed for each version, if necessary.

    Maintain Politeness and Inclusivity: Ensure that the revised sentences are polite, inclusive, and considerate of diverse backgrounds and perspectives.

    Request Clarification When Needed: If the input is too vague or lacks sufficient context to form a coherent sentence, politely ask the user for more information or clarification.

Example Usage:

    User Input: "tomorrow meeting important very is"

    Language Model Response: "The meeting tomorrow is very important."

    User Input: "we late can't afford be to"

    Language Model Response: "We can't afford to be late."

This template is designed to guide the language model in assisting users with varying levels of proficiency in English, helping them express their thoughts more clearly and accurately. Remember, the key is to ensure that the user's intended meaning is preserved while improving the grammatical structure of their sentences.
"""























    general_user_template = "Question:```{question}```"
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template),
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=qa_prompt,
    )

    # Vector + Knowledge Graph response
    kg = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=embeddings_store_url,
        username=username,
        password=password,
        database="neo4j",  # neo4j by default
        index_name="stackoverflow",  # vector by default
        text_node_property="body",  # text by default
        retrieval_query="""
    WITH node AS question, score AS similarity
    CALL  { with question
        MATCH (question)<-[:ANSWERS]-(answer)
        WITH answer
        ORDER BY answer.is_accepted DESC, answer.score DESC
        WITH collect(answer)[..2] as answers
        RETURN reduce(str='', answer IN answers | str + 
                '\n### Answer (Accepted: '+ answer.is_accepted +
                ' Score: ' + answer.score+ '): '+  answer.body + '\n') as answerTexts
    } 
    RETURN '##Question: ' + question.title + '\n' + question.body + '\n' 
        + answerTexts AS text, similarity as score, {source: question.link} AS metadata
    ORDER BY similarity ASC // so that best answers are the last
    """,
    )

    kg_qa = RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=kg.as_retriever(search_kwargs={"k": 2}),
        reduce_k_below_max_tokens=False,
        max_tokens_limit=3375,
    )
    return kg_qa






# # // Convert user's question into vector
# WITH userQuestion, vector(userQuestion) AS userVector

# # // Find similar nodes (translations) based on vector similarity
# MATCH (englishNode:English) node.language:english
# WHERE gds.alpha.similarity.cosine(englishNode.embedding, userVector) > 0.8
# WITH englishNode, userQuestion

# // Retrieve corresponding Portuguese translations
# MATCH (englishNode)-[:TRANSLATES_TO]->(portugueseNode:Portuguese)
# RETURN englishNode.question AS englishQuestion, portugueseNode.translation AS portugueseTranslation
# ORDER BY gds.alpha.similarity.cosine(englishNode.embedding, userVector) DESC
# LIMIT 5;
