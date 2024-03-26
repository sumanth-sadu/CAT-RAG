from graphdatascience import GraphDataScience
from neo4j import GraphDatabase
from ast import literal_eval
import csv
import pandas as pd
file_path = 'output.csv'

id_list, sent_list, en_vector_list, pt_vector_list = [], [], [], []
with open(file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)
    for row in reader:
        id, node1_name, node2_name, node1_properties, node2_properties = row
        id_list.append(int(id))
        sent_list.append(node1_name + '==>' + node2_name)
        en_vector_list.append(literal_eval(node1_properties))
        pt_vector_list.append(literal_eval(node2_properties))

def create_similar_relationships(driver,df):
    with driver.session() as session:
        for index, row in df.iterrows():
            # Extract node IDs and similarity score
            node1_id = row['node1']
            node2_id = row['node2']
            similarity_score = row['similarity']
            
            # Execute a Cypher query to create the relationship
            cypher_query = """
            MATCH (node1), (node2)
            WHERE ID(node1) = $node1_id AND ID(node2) = $node2_id
            MERGE (node1)-[r:SIMILAR]->(node2)
            SET r.similarity = $similarity_score
            """
            session.run(cypher_query, node1_id=node1_id, node2_id=node2_id, similarity_score=similarity_score)


def get_top_neighbors(driver, node_id):
    with driver.session() as session:
        result = session.run("""
            MATCH (sampleNode)-[r:SIMILAR]->(neighbor)
            WHERE ID(sampleNode) = $nodeId
            RETURN ID(neighbor) AS NeighborID, r.similarity AS Similarity
            ORDER BY Similarity DESC
            LIMIT 5
        """, nodeId=node_id)
        return [record for record in result]


nodes = pd.DataFrame(
    {
        "nodeId": id_list,
        "labels": sent_list,
        "prop1": pt_vector_list,
        "otherProperty": en_vector_list
    }
)

relationships = pd.DataFrame(
    {
        "sourceNodeId": id_list,
        "targetNodeId": id_list,
        "relationshipType": ["REL"] * (len(id_list)),
        "weight": [1.0] * (len(id_list))
    }
)

# relationships = pd.DataFrame(
#     {
#         "sourceNodeId": id_list,
#         "targetNodeId": id_list,
#         "relationshipType": ["REL"]*(len(id_list)),
#         "weight": [1.0]*(len(id_list))
#     }
# )

# Configure the driver with AuraDS-recommended settings
gds = GraphDataScience("neo4j://localhost:7687", auth=("neo4j", "temp@123"))
uri = "neo4j://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "temp@123"))

G = gds.graph.construct(
    "my-graph37",      # Graph name
    nodes,           # One or more dataframes containing node data
    # relationships    # One or more dataframes containing relationship data
)

# stream_result = gds.run_cypher("""
# CALL gds.graph.streamNodeProperties('my-graph29', ['prop1', 'otherProperty'])
# YIELD nodeId
# RETURN nodeId
# LIMIT 10
# """)

# print(stream_result)
# assert "REL" in G.relationship_types()

# Calculate similarities using gds.knn.stream
knn_stream_result = gds.knn.stream(
    G,
    topK=5,
    nodeProperties="otherProperty"
)

knn_df = pd.DataFrame(knn_stream_result)


create_similar_relationships(driver, knn_df)


print("k-NN Stream Results:")
print(knn_df.head(10))  # Print the first few rows to check




# Fetch and print top neighbors for a sample node
top_neighbors = get_top_neighbors(driver, 0)
print("Top Similar Neighbors for Node 0:")
neighbors = knn_df[knn_df['node1']==0]
similar_nodes = neighbors['node2'].tolist()
print(f"Source Sentence: {nodes[['labels']].iloc[0]['labels'].split('==>')[0]})") #
print(nodes[['labels']].iloc[similar_nodes])
# for neighbor in top_neighbors:
#     print(neighbor)