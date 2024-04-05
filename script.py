import csv
from langchain_community.embeddings import OllamaEmbeddings
from ast import literal_eval

def read_csv(input_file):
    with open(input_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)
        data = [row for row in reader]

    return header, data

def write_csv(output_file, header, data):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

# Example Usage
input_csv_file = 'en-pt_truncated.csv'
output_csv_file = 'output.csv'

# Reading from input CSV
input_header, input_data = read_csv(input_csv_file)

embeddings = []

# Do some processing on the data if needed

ollama_emb = OllamaEmbeddings(
    model="llama2",
)

for indx, row in enumerate(input_data):
    print(indx)
    embeddings.append([ollama_emb.embed_query(row[1]), ollama_emb.embed_query(row[2])])

input_header.extend([str(input_header[1]) + '_embeddings', str(input_header[2]) + '_embeddings'])

for indx, row2 in enumerate(input_data):
    print('round 2', indx)
    # print(row2[indx])
    row2.extend(embeddings[indx])

write_csv(output_csv_file, input_header, input_data)
