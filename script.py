import csv
from langchain_community.embeddings import OllamaEmbeddings

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

for i in range(len(input_data)):
    print('round 1',i)
    embeddings.append([ollama_emb.embed_query(input_data[1]), ollama_emb.embed_query(input_data[2])])

input_header.extend([str(input_header[1]) + '_embeddings', str(input_header[2]) + '_embeddings'])

for j in range(len(input_data)):
    print('round 2', j)
    input_data[j].extend(embeddings[j])

write_csv(output_csv_file, input_header, input_data)
