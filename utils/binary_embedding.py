# binary_embedding
# This collection of functions handles processing binary embeddings

# Internal Imports

# External Imports
import pandas as pd
from tqdm import tqdm

# Globals



def load(loc):
    result = {}
    with open(loc) as input_file:
        input_file.readline()
        for line in input_file:
            split = line.split()
            binary = split[1:]
            result[split[0]] = binary
        input_file.close()
    return pd.DataFrame(data=result)



def embed_list(data_list, embedding):
    result = []
    for item in tqdm(data_list, desc="Embedding data"):
        if item:
            result.append(embed_item(item, embedding))
        else:
            result.append(None)
    return pd.Series(result)



def embed_item(data_item, embedding):
    result = []
    for word in data_item.split():
        result.append(int(embedding[word].values[0]) if word in embedding.keys() else 0)
    return result



def get_closest(integers, embedding):
    pass