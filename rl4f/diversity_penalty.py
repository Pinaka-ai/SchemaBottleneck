from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def get_embedding(words):
    # inputs = tokenizer(word, return_tensors='pt')
    # outputs = model(**inputs)
    # # Get the embeddings for the [CLS] token (usually the first token)
    # cls_embedding = outputs.last_hidden_state[0][0]
    # return cls_embedding.detach().numpy()
    return model.encode(words)


def calculate_pairwise_similarity(words):
    embeddings = get_embedding(words)
    # print('embedding shape', embeddings.shape)
    # print('embeddings', embeddings)
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

def compute_diversity_penalty(words):
    similarity_matrix = calculate_pairwise_similarity(words)

    # Calculate the upper triangle of the similarity matrix without the diagonal
    # upper_triangle_indices = np.triu_indices(len(similarity_matrix), k=1)
    indices_and_values = []
    # for indices in upper_triangle_indices:
    #     indices_and_values.append((similarity_matrix[indices], indices))

    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            indices_and_values.append((similarity_matrix[i, j], i, j))

    # print('indices_and_values', indices_and_values)
    indices_and_values.sort(reverse=True)

    for idx, (sim, i, j) in enumerate(indices_and_values):
        print(idx, ', ', sim, ', ', words[i], ', ', words[j])
    
    
    # Average similarity as the diversity penalty
    # diversity_penalty = np.mean(upper_triangle_values)
    # return diversity_penalty

cleaned_aspects = json.load(open('cleaned_aspects.json', "r"))
np.random.seed(0)
aspects_array = np.random.choice(cleaned_aspects, 100, replace=False)

# print(compute_diversity_penalty(['harm', 'love', 'affection']))
print(compute_diversity_penalty(aspects_array))
