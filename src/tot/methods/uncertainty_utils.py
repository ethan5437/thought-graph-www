import json
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
from collections import Counter, defaultdict
from typing import List
    
import numpy as np
import json

SapBERT_tokenizer = AutoTokenizer.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
SapBERT_model = AutoModel.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')

def group_and_frequency_analyze_by_similarity(ys_list, top_n=3, exploration_rate=0.1, threshold=0.9):
    # Flatten the list to process all items, including duplicates
    ys = [item for items in ys_list for item in items]
    all_items_flat = [json.loads(item)['Biological Process'] for items in ys_list for item in items]

    # Get embeddings for all items, considering duplicates
    all_items_embeddings = getSentenceEmbedding(all_items_flat, SapBERT_tokenizer, SapBERT_model)
    # Calculate cosine similarity between all items
    similarity_matrix = cosine_similarity(all_items_embeddings)

    # Group items based on similarity, considering threshold
    groups = []
    for i in range(len(all_items_flat)):
        grouped = False
        for group in groups:
            if any(similarity_matrix[i][j] > threshold for j in group):
                group.append(i)
                grouped = True
                break
        if not grouped:
            groups.append([i])

    item_counts = sum([len(group) for group in groups])
    # Calculate group frequencies
    group_frequencies = {i: len(group) / item_counts for i, group in enumerate(groups)}

    # Adjust frequencies with exploration factor and store adjustments
    adjustments = {i: np.random.rand() * exploration_rate for i in group_frequencies.keys()}
    adjusted_group_frequencies = {i: freq + adjustments[i] for i, freq in group_frequencies.items()}
    print(f'group_frequencies: {group_frequencies}')
    print(f'adjusted_group_frequencies: {adjusted_group_frequencies}')
    # Sort groups by adjusted frequency
    sorted_group_freq = sorted(adjusted_group_frequencies.items(), key=lambda x: x[1], reverse=True)

    # Select top groups based on adjusted frequencies
    top_groups = sorted_group_freq[:top_n]

    new_ys = []
    frequencies = []
    adjusted_frequencies = []
    for index, _ in top_groups:
        ys_index = groups[index][0]
        new_ys.append(ys[ys_index])
        frequencies.append(group_frequencies[index])  # Original frequency
        adjusted_frequencies.append(adjusted_group_frequencies[index])  # Adjusted frequency
        print(f"Group {index + 1}: {', '.join(all_items_flat[i] for i in groups[index])}")

    grouped_items_by_index = []
    for index, _ in sorted_group_freq:
        grouped_items_by_index.append([
            {"index": index},
            {"grouped_names": ', '.join(all_items_flat[i] for i in groups[index])},
            {"original_frequency": group_frequencies[index]},
            {"adjusted_frequency": adjusted_group_frequencies[index]}
        ])

    return new_ys, frequencies, adjusted_frequencies, grouped_items_by_index

def explore_step_threshold_random_sample(ys_list: List[List[str]], exploration_rate: float=0.1) -> List:
    '''
    ys_list should be an array of num_samples x num_bio_processes (in our case, 3)
    We want do sampling over each column of ys_list, i.e. for each bio_process index
    For each bio process index, with probability exploration_rate, choose a random term from the column;
    with probability 1-exploration_rate, choose the term with the highest frequency
    '''

    num_samples = len(ys_list)
    num_bio_processes = len(ys_list[0])
    results = np.zeros(num_bio_processes)
    for i in range(num_bio_processes):
        bio_process_terms = [ys_list[j][i] for j in range(num_samples)]
        if np.random.rand() < exploration_rate:
            results[i] = np.random.choice(bio_process_terms)
        else:
            results[i] = Counter(bio_process_terms).most_common(1)[0][0]
    return results

def explore_step_temperature(ys_list: List[List[str]], temperature: float=0.7, eps=1e-3) -> List:
    '''
    Performs temperature sampling index-wise over each biological process index
    In other words, performs random sampling over frequency distribution, but after applying a temperature transformation to encourage more or less exploration
    Temperature = 0 corresponds to argmax, temperature < 1 discourages exploration relative to random sampling, temperature > 1 encourages exploration relative to random sampling
    '''

    num_samples = len(ys_list)
    bp1_candidates = [ys_list[j][0] for j in range(num_samples)]
    bp2_candidates = [ys_list[j][1] for j in range(num_samples)]
    bp3_candidates = [ys_list[j][2] for j in range(num_samples)]

    def temperature_transform_probs(probs, temperature):
        # takes prob
        # if temperature = 0, assign equal probability to the argmax terms
        if temperature == 0:
            max_indices = np.where(probs == np.max(probs))[0]
            probs = np.zeros(len(probs))
            probs[max_indices] = 1 / len(max_indices)
            return probs

        logits = np.log(probs) / (temperature + eps)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return probs

    bp1_counts = Counter(bp1_candidates)
    bp1_terms = list(bp1_counts.keys())
    bp1_probs = np.array([bp1_counts[bp1_terms[j]] for j in range(len(bp1_terms))]) / num_samples
    bp1_probs = temperature_transform_probs(bp1_probs, temperature)
    bp1_sample = np.random.choice(bp1_terms, p=bp1_probs)

    bp2_counts = Counter(bp2_candidates)
    bp2_terms = list(bp2_counts.keys())
    bp2_probs = np.array([bp2_counts[bp2_terms[j]] for j in range(len(bp2_terms))]) / num_samples
    bp2_probs = temperature_transform_probs(bp2_probs, temperature)
    bp2_sample = np.random.choice(bp2_terms, p=bp2_probs)

    bp3_counts = Counter(bp3_candidates)
    bp3_terms = list(bp3_counts.keys())
    bp3_probs = np.array([bp3_counts[bp3_terms[j]] for j in range(len(bp3_terms))]) / num_samples
    bp3_probs = temperature_transform_probs(bp3_probs, temperature)
    bp3_sample = np.random.choice(bp3_terms, p=bp3_probs)

    return [bp1_sample, bp2_sample, bp3_sample]



def getSentenceEmbedding(sentence, tokenizer, model):
    # Tokenize sentences
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        
    # Perform pooling. In this case, mean pooling.
    sentence_embedding = model_output.last_hidden_state.mean(dim=1)#mean_pooling(model_output, encoded_input['attention_mask'])
    
    return sentence_embedding


#depreciated
    
# def group_and_frequency_analyze_by_similarity(ys_list, top_n=3, e):
    
#     # Flatten the list to process all items, including duplicates 
#     ys = [item for items in ys_list for item in items]
#     all_items_flat = [json.loads(item)['Biological Process'] for items in ys_list for item in items]


#     # Get embeddings for all items, considering duplicates 
#     all_items_embeddings = getSentenceEmbedding(all_items_flat, SapBERT_tokenizer, SapBERT_model)
#     # Calculate cosine similarity between all items
#     similarity_matrix = cosine_similarity(all_items_embeddings)


#     # Group items based on similarity, considering threshold
#     threshold = 0.90
#     groups = []
#     for i in range(len(all_items_flat)):
#         grouped = False
#         for group in groups:
#             if any(similarity_matrix[i][j] > threshold for j in group):
#                 group.append(i)
#                 grouped = True
#                 break
#         if not grouped:
#             groups.append([i])

#     item_counts = sum([len(group) for group in groups])
#     # Calculate group frequencies
#     group_frequencies = {i: len(group)/item_counts for i, group in enumerate(groups)}

#     # Sort groups by frequency
#     sorted_group_freq = sorted(group_frequencies.items(), key=lambda x: x[1], reverse=True)

#     # Select top 3 groups
#     top_3_groups = sorted_group_freq[:top_n]

#     new_ys = []
#     frequencies = []
#     # Display top 3 grouped names and frequencies
# #     print("1) Top three grouped names based on frequency:")
# #     print("\n2) Top three frequencies:")
#     for index, freq in top_3_groups:
#         ys_index = groups[index][0]
#         new_ys.append(ys[ys_index])
#         frequencies.append(freq)
#         print(f"Group {index + 1}: {', '.join(all_items_flat[i] for i in groups[index])}")
# #         print(freq)        

#     grouped_items_by_index = []
#     # Display all grouped names and their frequencies
# #     print("\n3) All grouped names and their frequency:")
#     for index, freq in sorted_group_freq:
#         grouped_items_by_index.append([{index}, {', '.join(all_items_flat[i] for i in groups[index])}, {freq}])


#     return new_ys, frequencies, grouped_items_by_index

if __name__ == '__main__':
    ys = [
            [1, 2, 3],
            [1, 2, 3],
            [3, 2, 1],
            [3, 1, 2],
            [2, 1, 3],
          ]
    
    for i in range(5):
        print(explore_step_temperature(ys, 0))
    print()
    for i in range(5):
        print(explore_step_temperature(ys, 0.7))
    print()
    for i in range(5):
        print(explore_step_temperature(ys, 1))
    print()
    for i in range(5):
        print(explore_step_temperature(ys, 1.5))
    print()
    for i in range(5):
        print(explore_step_temperature(ys, 10))
