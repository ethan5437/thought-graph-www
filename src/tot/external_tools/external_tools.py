import pandas as pd
import gseapy as gseapy
from gprofiler import GProfiler
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

SapBERT_tokenizer = AutoTokenizer.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
SapBERT_model = AutoModel.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')

gp = GProfiler(return_dataframe=True)
dbs = gseapy.get_library_name()

#for SapBERT
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def getSentenceEmbedding(sentence, tokenizer, model):
    # Tokenize sentences
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        
    # Perform pooling. In this case, mean pooling.
    sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    
    return sentence_embedding

def getSentenceSimilarity(sentence1, sentence2, tokenizer, model, simMetric):
    sentence1_embedding = getSentenceEmbedding(sentence1, tokenizer, model)
    sentence2_embedding = getSentenceEmbedding(sentence2, tokenizer, model)

#     print('external_tools--sentence2_embedding', sentence2_embedding.shape)
    if simMetric == "cosine_similarity":
        sentenceSim = cosine_similarity(sentence1_embedding, sentence2_embedding).flatten()
    # ToDo: add other simMetrics
    #elif simMetric == "cosine_similarity_primitive": # use primitive operations
   #     sentenceSim = np.dot(sentence1_embedding, sentence2_embedding)/(norm(sentence1_embedding)*norm(sentence2_embedding))
    
    return sentenceSim, sentence1_embedding, sentence2_embedding

def similarity_score(x, y):
    return getSentenceSimilarity(x, y, SapBERT_tokenizer, SapBERT_model, "cosine_similarity")[0]

def get_GO_Enrich(systemGenes):
    systemGenes = systemGenes.split(', ')
    enr = gseapy.enrichr(gene_list=systemGenes, # or "./tests/data/gene_list.txt",
                 gene_sets=['GO_Biological_Process_2023'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                )
    top_terms = []
    for i in range(3):
        top_terms.append(enr.results.iloc[i]['Term'])
    return top_terms

def filter_result(proposed_term, enriched_terms, filter_method='sim', filter_size=10):
    if filter_method == 'sim':
        similiarity = similarity_score(proposed_term, enriched_terms)
        #get index
        top_indices = np.argsort(similiarity)[::-1][:filter_size]
    else:
        raise ValueError('another filter_method has not been implemented yet')
        
    return top_indices
    
    
def get_similar_term_GProfiler(protein_list, proposed_terms, source='GO:BP', filter_method='sim', filter_size=10):
    '''
    protein_list: gene set
    propsoed_terms: the GO terms proposed by model
    source: the type of data
    filter_method: method to filter the result of GProfiler
    sample size: how much to filter
    '''
    gp_results = gp.profile(organism='hsapiens', query=protein_list)
    enriched_terms = gp_results[gp_results['source']==source].name.tolist()
    top_indices = []
    for proposed_term in proposed_terms:
        top_indices_tmp = filter_result(proposed_term, enriched_terms, filter_method, filter_size)
        top_indices_tmp = top_indices_tmp.tolist()
        top_indices += top_indices_tmp
    top_indices = list(set(top_indices))
    filtered_terms = gp_results.iloc[top_indices]
    return filtered_terms
    
    
    