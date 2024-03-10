from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import argparse
from tot.methods.bfs import solve
from tot.tasks.bio_name import Bio_Name
import json
import pickle

SapBERT_tokenizer = AutoTokenizer.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
SapBERT_model = AutoModel.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')

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

# TODO: make relative path
all_go_terms_embeddings_dict = pickle.load(open('src/tot/data/gene_sets/all_go_terms_embeddings_dict.pkl', 'rb'))

def get_similarity_score(pred, label):
    pred_embedding = getSentenceEmbedding(pred, SapBERT_tokenizer, SapBERT_model)
    label_embedding = all_go_terms_embeddings_dict[label]
    similarity_score = cosine_similarity(pred_embedding, label_embedding)[0][0]
    return similarity_score

def get_similarity_percentile(similarity_score, pred):
    pred_embedding = getSentenceEmbedding(pred, SapBERT_tokenizer, SapBERT_model)
    null_dist = []
    for term in all_go_terms_embeddings_dict.keys():
        term_embedding = all_go_terms_embeddings_dict[term]
        sentenceSim = cosine_similarity(pred_embedding, term_embedding)[0][0]
        null_dist.append(sentenceSim)
    # get similarity percentile
    null_dist = np.array(null_dist)
    # return the fraction of scores that are smaller than the candidate
    percentile = (null_dist < similarity_score).mean()
    return percentile

def test_example(args, task, idx):
    label = task.get_label(idx)
    final_answer, ys, steps, trie, sw_prompt = solve(args, task, idx)
    return final_answer, ys, steps, trie, sw_prompt, label

def get_all_candidate_bio_processes(steps):
    candidate_processes = []
    step_count = 0
    for step in steps['steps'][:-1]:
        step_count += 1
        new_ys = [json.loads(step['new_ys'][i]) for i in range(len(step['new_ys']))]
        new_bio_processes = [y['Biological Process'] for y in new_ys]
        candidate_processes.extend(new_bio_processes)
    candidate_processes = list(set(candidate_processes))
    return candidate_processes
    
def get_best_candidate_bio_process(candidate_processes, label):
    scores = [get_similarity_score(candidate_process, label) for candidate_process in candidate_processes]
    best_candidate_process = candidate_processes[np.argmax(scores)]
    return best_candidate_process, np.max(scores)

def test_example_wrap(idx, args, task, verbose=False):
    final_answer, ys, steps, trie, sw_prompt, label = test_example(args, task, idx)
    candidate_processes = get_all_candidate_bio_processes(steps)
    best_candidate_process, best_candidate_similarity_score = get_best_candidate_bio_process(candidate_processes, label)
    final_answer_similarity_score = get_similarity_score(final_answer, label)
    final_answer_similarity_quantile = get_similarity_percentile(final_answer_similarity_score, final_answer)
    best_candidate_similarity_quantile = get_similarity_percentile(best_candidate_similarity_score, best_candidate_process)
    if verbose:
        print('Index:', idx)
        print('Final answer:', final_answer)
        print('True answer:', label.strip())
        print('Final answer similarity score:', get_similarity_score(final_answer, label))
        print('Best candidate process:', best_candidate_process)
        print('Best candidate similarity score:', best_candidate_similarity_score)
        print('Final Answer Similarity Quantile:', final_answer_similarity_quantile)
        print('Best Candidate Similarity Quantile:', best_candidate_similarity_quantile)
        print()
    return {'index': idx, 'final answer': final_answer, 'ys': ys, 'steps': steps, 'label': label, 'final answer similarity score': final_answer_similarity_score,'best candidate process': best_candidate_process, 'best similarity score': best_candidate_similarity_score,
            'final answer similarity quantile': final_answer_similarity_quantile, 'best candidate similarity quantile': best_candidate_similarity_quantile, 'trie': trie, 'sw_prompt': sw_prompt}

def get_cot_prompt(test_gene_set):
    content = f'''
    Question: Give a brief name for the most prominent biological process performed by the following set of genes. Respond in the format "Process: <name>".\nGenes: CAD UCK1 UCK2 CDA UMPS CMPK1 UPP1 UPP2 UCKL1 DHODH
    Let's think step by step:
    Step: 1. Biological Process: Nucleotide biosynthetic process. Reason: The genes listed are involved in the synthesis of nucleotides, such as CAD (involved in pyrimidine biosynthesis), UCK1/UCK2 (kinases in nucleotide synthesis), and DHODH (dihydroorotate dehydrogenase in pyrimidine synthesis).
    Step: 2. Parent: Nucleotide biosynthetic process. Relation: is a. Reason: This is a more specific process because it focuses on the subset of nucleotides that are pyrimidines. The genes CAD, DHODH, and UMPS are relevant as they are directly involved in the synthesis of pyrimidine nucleotides.
    Biological Process: Pyrimidine nucleotide biosynthetic process
    Step: 3. Parent: Pyrimidine nucleotide biosynthetic process. Relation: is a. Reason: This is a more specific process because it focuses on the synthesis of UMP, which is a central precursor for other pyrimidine nucleotides. The gene UMPS is essential for this process as it encodes for the enzyme uridine monophosphate synthetase, which is responsible for the final steps of UMP synthesis. Biological Process: Uridine monophosphate biosynthetic process
    Step: 4. Parent: Uridine monophosphate biosynthetic process. Relation: part of, Reason: This process is more specific as it refers to the larger pathway of which the uridine monophosphate biosynthetic process is a part. The genes CAD, DHODH, and UMPS are all involved in the de novo synthesis of pyrimidine ribonucleotides, which includes the synthesis of UMP. Biological Process: De novo pyrimidine ribonucleotide biosynthesis
    Process: De novo pyrimidine ribonucleotide biosynthesis
    
    
    Question: Give a brief name for the most prominent biological process performed by the following set of genes. Respond in the format "Process: <name>".\nGenes: LOC102724560 MMUT MTHFR DPEP1 MTHFD1 MTRR CBS MPST CPS1 TST BLMH BHMT CTH NOX4
    Let's think step by step:
    Step: 1. Biological Process: Metabolism. Reason: Many of the listed genes are involved in metabolic pathways, such as amino acid metabolism (CTH, CBS), nucleotide metabolism (MTHFR, MTRR), and detoxification processes (MPST, TST, BLMH). Metabolism is a high-level biological process that encompasses the chemical reactions within cells, including catabolism and anabolism.
    Step: 2. Relation: is a. Reason: Amino Acid Metabolism is a more specific category within the broad spectrum of metabolic processes. It involves the breakdown and synthesis of amino acids. Genes such as CTH (cystathionine gamma-lyase), CBS (cystathionine beta-synthase), and BHMT (betaine-homocysteine S-methyltransferase) are specifically involved in the metabolism of sulfur-containing amino acids, indicating the relevance of this more specific biological process. Biological Process: Amino Acid Metabolism
    Step: 3. Relation: is a. Reason: Sulfur Amino Acid Metabolism is a more specific category within Amino Acid Metabolism. It refers to the metabolic processes involving amino acids that contain sulfur, such as methionine and cysteine. Genes such as CTH, CBS, and MPST (3-mercaptopyruvate sulfurtransferase) are directly involved in the metabolism of sulfur-containing amino acids. Biological Process: Sulfur Amino Acid Metabolism
    Step: 4. Relation: is a. Reason: Homocysteine Metabolism is a more specific biological process within Sulfur Amino Acid Metabolism. It focuses on the metabolic pathways involving homocysteine, including its synthesis and catabolism. Genes such as MTHFR (methylenetetrahydrofolate reductase), CBS (cystathionine beta-synthase), and MTRR (methionine synthase reductase) are relevant because they are directly involved in the conversion of homocysteine to methionine or its catabolism to cysteine. Biological Process: Homocysteine Metabolism    
    Step: 5. Relation: is a. Reason: Methionine Synthesis is a more specific biological process within Homocysteine Metabolism. It refers to the conversion of homocysteine into methionine, which is an essential amino acid. Genes like MTHFR, MTRR, and CBS are relevant to this process as they encode enzymes that are directly involved in the biochemical reactions that synthesize methionine from homocysteine. Biological Process: Methionine Synthesis
    Process: Methionine Synthesis
    
     Question: Give a brief name for the most prominent biological process performed by the following set of genes. Respond in the format "Process: <name>".\nGenes: {test_gene_set}
    '''
    messages = []
    messages.append({'role': 'system', 'content': 'You are a helpful and knowledgable assistant to a molecular biologist.'})
    messages.append({'role': 'user', 'content': content})
    return messages

def parse_cot_response(response):
    return response[0].split('Process: ')[-1].strip()

def get_few_shot_prompt(gene_sets, labels, test_gene_set):
    assert len(gene_sets) == len(labels), "gene_sets and labels must have the same length"
    messages = []
    messages.append({'role': 'system', 'content': 'You are a helpful and knowledgable assistant to a molecular biologist.'})
    for i, gene_set in enumerate(gene_sets):
        messages.append({'role': 'user', 'content': 'Give a brief name for the most prominent biological process performed by the following set of genes. Respond in the format "Process: <name>".\nGenes: {}'.format(gene_set)})
        messages.append({'role': 'assistant', 'content': 'Process: {}'.format(labels[i])})
    
    messages.append({'role': 'user', 'content': 'Give a brief name for the most prominent biological process performed by the following set of genes. Respond in the format "Process: <name>".\nGenes: {}'.format(test_gene_set)})
    return messages

def parse_few_shot_response(response):
    return response[0].split(': ')[1].strip()
    
def few_shot(test_idx, eval_data, train_data, model, n=2):
    test_gene_set = eval_data[test_idx][1]
    true_label = eval_data[test_idx][2]
    indices = np.random.choice(len(train_data), n, replace=False)
    genes = [train_data[i][1] for i in indices]
    labels = [train_data[i][2] for i in indices]
    messages = get_few_shot_prompt(genes, labels, test_gene_set)
    response = model(messages)
    response = parse_few_shot_response(response)
    return test_gene_set, response, true_label

def load_few_shot_data():
    eval_x_filename = 'src/tot/data/gene_sets/x_eval.txt'
    eval_y_filename = 'src/tot/data/gene_sets/y_eval.txt'
    x_filename = 'src/tot/data/gene_sets/x.txt'
    y_filename = 'src/tot/data/gene_sets/y.txt'

    with open(x_filename, 'r') as f:
        x = f.readlines()
        x = [i.strip() for i in x]

    with open(y_filename, 'r') as f:
        y = f.readlines()
        y = [i.strip() for i in y]

    with open(eval_x_filename, 'r') as f:
        eval_x = f.readlines()
        eval_x = [i.strip() for i in eval_x]

    with open(eval_y_filename, 'r') as f:
        eval_y = f.readlines()
        eval_y = [i.strip() for i in eval_y]

    data = list(zip(x, y))
    eval_data = list(zip(range(len(eval_x)), eval_x, eval_y))
    train_data = [tup for tup in data if tup[0] not in eval_x]
    train_data = list(zip(range(len(train_data)), [gene_set for gene_set, _ in train_data], [label for _, label in train_data]))
    return train_data, eval_data

def get_zero_shot_prompt(test_gene_set):
    messages = []
    messages.append({'role': 'system', 'content': 'You are a helpful and knowledgable assistant to a molecular biologist.'})
    messages.append({'role': 'user', 'content': 'Give a brief name for the most prominent biological process performed by the following set of genes. Respond in the format "Process: <name>".\nGenes: {}'.format(test_gene_set)})
    return messages

def parse_zero_shot_response(response):
    return response[0].split(': ')[1].strip()

def best_of_6_prompt(test_gene_set):
    messages = []
    messages.append({'role': 'system', 'content': 'You are a helpful and knowledgable assistant to a molecular biologist.'})
    messages.append({'role': 'user', 'content': 'Give 6 unique names for the prominent biological processes performed by the following set of genes. Respond with a semicolon-sepated list in the format "Name1; Name2; Name3; Name4; Name5; Name6".\nGenes: {}'.format(test_gene_set)})
    return messages

def parse_best_of_6_response(response):
    return response[0].split('; ')

def make_expert_prompt(genes, feature_df = [], direct = False, customized_prompt = None):
    """
    Create a ChatGPT prompt based on the list of genes
    :return: A string containing the ChatGPT prompt text
    """

    general_analysis_instructions = """
Be concise, do not use unneccesary words. Be specific, avoid overly general
statements such as 'the proteins are involved in various cellular processes'
Be factual, do not editorialize.
For each important point, describe your reasoning and supporting information. 
    """
    
    task_instructions = """
Write a critical analysis of the biological processes performed 
by this system of interacting proteins. Propose a brief name for
for the most prominant biological process performed by the system.
    """
    direct_instructions = """
Propose a name and provide analysis for the following gene set.
    """
    
    format_placeholder = """ 
Put the name at the top of the analysis as 'Process: <name>' 
    """
    
    if direct == True:
        prompt_text = direct_instructions
        prompt_text += format_placeholder
    elif customized_prompt:
        prompt_text = customized_prompt
        prompt_text += format_placeholder
    else:
        prompt_text = task_instructions
        prompt_text += format_placeholder
        prompt_text += general_analysis_instructions
        
    prompt_text += "\n\nHere are the interacting proteins:\n"
    prompt_text += f'\nProteins: '
    prompt_text += ", ".join(genes) + ".\n\n"

    if feature_df:
        prompt_text += "\n\nHere are the gene features:\n"
        prompt_text  = add_gene_feature_summary(prompt_text, feature_df)
    
    

    return prompt_text