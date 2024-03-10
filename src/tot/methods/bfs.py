import itertools
import numpy as np
from functools import partial
from tot.models import gpt
import json
from graphviz import Digraph
from MedAgents.data_utils import MyDataset
from MedAgents.api_utils import api_handler
from string import punctuation
import argparse
import tqdm
import json
from MedAgents.utils import *
import time
from tot.methods.uncertainty_utils import *

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, ys)
    return values

def get_votes_for_bionames(task, x, ys, n_evaluate_sample, step, voter_type, args):
    attempts = 0
    while attempts < 3:
        try:
            system_message, user_message  = task.vote_prompt_wrap(x, ys)
            vote_outputs = gpt(system_message, user_message, n=n_evaluate_sample, stop=None)        
            values = task.vote_outputs_unwrap(vote_outputs, ys)
            break
        except Exception as e:
            print(f"Attempt {attempts + 1} failed with error: {str(e)}")
            attempts += 1
            if attempts >= 3:
                print("Maximum retry attempts reached. Exiting.")
                return None
    return values, vote_outputs

def get_stop_metrics_for_bionames(task, x, ys, n_evaluate_sample, args):
    attempts = 0
    while attempts < 3:
        try:
            system_message, user_message  = task.stop_prompt_wrap(x, ys)
#             print('user_message',user_message)
            stop_outputs = gpt(system_message, user_message, n=n_evaluate_sample, stop=None) 
#             print('get_stop_metrics_for_bionames -- stop_outputs', stop_outputs)
            stop_metrics = task.stop_outputs_unwrap(stop_outputs, ys)
            print('get_stop_metrics_for_bionames -- stop_metrics', stop_metrics)
            stop_metrics = [1 if x >= (n_evaluate_sample/2) else 0 for x in stop_metrics] #more than half = 1, else = 0
            break
        except Exception as e:
            print(f"Attempt {attempts + 1} failed with error: {str(e)}")
            attempts += 1
            if attempts >= 3:
                print("Maximum retry attempts reached. Exiting.")
                return None
#     error
    return stop_metrics, stop_outputs

def get_multivotes_for_bionames(task, x, ys, n_evaluate_sample, args):
    attempts = 0
    while attempts < 3:
        try:
            system_message1, user_message1  = task.vote_w_tool_prompt_wrap(x, ys, args)
            vote_outputs1 = gpt(system_message1, user_message1, n=n_evaluate_sample, stop=None)        
            values1 = task.vote_outputs_unwrap(vote_outputs1, ys)
            break
        except Exception as e:
            print(f"Attempt {attempts + 1} failed with error: {str(e)}")
            attempts += 1
            if attempts >= 3:
                print("Maximum retry attempts reached. Exiting.")
                return None
            
    attempts = 0   
    while attempts < 3:
        try:
            system_message2, user_message2  = task.vote_prompt_wrap(x, ys)
            vote_outputs2 = gpt(system_message2, user_message2, n=n_evaluate_sample, stop=None)        
            values2 = task.vote_outputs_unwrap(vote_outputs2, ys)
            break
        except Exception as e:
            print(f"Attempt {attempts + 1} failed with error: {str(e)}")
            attempts += 1
            if attempts >= 3:
                print("Maximum retry attempts reached. Exiting.")
                return None        

    values = values1 + values2
    vote_outputs = vote_outputs1 + vote_outputs2
#     print('bfs--get_multivotes_for_bionames--values', values)
    return values, vote_outputs

def run_medagents(task, x, ys, label, args, tools):
    handler = api_handler(args.model_name)
    raw_sample = task.to_MedQAformat(x, ys, label)
    question = raw_sample['question'] if raw_sample['question'][-1] in punctuation else raw_sample['question'] + '?'
    options = raw_sample['options'] 
    gold_answer = raw_sample['answer_idx']
    #0, 0 -->qid, realqid --> redundant parameter-- to remove
    data_info = fully_decode(question, options, gold_answer, handler, tools, args) 
    values = list([0]*len(ys))
    for v in data_info['pred_answer']:
        values[v] = 1     
    return values, data_info
    
def get_medagents_votes(task, x, ys, label, args, use_tool=False):
    if use_tool:
        tool_analyses = []
        tool_anal = task.get_tool_analyses(x, ys, dict, args)
        tool_analyses.append(tool_anal)
    else:
        tool_analyses = None
    values, data_info = run_medagents(task, x, ys, label, args, tool_analyses)
    return values, data_info

def combine_vote_to_answer(task, vote_outputs, ys):
    new_ys = task.combine_vote_to_answer(vote_outputs[0], ys)
    return new_ys

def get_proposals(task, x, y): 
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')
    return [y + _ + '\n' for _ in proposals]

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=1, stop=stop)
    return [y + _ for _ in samples]

def get_samples_for_bionames(task, x, y, n_generate_sample, prompt_sample, step, use_uncertainty=False, exploration_rate=0.0, threshold=0.9):
    system_message, user_message = task.propose_prompt_wrap(x, y, step)
    if n_generate_sample == 1:
        samples = gpt(system_message, user_message, n=n_generate_sample)
        samples = task.into_choices(samples[0], y, step)
        return samples
    
    elif n_generate_sample >= 2:
        attempts = 0
        while attempts < 5:
            try:
                samples = gpt(system_message, user_message, n=n_generate_sample)
                
                processed_samples = []
                for sample in samples:
                    processed_sample = task.into_choices(sample, y, step)
                    processed_samples.append(processed_sample)
                # Assuming group_and_frequency_analyze_by_similarity is defined and ready to be used
                # and it handles the entire process including retries within itself as discussed.
                new_ys, frequencies, adjusted_frequencies, grouped_items_by_index =\
                    group_and_frequency_analyze_by_similarity(processed_samples, top_n=3, \
                                                              exploration_rate=exploration_rate, threshold=threshold)
                
                # If the function execution is successful, break out of the loop
                break
            except Exception as e:
                print(f"Attempt {attempts + 1} failed with error: {str(e)}")
                attempts += 1
                if attempts >= 5:
                    print("Maximum retry attempts reached. Exiting.")
                    print('sample',sample)
                    return None  # Or handle as appropriate for your application
        # This section will only execute if the try block is successful before attempts run out
        return new_ys, frequencies, adjusted_frequencies, grouped_items_by_index

def get_tool_reflection(task, x, ys): 
    propose_prompt = task.propose_prompt_tools(x,ys)
    tools_output = gpt(propose_prompt, n=1)
    ys = task.combine_tools_to_answer(tools_output[0], ys)
    return ys

def get_final_answer_for_bionames(task, x, ys, n_generate_sample, prompt_sample, use_uncertainty=False, threshold=0.9): 
    system_message, user_message = task.propose_prompt_final_wrap(x, ys)
    if n_generate_sample == 1:
        samples = gpt(system_message, user_message, n=1)
        final_answer = task.process_final_answers(samples[0])
        return final_answer, samples
    
    elif n_generate_sample >= 2:
        samples = gpt(system_message, user_message, n=n_generate_sample)

        proceesed_samples = []
        for sample in samples:
            final_answer = task.process_final_answers(sample)
            proceesed_samples.append([sample])
            
        new_ys, frequencies, adjusted_frequencies, grouped_items_by_index = group_and_frequency_analyze_by_similarity(proceesed_samples, top_n=1, exploration_rate=0, threshold=threshold) 
        print('new_ys',new_ys)
        final_answer =  task.process_final_answers(new_ys[0])
        print('final_answer',final_answer)
        if use_uncertainty:
            return final_answer, samples, frequencies, adjusted_frequencies, grouped_items_by_index
        else:
            return final_answer, samples
        

def get_final_answer_for_bionames_sw(task, x, ys, n_generate_sample, prompt_sample): 
    system_message, user_message = task.propose_prompt_sw_final_wrap(x, ys)
    samples = gpt(system_message, user_message, n=1)
    final_answer, samples = task.process_final_answers(samples[0])
    return final_answer, samples

def get_criticism_for_bionames(task, x, omit_y_path):
    # TODO: fix
    system_message, user_message = task.prompt_criticism_wrap(x, omit_y_path)
    sample = gpt(system_message, user_message, n=1)
    sample = task.process_criticism(sample[0])
    sample = {'Genes': x, 'Path': omit_y_path, 'Criticism': sample}
    return sample

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.value = None
        self.relations = {}  # Separate dictionary for relations

    def insert(self, word, value, relations):
        node = self
        for i, char in enumerate(word):
            if char not in node.children:
                node.children[char] = TrieNode()
            if i < len(relations):  # Add relations between current and next character
                node.relations[char] = relations[i]
            node = node.children[char]
        node.is_end = True
        node.value = value

    def visualize(self, graph, parent_name, char, end, relation=""):
        # Create a unique name for each node based on its character and parent
        node_label = f'{char} ({self.value})' if self.value else char
        node_name = f"{parent_name}{char}_{id(self)}"
        if end:
            graph.node(node_name, node_label, shape='circle')
        else:
            graph.node(node_name, node_label)
        
        # Connect this node to its parent in the graph
        if parent_name:
            graph.edge(parent_name, node_name, label=relation)
        
        # Recursively visualize children nodes with the relation for each edge
        for child_char, child_node in self.children.items():
            child_relation = self.relations.get(child_char, "")  # Get the relation for the current edge
            child_node.visualize(graph, node_name, child_char, child_node.is_end, child_relation)

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word, value, relations):
        self.root.insert(word, value, relations)
    
    def search(self, word):
        return self.root.search(word)
    
    def visualize(self):
        graph = Digraph(comment='Trie')
        if self.root:
            self.root.visualize(graph, '', 'Biological Process', False)
        return graph

def process_uncertainty_output(system_messages, user_messages, pertub_user_messages, output_list, x, layer_info):
    messages = []
    for s,u in zip(system_message, user_messages):
        messages.append(s+u)
    
    uncertainty_output = {
        "orig_instruction": user_messages[0],
        "gt_clarification": pertub_user_messages,
        "input": f"{x}",
        "target": output_list,
        "isambig": True,
        "layer": layer_info,
    }
    return uncertainty_output

def solve(args, task, idx, to_print=True):
    global gpt    
    
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    args.model_name = args.backend
    
    x = task.get_input(idx)  # input
    label = task.get_label(idx)
    max_mem_size = 3
    trie = Trie()
    
    exploration_rate = getattr(args, 'exploration_rate', 0)
    stop_expansion = getattr(args, 'stop_expansion', False)
    threshold = getattr(args, 'threshold', 0.9)
    print('stop_expansion',stop_expansion)
    print('threshold',threshold)
    
    mem = []  # cache for self-reflection
    y_paths = [[]]  # current output candidate paths
    final_ys = []
    relations = [[]]
    ys = ['']  # current output candidates
    infos = []
    for step in range(task.steps):
        print(f'running {step+1}/{task.steps} step')
        start = time.time()
        # print('-- step', step, '--')
        # update memory to keep under max_mem_size
        mem = mem[-max_mem_size:]

        new_y_paths = []
        frequencies = []
        adjusted_frequencies = []
        sorted_grouped_items = []
        
        if step > 0:
            new_relations = []
        # generation
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
            
        elif args.method_generate == 'sample_bionames':
            assert args.n_generate_sample == 1, 'args.n_generate_sample != 1'
            new_ys = []
            for i, y in enumerate(ys):
                item = get_samples_for_bionames(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, step=step)
                for y in item:
                    new_ys.append(y)
                    new_y_paths.append(y_paths[i] + [y])
                    if step > 0:
                        new_relations.append(relations[i] + [json.loads(y)['Relation']])
                        
        elif args.method_generate == 'sample_bioname_uncertainty':
            new_ys = []
            for i, y in enumerate(ys):
                item, frequency, adjusted_frequency, sorted_grouped_item = get_samples_for_bionames(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, step=step, use_uncertainty=args.use_uncertainty, exploration_rate=exploration_rate, threshold=threshold)
                frequencies.extend(frequency)
                adjusted_frequencies.extend(adjusted_frequency)
                sorted_grouped_items.extend(sorted_grouped_item)
                for y in item:
                    new_ys.append(y)
                    new_y_paths.append(y_paths[i] + [y])
                    if step > 0:
                        new_relations.append(relations[i] + [json.loads(y)['Relation']])
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys] 
            
        # new_ys = list(itertools.chain(*new_ys))
        # print('-- new ys pre vote --')
        # for y in new_ys:
        #     print(json.loads(y)['Biological Process'])
        # there is only one output each time rn
        ids = list(range(len(new_ys)))
        # evaluation
        if args.method_evaluate == 'vote':
            votes = get_votes(task, x, new_ys, args.n_evaluate_sample)
            values = votes
        elif args.method_evaluate == 'value':
            votes = get_values(task, x, new_ys, args.n_evaluate_sample)
            values = votes
        elif args.method_evaluate == 'votes_for_bionames':
            votes, vote_outputs = get_votes_for_bionames(task, x, new_ys, args.n_evaluate_sample, step, None, args)
            values = votes
        elif args.method_evaluate == 'multi_voters':
            votes, vote_outputs = get_multivotes_for_bionames(task, x, new_ys, args.n_evaluate_sample, args)
            values = votes
        elif args.method_evaluate == 'uncertainty_voters':
            votes, vote_outputs = get_votes_for_bionames(task, x, new_ys, args.n_evaluate_sample, step, None, args)
            np_votes = np.array(votes)
            np_freq = np.array(frequencies)
            values = np_votes * np_freq
            values = values.tolist()
            print(f'np_votes:{np_votes}, np_freq:{np_freq},\nvalues:{values}')
        elif args.method_evaluate == 'medagents':
            args.n_evaluate_sample = args.ans_num
            args.n_select_sample = args.ans_num
            votes, data_info = get_medagents_votes(task, x, new_ys, label, args, False)
            vote_outputs = data_info['raw_output']
            values = votes
            print(values, vote_outputs)
        elif args.method_evaluate == 'medagents_w_tools':
            args.n_evaluate_sample = args.ans_num
            args.n_select_sample = args.ans_num
            votes, data_info = get_medagents_votes(task, x, new_ys, label, args, True)
            vote_outputs = data_info['raw_output']
            values = votes
            
        stop_metrics = None
        if stop_expansion == True:
            stop_metrics, stop_outputs = get_stop_metrics_for_bionames(task, x, new_ys, args.n_evaluate_sample, args)
            np_stop_metrics = np.array(stop_metrics)
            print(f'stop_metrics: {stop_metrics}')
            values = values * np_stop_metrics
            values = values.tolist()
            print(f'values:{values}')
            
        # selection
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            sorted_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
            if stop_expansion:
                # Step 3 & 4: Check if the top 2 values have corresponding stop_metrics >= 1 and return their ids
                select_ids = [ids[i] for i in sorted_ids if stop_metrics[i] >= 1]
                final_ys.extend([new_ys[ids[i]] for i in sorted_ids if stop_metrics[i] == 0])
#                 final_ys.extend([new_ys[ids[i]] for i in sorted_ids])
            else:
                select_ids = sorted_ids
                final_ys.extend([new_ys[ids[i]] for i in sorted_ids])
                
        print('select_ids',select_ids)
        select_new_y_paths = [new_y_paths[select_id] for select_id in select_ids]
        omit_y_paths = [new_y_paths[id] for id in ids if id not in select_ids]
        omit_y_paths = [[json.loads(y)['Biological Process'] for y in omit_y_path] for omit_y_path in omit_y_paths]

        if step > 0:
            select_new_relations = [new_relations[select_id] for select_id in select_ids]
            omit_relations = [new_relations[id] for id in ids if id not in select_ids]
            omit_relations = [relation[0] for relation in omit_relations]
            # print('-- relations --')
            # print(select_new_relations)
            # print(omit_relations)

        # print('-- paths --')
        for i in range(len(new_y_paths)):
            path = [json.loads(y)['Biological Process'] for y in new_y_paths[i]]
            # print(' -> '.join([el for el in path]))
            value = values[i]
            # if trie.search(path) is None:
            if True:
                if step > 0:
                    trie.insert(path, value, new_relations[i])
                else:
                    trie.insert(path, value, [])
            else:
                if step > 0:
                    trie.insert(path, max(value, trie.search(path)), new_relations[i])
                else:
                    trie.insert(path, max(value, trie.search(path)), [])
    

        # criticisms = []
        # for omit_y_path in omit_y_paths:
        #     criticisms.append(get_criticism_for_bionames(task, x, omit_y_path))
        # mem.extend(criticisms)
        # print('solve -- criticisms', criticisms)
        
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        if len(omit_y_paths) > 0:
            pass
            # print('omitted paths', omit_y_paths)
        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            # bp_y_paths = [[json.loads(y)['Biological Process'] for y in y_path] for y_path in new_y_paths]
            # print(f'-- new_ys --: {[json.loads(y)["Biological Process"] for y in sorted_new_ys]}\n-- sol values --: {sorted_values}\n-- choices --: {[json.loads(y)["Biological Process"] for y in select_new_ys]}\n')
            # print('-- y paths --: {}\n'.format([' -> '.join(path) for path in bp_y_paths]))
            # if step > 0:
            #     print('-- Relations --')
            #     print([json.loads(y)["Relation"] for y in select_new_ys])
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'votes': votes, \
                      'select_new_ys': select_new_ys, 'frequencies':frequencies, 'adjusted_frequencies':adjusted_frequencies, \
                      'stop_metrics': stop_metrics, 'sorted_grouped_items':sorted_grouped_items})
        ys = select_new_ys
            
        y_paths = select_new_y_paths
        if step > 0:
            relations = select_new_relations
        else:
            relations = [[''] for _ in range(len(y_paths))]
        
        dot = trie.visualize()
        dot.render('viz/trie_visualization_{}'.format(idx), format='png')
        print('Time taken:', time.time() - start)
        #This is where we break the loop, if none of them should be continue to expand, stop here. 
        if select_ids == []:
            break
            
    final_ys.extend(ys)
    final_ys = list(set(final_ys))
#     print('final_ys',final_ys)
    
    sw_prompt = ''   
    frequencies = []
    sorted_grouped_items = []
    if args.task == 'bio_name':
        if args.final == 'semantic_web':
            sw_prompt = to_semantic_web(infos)
            final_answer, new_ys = get_final_answer_for_bionames_sw(task, x, sw_prompt, args.n_generate_sample, prompt_sample=args.prompt_sample)

        else:
            if args.use_uncertainty == False:
                final_answer, new_ys = get_final_answer_for_bionames(task, x, final_ys, args.n_generate_sample, prompt_sample=args.prompt_sample, threshold=threshold)
            else:
                final_answer, samples, frequencies, adjusted_frequencies, grouped_items_by_index =\
                    get_final_answer_for_bionames(task, x, final_ys, args.n_generate_sample, prompt_sample=args.prompt_sample, use_uncertainty=args.use_uncertainty, threshold=threshold)
                
            
        infos.append({'step': step+1, 'x': x, 'ys': final_ys, 'new_ys': new_ys, 'values': None, 'select_new_ys': None, 'frequencies':frequencies, 'sorted_grouped_items':sorted_grouped_items})
    if to_print: 
        pass
        # print('solve -- ys', ys)

    
    final_path = None
    for i, path in enumerate(y_paths):
        path = [json.loads(y)['Biological Process'] for y in path]
        if path[-1].split('(')[0] == final_answer:
            final_path = path
            final_relation = relations[i]
            break
    
    if final_path:
        trie.insert(final_path, '*', final_relation)
    else:
        print('No Final Path Found')
        print('y paths', y_paths)
        print('final answer', final_answer)
    dot = trie.visualize()
    dot.render('viz/trie_visualization_{}'.format(idx), format='png')

    
    return final_answer, ys, {'steps': infos}, trie, sw_prompt

def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}

import json
def cut_and_sum(input_list, cut_by):
    # Split the list into two parts
    part1 = input_list[:cut_by]
    part2 = input_list[cut_by:]
    # Sum the corresponding elements of the two parts
    summed_list = [x + y for x, y in zip(part1, part2)]
    return summed_list

def rank_by_value_and_output_indices(input_list):
    # Pair each value with its index
    indexed_values = list(enumerate(input_list))
    # Sort the pairs by value, then extract the indices
    sorted_indices = [index for index, value in sorted(indexed_values, key=lambda x: x[1], reverse=True)]
    return sorted_indices

def to_semantic_web(info):
    dic_list = [{
      "nodes": [{"id": "Biological Process", "edges":[]}]
    }]
    parents = ['Biological Process']
    indices = [0,0]
    count = 0
    for j, t in enumerate(info):
        new_parents = []
        parent_id = 0
        dic_list.append({"nodes":[]})
        if t['values']:
            for i, (y,v) in enumerate(zip(t['new_ys'], t['values'])):
                if i <= 2:
                    parent_id = 0
                else:
                    parent_id = 1
                parent_node = parents[parent_id]

                y = json.loads(y)
                if v >= 1:
                    new_parents.append(y['Biological Process'])  
                dic_list[j+1]["nodes"].append({"id": y['Biological Process'], "edges": []})

                if "Relation" in y:
                    relation = y['Relation']
                else:
                    relation = 'is_a'

                for k, d in enumerate(dic_list[j]['nodes']):
                    d = dic_list[j]['nodes'][k]
                    if d['id'] == parent_node:
                        dic_list[j]['nodes'][k]['edges'].append({"to":  y['Biological Process'], "type": relation})
                count +=1
            # Rank by value and output indices for the summed lists
            parents = new_parents
            
    # Combining all 'nodes' into the first 'nodes'
    for additional in dic_list[1:]:  # Start from the second item
        dic_list[0]['nodes'].extend(additional['nodes'])

    # The result is now in the first item of list_of_dicts
    combined_nodes = dic_list[0]['nodes']
    final_structure = {'nodes': combined_nodes}
    final_structure
    return final_structure
    