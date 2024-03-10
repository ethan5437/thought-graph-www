propose_prompt = '''
You are an efficient and insightful assistant to a molecular biologist.
         
Be concise; do not use unnecessary words. Be specific; avoid overly general
statements, such as "the proteins are involved in various cellular processes."
Be factual; do not editorialize.
Be inclusive; be sure to include all proteins.
Be comprehensive, but don't overgeneralize.   
Stay on the subject; do not deviate from the goal. (Goal: Propose a brief name for the most prominent biological process performed by the system.) 
    
Here are the interacting proteins:
Proteins: {input}
Goal: Propose a brief name for the most prominent biological process performed by the system. 

You should write your thought process in steps. For now, please give me only Step 1 and as many potential biological perspectives as possible (at least three) for this set of genes. 

Please answer strictly in JSON format, do not use \\n:
{format_0}
'''

format_0 = [{"Step": '1',\
             "Biological Perspective": "Your First Biological Perspective", \
             "Analysis & Reason": {"Genes involved": "list all the genes involved (count: number of genes involved).", \
                                   "Reason": "Your reason should include why did you choose this name? How does this help you to infer the name?"}},\
            {"Step": '1',\
             "Biological Perspective": "Your First Biological Perspective", \
             "Analysis & Reason": {"Genes involved": "list all the genes involved (count: number of genes involved).", \
                                   "Reason": "Your reason should include why did you choose this name? How does this help you to infer the name?"}},\
]

next_step_prompt = '''
You are an efficient and insightful assistant to a molecular biologist.

Be concise; do not use unnecessary words. Be specific; avoid overly general
statements, such as "the proteins are involved in various cellular processes."
Be factual; do not editorialize.
Be inclusive; be sure to include all proteins.
Be comprehensive, but don't overgeneralize.   
Stay on the subject; do not deviate from the goal. (Goal: Propose a brief name for the most prominent biological process performed by the system.)

Here are the interacting proteins:
Proteins: {input}

Here are the biological perspectives we've discussed: 
---
{y}
---

Base on the last step (Step: {prev_step_num}):
1) Determine if the selected perspective is specific enough to infer a biological name. 
2) Please review the selected perspective, edit improve the selected perspectives based on the pros and cons to provide as many biological perspectives as possible (at least three), and describe their interaction. 

step_num = {step_num}

Please answer strictly in JSON format, do not use \\n::
{format_1}
'''

format_1 = [{"Step": "{step_num}",\
           "Selected Biological Perspective": "The selected perspective",\
           "New Biological Perspective": "Your First Biological Perspective",\
           "Analysis & Reason": {"Genes involved": "list all the genes involved (count: number of genes involved).",\
                                 "Reason": "Your reason should include why did you choose this name? How does this help you to infer the name?"}},\
          {"Step": "{step_num}",\
           "Selected Biological Perspective": "The selected perspective",\
           "New Biological Perspective": "Your First Biological Perspective",\
           "Analysis & Reason": {"Genes involved": "list all the genes involved (count: number of genes involved).",\
                                 "Reason": "Your reason should include why did you choose this name? How does this help you to infer the name?"}}]

last_step_prompt = '''You are an efficient and insightful assistant to a molecular biologist.
         
Be concise; do not use unnecessary words. Be specific; avoid overly general
statements, such as "the proteins are involved in various cellular processes."
Be factual; do not editorialize.
Be inclusive; be sure to include all proteins.
Be comprehensive, but don't overgeneralize.   
Stay on the subject; do not deviate from the goal. (Goal: Propose a brief name for the most prominent biological process performed by the system.)
    
Here are the interacting genes:
{input}

Goal: Propose a brief name for the most prominent biological process performed by the system. 

Here are the steps we've discussed: 

{y}

Based on all the steps before, write the answer to the goal: propose a brief name for the most prominent biological process performed by the system, and write your reasoning.

Please answer strictly in JSON format, do not use \\n:: 
{format_3}
'''

format_3 = {'Reasons': 'Your Reason',\
 'Proposed Name': 'Your Proposed Name'}


vote_prompt = '''
You are an efficient and insightful assistant to a molecular biologist.

Here are the biological perspectives we're going to discuss:
{choice}

Goal: Analyze whether each biological perspective appropriately describes the proteins listed below, as well as analyze the criteria described in the following: 
Be concise; do not use unnecessary words. Be specific; avoid overly general
statements, such as "the proteins are involved in various cellular processes."
Be factual; do not editorialize.
Be inclusive; be sure to include all proteins.
Be comprehensive, but don't overgeneralize.   
Stay on the subject; do not deviate from the goal. (Goal: Propose a brief name for the most prominent biological process performed by the system.) 

Proteins: {input}

Lastly, provide pros and cons of each biological perspective and conclude the best biological perspective is 's', where 's' is the index of the choice, 0 is the first index.

Please answer strictly in JSON format, do not use \\n::
{format_2}
'''

format_2 = {'Analysis': [\
    {'Biological Perspective': 'The First Biological Perspective',\
   'Pros': 'Your Pros', \
   'Cons' : 'Your Cons'}, \
    {'Biological Perspective': 'The Second Biological Perspective', \
   'Pros': 'Your Pros', \
   'Cons' : 'Your Cons'}], \
  'Best biological perspective': {'Biological Perspective': 'The Best Biological Perspective', 'index': 's'}}
# '''
# Given the following biological perspectives, decide which biological perspective is more promising. Analyze each choice in detail ; provide pros and cons, then conclude in the last line "The best biological perspective is s,  where s is the integer id of the choice.

# {choice}

# Format: 
# Biological Perspective 1:
# - Pros: Your Pros
# - Cons Your Cons
# Biological Perspective 2:
# - Pros: Your Pros
# - Cons Your Cons
# ...
# The best biological perspective is s
# '''
#edit: perspective, promising

GO_Enrich_prompt = '''
Here are the perspectives we've discussed: 
{y}

Enrichr has proposed these perspectives: {terms}. 
Could you use these perspectives to compare against the proposed biological perspectives mentioned above, name the differences between them, and provide a summary? 
Conclude, out of all the perspectives, including perspectives by Enrichr, which perspective is the best. 

Please answer strictly in JSON format:
{format_4}
'''

format_4 = {'Analysis': [
            {'Biological Perspective': 'Your First Biological Perspective', 
             'Comparison': 'Comparing the biological perspectives to terms proposed by Enrichr'}, 
            {'Biological Perspective': 'Your Second Biological Perspective', 
             'Comparison': 'Comparing the biological perspectives to terms proposed by Enrichr'}], 
             'Best biological perspective': {'Biological Perspective': 'The Best Biological Perspective'}
           }


criticism_prompt = '''
You are an efficient and insightful assistant to a molecular biologist.

You have been working on a project to generate a brief name for the most prominent biological process performed by the set of proteins listed below.

Proteins: {input}

You have generated different candidate biological processes in an iterative way. Initially, you proposed a biological proposed one or more candidate biological processes. Then, for each candidate biological process, you reflected on the candidate biological process and edited it to become more accurate.

In this process, you decided that the following biological process, generated by the sequence below, was a poor candidate for the most prominent biological process performed by the set of proteins listed above. In one or two sentences, provide a criticism of why this biological process is a poor candidate.

Biological Process Generation Sequence: {omit_y_path}

Please answer strictly in JSON format:
{format_5}
'''

format_5 = {'Criticism': 'Your Criticism'}