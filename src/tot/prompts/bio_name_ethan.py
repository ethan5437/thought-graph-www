# TODO: is there extra whitespace surrounding the input? proteins or genes?
# TODO: I think sometimes it is doing step num 1, 2, 3 for different items in same step

# CURRENTLY: generate three reasons for each step

system_prompt = 'You are a helpful and knowledgable assistant to a molecular biologist. Respond to questions in JSON format, please stictly following this template: {json_format}.' 

# propose initial biological processes
# format_0 = '''{"Answer 1": {"Step": "1", "Biological Process": "<Your first proposed biological process>", "Reason": "<Why did you choose this name?>"},
#             "Answer 2": {"Step": "1", "Biological Process": "<Your second proposed biological process>", "Reason": "<Why did you choose this name?>"},
#             "Answer 3": {"Step": "1", "Biological Process": "<Your third proposed biological process>", "Reason": "<Why did you choose this name?>"},...}'''

format_0 = {"Answer": {"Step": "1", "Biological Process": "<Your proposed biological process>", "Reason": "<Why did you choose this name?>"}}

# propose more specific biological processes
# format_1 = {"Answer 1": {
#             "Step": "{step_num}",\
#             "Previous Biological Process": "<The previous biological process>",\
#             "New Biological Process": "<Your first proposed biological process, more specific than the previous",\
#             "Relation": "<How does the new biological process relate to the previous biological process?",\
#             "Reason": "<Why did you choose this name? Which genes are relevant to this process?>",\
#             "Relevance": "<Explanation for Relevance>",\
#             "Uniqueness": "<Explanation for Uniqueness>",\
#             "Scientific Associations": "<Explanation for Scientific Associations>"},\
#             "Answer 2": {
#             "Step": "{step_num}",\
#             "Previous Biological Process": "<The previous biological process>",\
#             "New Biological Process": "<Your second proposed biological process, more specific than the previous",\
#             "Relation": "<How does the new biological process relate to the previous biological process?",\
#             "Reason": "<Why did you choose this name? Which genes are relevant to this process?>",
#             "Relevance": "<Explanation for Relevance>",\
#             "Uniqueness": "<Explanation for Uniqueness>",\
#             "Scientific Associations": "<Explanation for Scientific Associations>"},\
#             "Answer 3": {
#             "Step": "{step_num}",\
#             "Previous Biological Process": "<The previous biological process>",\
#             "New Biological Process": "<Your third proposed biological process, more specific than the previous",\
#             "Relation": "<How does the new biological process relate to the previous biological process?",\
#             "Reason": "<Why did you choose this name? Which genes are relevant to this process?>",\
#             "Relevance": "<Explanation for Relevance>",\
#             "Uniqueness": "<Explanation for Uniqueness>",\
#             "Scientific Associations": "<Explanation for Scientific Associations>"}}
format_1 = {"Answer": {"Step": "{step_num}",\
            "Previous Biological Process": "<The previous biological process>",\
            "New Biological Process": "<Your proposed biological process, more specific than the previous",\
            "Relation": "<How does the new biological process relate to the previous biological process?",\
            "Reason": "<Why did you choose this name? Which genes are relevant to this process?>"}}

# vote prompt
format_2 = {'Votes': [{'Biological Process': '<The First biological process>'}, {'Biological Process': '<The Second biological process>'}]}

# last step prompt
format_3 = {'Biological Process': '<The best biological process>', 'Reason': '<Your reasoning>'}

format_7 = '''{'Answer': [{"Biological Process": "First Biological Process Name", "Specificity Score": 1~10, "Specific Enough": True or False, "Evaluation": "Provide a brief explanation for your specificity score, including considerations of relevance, uniqueness, and scientific associations. Explain whether the biological process name is specific enough based on a threshold score of 8."}, {"Biological Process": "First Biological Process Name", "Specificity Score": 1~10, "Specific Enough": True or False, "Evaluation": "Provide a brief explanation for your specificity score, including considerations of relevance, uniqueness, and scientific associations. Explain whether the biological process name is specific enough based on a threshold score of 8."},...]}'''

format_6 = {'Similarity Score': '<Your similarity score>'}

propose_instruction = '''You are given a set of genes, and your task is to propose three high-level biological processes that may be likely to be performed by the system involving expression of these genes.\n'''
propose_one_instruction = '''You are given a set of genes, and your task is to propose one high-level biological process that may be likely to be performed by the system involving expression of these genes.\n'''

propose_content = f'''
Biological processes are organized in a hierarchical ontology, and the most general biological processes are at the top of the hierarchy.
Biological processes can have four relations:
1. is a: A is a B if biological process A is a subtype of biological process B. If A is a subtype of B, then we say A is more specific than B.
2. has part: A has part B if the biological process A always has part B. If A exists, then B will always exist. If A has part B, then we say B is more specific than A.
3. part of: A is part B if, whenever biological process A exists, it is as a part of biological process B. If A is part of B, then we say A is more specific than B.
4. regulates: A regulates B if biological process A always regulates biological process B. If A regulates B, then we say B is more specific than A.

These relationships create a hierarchical ontology. Your job is to propose one biological processes that are as general as possible.

Here is the set of genes:
Genes: {input}'''
propose_prompt = propose_instruction + propose_content


next_step_instruction = '''Given a set of genes and proposed biological processes describing the system, your task is to generate more specific biological processes describing the system. A specific biological process includes: 
Relevance: How relevant is the biological process to the functions of the provided genes? Consider the primary functions of these genes and how they relate to the biological process in question.

Uniqueness: How unique is the biological process to these genes? Assess if the biological process is commonly associated with these genes or if it's a broad process that could apply to many other genes as well.

Scientific Associations: Based on scientific literature up to your last training cut-off, are there any known associations between the genes and the biological process? Highlight any direct connections or research findings that link these genes to the biological process mentioned.\n'''

next_step_one_instruction = '''Given a set of genes and proposed biological processes describing the system, your task is to generate a more specific biological process describing the system.\n'''

next_step_content = '''
Biological processes are organized in a hierarchical ontology, and the most general biological processes are at the top of the hierarchy.
Biological processes can have three relations:
1. is a: A is a B if biological process A is a subtype of biological process B. If A is a subtype of B, then we say A is more specific than B.
2. has part: A has part B if the biological process A always has part B. If A exists, then B will always exist. If A has part B, then we say B is more specific than A.
3. part of: A is part B if, whenever biological process A exists, it is as a part of biological process B. If A is part of B, then we say A is more specific than B.
4. regulates: A regulates B if biological process A always regulates biological process B. If A regulates B, then we say B is more specific than A.

You should propose one biological process that are more specific than the proposed biological process. You should describe how the proposed biological processes relates to the current biological process using one of the four relations above, and then give your reasoning for why the proposed biological processes describes the system.

Here is the set of genes:
Genes: {input}

Here is the current biological process:
---
{y}
---
'''

next_step_prompt = next_step_instruction + next_step_content


last_step_instruction = '''Given a set of genes and proposed biological processes describing the system, your task is to choose the best biological process. The chosen biological process must come from the list of biological processes given to you.\n

Your evaluation should be based on the following criteria:
Relevance: How relevant is the biological process to the functions of the provided genes? Consider the primary functions of these genes and how they relate to the biological process in question.

Uniqueness: How unique is the biological process to these genes? Assess if the biological process is commonly associated with these genes or if it's a broad process that could apply to many other genes as well.

Scientific Associations: Based on scientific literature up to your last training cut-off, are there any known associations between the genes and the biological process? Highlight any direct connections or research findings that link these genes to the biological process mentioned.\n
'''

last_step_content = '''
Genes: {input}

Proposed Bioligical Processes:
{y}

Out of the proposed biological processes, choose the most accurate biological process.'''

last_step_sw_prompt = '''Given a set of genes and a semantic biological processes web describing the system, your task is to choose the most accurate biological process from the semantic web. Choose the node in the given semantic web and provide the term associated with the "id" key. For example, if you encounter a node with "id": "Regulation of cholesterol metabolic process", your answer should be "Regulation of cholesterol metabolic process". The chosen biological process must come from the semantic web given to you. 

Genes: {input}

Semantic Web:
{y}

Answer: '''

last_step_prompt = last_step_instruction + last_step_content


vote_instruction = '''Given a set of genes and proposed biological processes describing the system, your task is to vote on the two best biological processes describing the system.'''
vote_content = '''

Here is the set of genes:
Genes: {input}

Here are the biological processes for you to vote on:
Biological Processes: {choice}

Given the set of genes and the proposed biological processes, vote on the two best biological processes.'''

vote_prompt = vote_instruction + vote_content

stop_instruction = '''Given a list of genes representing a biological system, your task is to evaluate the specificity of each provided biological process name. Determine if the name adequately describes the system's function or if a more specific name is required.

'''
stop_content = '''
Given a list of gene names and names of biological processes, your task is to evaluate the specificity of each biological process name to the provided set of genes on a scale from 1 to 10. A specificity score of 1 indicates that the biological process name is very general and not closely related to the genes, while a score of 10 indicates that the biological process name is highly specific and closely related to the genes. Your evaluation should be based on the following criteria:

Relevance: How relevant is the biological process to the functions of the provided genes? Consider the primary functions of these genes and how they relate to the biological process in question.

Uniqueness: How unique is the biological process to these genes? Assess if the biological process is commonly associated with these genes or if it's a broad process that could apply to many other genes as well.

Scientific Associations: Based on scientific literature up to your last training cut-off, are there any known associations between the genes and the biological process? Highlight any direct connections or research findings that link these genes to the biological process mentioned.

Remember, a specificity score of 8 or above indicates that the biological process name is "specific enough" for the provided genes. Evaluate whether each provided biological process name meets this threshold based on the criteria outlined above. The "Specific Enough" field should be "True" if the specificity score is 8 or above, indicating that the name is specific enough, and "False" if the score is below 8.

Gene Set: {input}

Biological Processes: {choice}
'''

stop_prompt = stop_instruction + stop_content



other_examples = '''
1. For the gene set [TGFBR2, BMPR1A, BMPR2, ZFPM1, HEY2], the biological process "tricuspid valve morphogenesis" is considered specific enough.
2. For the gene set [TGFBR2, BMPR1A, BMPR2, ZFPM1, HEY2], the biological process "valve morphogenesis" is considered not specific enough.
3. For the gene set [IL15, CYP24A1, FOLR1, FOXA2, BRIP1, GAS6, TRIM24, PHEX, FES, CASR, LPL, GDAP1, PIM1, FGF23, ATP2B1, NR1H4, DCPS, MED1, SRF, SFRP1, CYP27B1, BGLAP, PENK, MN1, FOLR2, HMOX1, PDK2, USF2, NCOA1, USF1, XBP1, KANK2, COL1A1, POSTN, LEP, RXRB, SNAI2, CDKN2B, VDR, SNW1, NDOR1, RXRA], the biological process "cellular response to nutrient" is considered specific enough.
4. For the gene set [IL15, CYP24A1, FOLR1, FOXA2, BRIP1, GAS6, TRIM24, PHEX, FES, CASR, LPL, GDAP1, PIM1, FGF23, ATP2B1, NR1H4, DCPS, MED1, SRF, SFRP1, CYP27B1, BGLAP, PENK, MN1, FOLR2, HMOX1, PDK2, USF2, NCOA1, USF1, XBP1, KANK2, COL1A1, POSTN, LEP, RXRB, SNAI2, CDKN2B, VDR, SNW1, NDOR1, RXRA], the biological process "cellular metabolic process" is considered not specific enough.
5. For the gene set [VPS13B, ACTL7A, AGFG1, SOX30, SPACA1, PLA2G3, ACRBP, MFSD14A, PAFAH1B1, RFX2, ZPBP, FABP9, CCDC136, GARIN1A, TMF1, GARIN1B, SPINK2, TBPL1, NECTIN2, ACTL9, ZPBP2, PLN, TBC1D20, SPPL2C, KNL1], the biological process "acrosome assembly" is considered specific enough.
6. For the gene set [VPS13B, ACTL7A, AGFG1, SOX30, SPACA1, PLA2G3, ACRBP, MFSD14A, PAFAH1B1, RFX2, ZPBP, FABP9, CCDC136, GARIN1A, TMF1, GARIN1B, SPINK2, TBPL1, NECTIN2, ACTL9, ZPBP2, PLN, TBC1D20, SPPL2C, KNL1], the biological process "gamete maturation" is considered not specific enough.
7. For the gene set [DRD3, ASCL1], the biological process "musculoskeletal movement, spinal reflex action" is considered specific enough.
8. For the gene set [DRD3, ASCL1], the biological process "reflex actions" is considered noy specific enough.
9. For the gene set [OLFM1, TGFBR2, BMP2, TGFB2, TBX5, APLNR, HEYL, EFNA1, MDM4, GJA5, BMPR2, SLIT3, HEY1, NOTCH1, GATA4, MDM2, TWIST1, CCN1, ACVR1, SMAD4, HEY2, SOX4, SMAD6, BMPR1A, DCHS1, ZFPM1, TBX20], the biological process "atrioventricular valve development" is considered specific enough.
10. For the gene set [OLFM1, TGFBR2, BMP2, TGFB2, TBX5, APLNR, HEYL, EFNA1, MDM4, GJA5, BMPR2, SLIT3, HEY1, NOTCH1, GATA4, MDM2, TWIST1, CCN1, ACVR1, SMAD4, HEY2, SOX4, SMAD6, BMPR1A, DCHS1, ZFPM1, TBX20], the biological process "Cardiac valve development" is considered not specific enough.



5. For the gene set [OLFM1, TGFBR2, BMP2, TGFB2, TBX5, APLNR, HEYL, EFNA1, MDM4, GJA5, BMPR2, SLIT3, HEY1, NOTCH1, GATA4, MDM2, TWIST1, CCN1, ACVR1, SMAD4, HEY2, SOX4, SMAD6, BMPR1A, DCHS1, ZFPM1, TBX20], the biological process "atrioventricular valve development" is considered specific enough.
6. For the gene set [WARS1, WARS2], the biological process "musculoskeletal movement, spinal reflex action" is considered specific enough.
7. For the gene set [DEGS2, PIGG, SGPP1, SERINC5, SMPDL3B, ACER2, VPS54, DEGS1, SERINC2, ST8SIA3, PIGB, HACD3, AGMO, FUT3, ELOVL5, PIGO, ALOX12B, PIGU, NAAA, ST6GALNAC5, ST6GALNAC6, FUT7, BAX, HACD4, PNLIPRP2, AGK, CLN6, NEU2, GAL3ST4, P2RX7, PIGF, ELOVL4, CERS2, ORMDL1, MPPE1, CERS1, SPTLC1, VAPA, ZPBP2, FUT9, MGST2, SMPD2, ORMDL3, SFTPB, PLPP2, B3GALT4, ALOXE3, PSAPL1, ORMDL2, B4GALT3, GBA1, LCT, FUT5, SPHK1, B4GALT6, PIGT, PIGX, ABCA2, ST8SIA4, PLPP3, SMPD4, PRKD1, KIT, CERK, GLTP, CERS6, SGMS2, ARV1, PSAP, PIGS, GALC, ELOVL6, SPTLC2, PLA2G6, DPM3, PIGP, CPTP, PIGK, TM9SF2, SGPL1, B3GNT5, NAGA, SERINC1, B3GALNT1, ABCA12, ST3GAL4, LARGE1, CCN1, SPTSSB, GAL3ST2, GLA, ASAH2B, TECR, ST3GAL1, B4GALNT1, ASAH1, ELOVL1, NEU3, PIGZ, A3GALT2, SPTSSA, SPHK2, B3GALT2, KDSR, SAMD8, ST8SIA1, SCCPDH, CERT1, PRKAA1, PGAP2, ABCA8, TH, PIGA, HEXA, NEU4, PGAP4, PPP2R1A, PGAP1, SGMS1, PIGY, PPT1, ST3GAL5, FA2H, ST8SIA2, PIGH, PPP2CA, ALDH3B1, PIGW, FUCA1, GPAA1, ACER1, GLB1, B4GALT4, PIGQ, SMPDL3A, CYP4F22, PRKD3, SIRT3, SMPD1, ELOVL7, ST3GAL6, PIGV, PIGM, B4GALT5, GPLD1, DPM1, ST3GAL3, ITGB8, ST8SIA5, PPM1L, GAL3ST3, UGT8, ST8SIA6, SERINC3, FADS3, PGAP3, CERKL, CERS5, PIGN, SUMF1, PRKD2, ASAH2, B3GALT1, GBA3, GBGT1, SPNS2, ELOVL3, SLC30A5, ST6GALNAC3, PLPP1, PNPLA1, CLN8, SGPP2, ENPP2, GAL3ST1, SMPD3, HACD1, GBA2, DPM2, PEMT, CSNK1G2, HACD2, TLCD3B, ENPP7, TEX2, NSMAF, UGCG, ELOVL2, FUT6, CERS3, SPTLC3, P2RX1, ACER3, CERS4, CYP1B1, HTRA2, NEU1, OSBP, ST6GALNAC4, PLA2G15, FUT2, A4GALT, HEXB, ST3GAL2, PRKCD, CWH43, C20orf173, PIGC, GM2A, PIGL], the biological process "negative regulation of bone remodeling" is considered specific enough.
'''


### DONT USE

GO_Enrich_prompt = '''Here are the processes we've discussed: 
{y}

Enrichr has proposed these processes: {terms}. 
Could you use these processes to compare against the proposed biological processes mentioned above, name the differences between them, and provide a summary? 
Conclude, out of all the processes, including processes by Enrichr, which process is the best. 

Please answer strictly in JSON format:
{format_4}'''

format_4 = {'Analysis': [
            {'Biological Process': 'Your First Biological Process', 
             'Comparison': 'Comparing the biological processes to terms proposed by Enrichr'}, 
            {'Biological Process': 'Your Second Biological Process', 
             'Comparison': '<Comparing the biological processes to terms proposed by Enrichr>'}], 
             'Best Biological Process': {'Biological Process': '<The best biological Process>'}
           }


criticism_prompt = '''You are an efficient and insightful assistant to a molecular biologist.

You have been working on a project to generate a brief name for the most prominent biological process performed by the set of proteins listed below.

Proteins: {input}

You have generated different candidate biological processes in an iterative way. Initially, you proposed a biological proposed one or more candidate biological processes. Then, for each candidate biological process, you reflected on the candidate biological process and edited it to become more accurate.

In this process, you decided that the following biological process, generated by the sequence below, was a poor candidate for the most prominent biological process performed by the set of proteins listed above. In one or two sentences, provide a criticism of why this biological process is a poor candidate.

Biological Process Generation Sequence: {omit_y_path}

Please answer strictly in JSON format:
{format_5}'''

format_5 = {'Criticism': 'Your Criticism'}

### END DONT USE

similarity_prompt = '''How similar are the following two biological processes? Please answer on a scale from 1 to 10, with 1 being not similar at all and 10 being very similar.

Biological Process 1: {y1}
Biological Process 2: {y2}'''


vote_gprofiler_prompt = '''
Given the gene set: {input}

Our model has proposed {y} to be the term of this gene set. 

We used GProfiler to conduct an enrichment analysis
Here are the top {n} results:
{table}

Would the proposed term be suitable and supported by GProfiler? 
'''