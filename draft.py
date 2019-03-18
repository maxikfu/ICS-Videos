import os
# import spacy
# import json
# import numpy as np
#
# nlp = spacy.load('en_core_web_sm')
#
# video_id = 4623
# video_directory = 'data/GEOL1330Fall18_Jinny/v' + str(video_id) + '/'
# os.chdir(video_directory)
# try:
#     with open('../tt_Earth_cleaned.json', 'r') as f:
#         raw_book_segs = f.readlines()
# except IOError:
#     print("Error: Text book is not found.")
#     exit()
#
# n_sents = []
# n_tokens = []
# for line in raw_book_segs:
#     j = json.loads(line)
#     text = j['text']
#     doc = nlp(text)
#     s_c  = 0
#     for s in doc.sents:
#         s_c += 1
#         n_tokens.append(len([t for t in s]))
#     n_sents.append(s_c)
# print(np.mean(n_sents), np.std(n_sents))
# print(np.mean(n_tokens), np.std(n_tokens))

os.chdir('data/Evaluation/')
filenames = ['Microbiology_An_Evolving_Science_4th_Edition_part1.txt', 'Microbiology_An_Evolving_Science_4th_Edition_part2.txt', 'Microbiology_An_Evolving_Science_4th_Edition_part3.txt', 'Microbiology_An_Evolving_Science_4th_Edition_part4.txt']
with open('Microbiology_full.txt', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)
