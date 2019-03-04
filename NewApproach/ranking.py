import pytextrank
import pprint
import json



with open('tt_Earth_cleaned.json', 'r') as f:
    stage0 = json.loads(f.readlines()[2])


path_stage1 = "o1.json"
path_stage2 = "o2.json"
path_stage3 = "o3.json"

"""Perform statistical parsing/tagging on a document in JSON format"""
with open(path_stage1, 'w') as f:
    for graf in pytextrank.parse_doc([stage0]):
        f.write("%s\n" % pytextrank.pretty_print(graf._asdict()))
        # to view output in this notebook
        # print(pytextrank.pretty_print(graf))
graph, ranks = pytextrank.text_rank(path_stage1)
"""Collect and normalize the key phrases from a parsed document"""
with open(path_stage2, 'w') as f:
    for rl in pytextrank.normalize_key_phrases(path_stage1, ranks):
        f.write("%s\n" % pytextrank.pretty_print(rl._asdict()))
        # to view output in this notebook
        # print(pytextrank.pretty_print(rl))
kernel = pytextrank.rank_kernel(path_stage2)
"""Calculate a significance weight for each sentence, 
using MinHash to approximate a Jaccard distance from key phrases determined by TextRank"""
with open(path_stage3, 'w') as f:
    for s in pytextrank.top_sentences(kernel, path_stage1):
        f.write(pytextrank.pretty_print(s._asdict()))
        f.write("\n")
        # to view output in this notebook
        print(s)
"""Summarize a document based on most significant sentences and key phrases"""
phrases = ", ".join(set([p for p in pytextrank.limit_keyphrases(path_stage2, phrase_limit=12)]))
sent_iter = sorted(pytextrank.limit_sentences(path_stage3, word_limit=150), key=lambda x: x[1])
print(sent_iter)
# s = []
#
# for sent_text, idx in sent_iter:
#     s.append(pytextrank.make_sentence(sent_text))
#
# graf_text = " ".join(s)
# pprint.pprint("**excerpts:** %s\n\n**keywords:** %s" % (graf_text, phrases,))
#
