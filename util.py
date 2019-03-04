import pytextrank

path_stage0 = "data/textrank_data.json"
path_stage1 = "o1.json"
with open(path_stage1, 'w') as f:
    for graf in pytextrank.parse_doc(pytextrank.json_iter(path_stage0)):
        f.write("%s\n" % pytextrank.pretty_print(graf._asdict()))
        # to view output in this notebook
        print(pytextrank.pretty_print(graf))
