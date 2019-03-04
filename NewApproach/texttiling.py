import nltk
import os
import json


def text_tiling(path):
    """
    text tiling book, for ease access and summary
    """
    out_fname = 'tt_' + os.path.splitext(os.path.basename(path))[0]
    working_dir = os.path.dirname(os.path.abspath(path)) + '/'
    with open(path, 'r') as f:
        raw_text = f.read()
    tt = nltk.TextTilingTokenizer(w=1000)
    tokens = tt.tokenize(raw_text)
    list_dic = []
    i = 1
    for token in tokens:
        dic = {"id": i, "text": token}
        i += 1
        list_dic.append(json.dumps(dic))
    with open(working_dir + out_fname + '.json', 'w') as f:
        for l in list_dic:
            f.write(l+'\n')


if __name__ == '__main__':
    text_tiling('Earth_cleaned.txt')
