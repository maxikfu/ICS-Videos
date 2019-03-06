import nltk
import os
import json


def text_tiling(path, word_window):
    """
    Tokenize a document into topical sections using the TextTiling algorithm.
    This algorithm detects subtopic shifts based on the analysis of lexical
    co-occurrence patterns.
    :param word_window: Pseudosentence size
    :type word_window: int
    :param path:  path to the cleaned text book
    :return: json file each segment of the book in the single line with id
    """
    out_fname = 'tt_' + os.path.splitext(os.path.basename(path))[0]
    working_dir = os.path.dirname(os.path.abspath(path)) + '/'
    with open(path, 'r') as f:
        raw_text = f.read()
    tt = nltk.TextTilingTokenizer(w=word_window)
    tokens = tt.tokenize(raw_text)
    list_dic = []
    i = 1
    for token in tokens:
        dic = {"id": i, "text": token.strip()}
        i += 1
        list_dic.append(json.dumps(dic))
    with open(working_dir + out_fname + '.json', 'w') as f:
        for l in list_dic:
            f.write(l+'\n')


if __name__ == '__main__':
    text_tiling('Earth_cleaned.txt', 20)
