import pytextrank
import pprint
import json


def text_ranking(book_segment):
    """
    :param book_segment: book segment in json format
    :return: key sentences and key phrases
    """
    path_stage1 = "o1.json"
    path_stage2 = "key_ph.json"
    path_stage3 = "imp_sent.json"

    """Perform statistical parsing/tagging on a document in JSON format"""
    with open(path_stage1, 'w') as f:
        for graf in pytextrank.parse_doc([book_segment]):
            f.write("%s\n" % pytextrank.pretty_print(graf._asdict()))

    graph, ranks = pytextrank.text_rank(path_stage1)
    """Collect and normalize the key phrases from a parsed document"""

    key_phrases = list(pytextrank.normalize_key_phrases(path_stage1, ranks))
    with open(path_stage2, 'w') as f:
        for rl in key_phrases:
            f.write("%s\n" % pytextrank.pretty_print(rl._asdict()))

    kernel = pytextrank.rank_kernel(path_stage2)
    """Calculate a significance weight for each sentence, 
    using MinHash to approximate a Jaccard distance from key phrases determined by TextRank"""
    key_sentences = list(pytextrank.top_sentences(kernel, path_stage1))
    with open(path_stage3, 'w') as f:
        for s in key_sentences:
            f.write(pytextrank.pretty_print(s._asdict()))
            f.write("\n")
    return key_sentences, key_phrases


if __name__ == '__main__':
    with open('tt_Earth_cleaned.json', 'r') as f:
        stage0 = json.loads(f.readlines()[42])
    key_sent, key_phr = text_ranking(stage0)
    for s in key_sent:
        print(s)


