import pytextrank
import os
import json


def text_ranking(video_seg_id, book_segment):
    """
    :param book_segment: book segment in json format
    :return: key sentences and key phrases
    """
    # os.chdir(video_path)
    # creating directory to store segments for clean structure
    if not os.path.exists('TextRank_data'):
        os.mkdir('TextRank_data')
    if not os.path.exists('TextRank_data/seg' + str(video_seg_id)):
        os.mkdir('TextRank_data/seg' + str(video_seg_id))
    subdir = 'TextRank_data/seg' + str(video_seg_id) + '/'
    path_stage1 = subdir + "stage1.json"
    path_stage2 = subdir + "stage2_key_ph.json"
    path_stage3 = subdir + "stage3_imp_sent.json"

    """Perform statistical parsing/tagging on a document in JSON format"""
    parse_book_seg = pytextrank.parse_doc([book_segment])
    with open(path_stage1, 'w') as f:
        for graf in parse_book_seg:
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
    key_sent, key_phr = text_ranking(1,2,'../data/GEOL1330Fall18_Jinny/v' + str(4609) + '/')



