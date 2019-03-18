import spacy
import os
import numpy as np
import json
from collections import defaultdict


nlp = spacy.load('en_core_web_lg')


def is_stop(word):
    """
    Overwriting SpaCy stop word, since it doesn't support lower/upper case words yet
    :param word: string
    :return: True if it is stop word, False otherwise
    """
    return nlp.vocab[word.lower()].is_stop


def distr_extraction(path_to_book):
    # load tt book
    try:
        with open(path_to_book, 'r') as f:
            raw_book_segs = f.readlines()
    except IOError:
        print("Error: Text book is not found.")
        exit()
    potential_distractors = set()
    for book_seg in raw_book_segs:
        print(json.loads(book_seg)['id'])
        text = json.loads(book_seg)['text']
        doc = nlp(text)
        for noun_phrase in doc.noun_chunks:
            if not is_stop(noun_phrase.lower_):
                potential_distractors.add(noun_phrase.lower_)
    return potential_distractors


def min_max_normalize(list_of_scores):
    """
    Min Max normalization of the scores between 0 and 1
    :param list_of_scores:
    :return:
    """
    min_score = min(list_of_scores)
    max_score = max(list_of_scores)
    for i in range(len(list_of_scores)):
        list_of_scores[i] = (list_of_scores[i] - min_score)/(max_score - min_score)


def token_dep_height(tokens):
    """
    Calculates how height provided token in the syntactic tree of the sentence.
    The height of the tree is the length of the path from the deepest node in the tree to the root.
    :param tokens: collection of SpaCy objects (token)
    :type tokens: list
    :return: int level
    """
    nodes_on_level = []
    level = 1
    for token in tokens:
        nodes_on_level = nodes_on_level + [t for t in token.children]
    if nodes_on_level:
        level += token_dep_height(nodes_on_level)
    return level


def sentence_selection(video_seg_id, video_seg_text, book_segment_json):
    """
        Calculating features for each sentence S and then calculating sentence score based on assigned weights to each feature
        Features description:
        F1 - number of tokens common in video segment and S/ length(S)
        F2 - number S contains any abbreviation/ length(S)
        F3 - does S contains words in superlative degree (POS - ‘JJS’)
        F4 - does S beginning with a discourse connective (because, since, when, thus, however etc.)
        F5 - number of nouns in S/ length(S)
        F6 - number of pronouns in S/ length(S)
        :return: list of all sentences with their scores
        """
    if not os.path.exists('GFQG_data'):
        os.mkdir('GFQG_data')
    if not os.path.exists('GFQG_data/seg' + str(video_seg_id)):
        os.mkdir('GFQG_data/seg' + str(video_seg_id))
    weights = [2, 1, 0.2, 1, 2, 0.5]
    subdir = 'GFQG_data/seg' + str(video_seg_id) + '/'
    path_stage1 = subdir + "stage1_imp_sent.json"
    text = book_segment_json['text']
    doc = nlp(text)
    sent_scores = []
    details = []
    good_sent = []
    for sent in doc.sents:
        number_of_words = len(set(token.lemma_ for token in sent if not is_stop(token.text) and not token.is_punct))
        if number_of_words == 0:
            pass
        else:
            # sentences with more then 4 tokens, starts with Uppercase word, ends with punctuation
            features = []
            pos_tags = [token.tag_ for token in sent]
            common_words = set([str(i.lemma_).lower() for i in sent if str(i.lemma_).lower() in video_seg_text])
            f1 = len(common_words) / number_of_words
            features.append(round(f1, 2))
            # f2 = len(set([str(i.lemma_).lower() for i in sentence if str(i.lemma_).lower() in important_words]))
            # features.append(round(f2, 2))
            f2 = len([i for i in sent if
                      i.is_upper and len(i.text) > 1 and i.is_alpha and i.pos_ == 'PROPN']) / number_of_words
            features.append(round(f2, 2))
            f3 = 0
            if 'JJS' in pos_tags:
                f3 = 1
            features.append(f3)
            if any(discourse in sent.text for discourse in ['Because', 'Here’s',
                                                                        'Ultimately', 'Chapter', 'Finally', 'As described',
                                                                        'The following', 'So', 'Above',
                                                                        'Figure', 'like this one', 'fig.', 'These',
                                                                        'This', 'That', 'Thus', 'Although',
                                                                        'Since', 'As a result', 'shown in']):
                f4 = 0
            else:
                f4 = 1
            features.append(f4)
            f5 = len([p for p in pos_tags if p in ['NN', 'NNS']]) / number_of_words
            features.append(round(f5, 2))
            f6 = len([p for p in pos_tags if p in ['NNP', 'NNPS']]) / number_of_words
            features.append(round(f6, 2))
            score = np.dot(weights, features)
            if number_of_words < 8 \
                    or not sent[0].text[0].isupper() \
                    or not sent[0].is_alpha \
                    or not sent.text.strip()[-1] == '.':
                score = score * 0
            sent_scores.append(score)
            features.append(len(common_words))
            details.append(features)
            good_sent.append(sent)
    # in this step we do selection based on the score. At this moment max score selected
    # min_max_normalize(sent_scores)
    selection = [(y, x, z) for y, x, z in sorted(zip(sent_scores, good_sent, details), reverse=True)]
    id = 0
    output = []
    with open(path_stage1, 'w') as f:
        for res in selection:
            id += 1
            dic = {"id": id, "score": round(res[0], 2), "relevant": "No", "text": res[1].text, "common_words": res[2][-1], "features": res[2][:-1]}
            dic_1 = {"id": id, "score": round(res[0], 2), "text": res[1], "relevant": "No"}
            if res[2][-1] >= 4 and res[2][0] >= 0.36:  # relevant criteria
                dic['relevant'] = 'Yes'
                dic_1['relevant'] = 'Yes'
            output.append(dic_1)
            f.write(json.dumps(dic) + '\n')
    return output


def key_list_formation(video_seg_id, stage1_results, video_seg_words):
    """
    Here we choosing noun chunk as a key what will be replaces with blank space in the sentence ergo
   it will be correct answer to this question. Calculating score based on features:
   F1 - number of occurrences of the key in the textbook segment.
   F2 - does video lecture segment contain key
   F3 - height of the key in the syntactic tree of the sentence.
    """
    subdir = 'GFQG_data/seg' + str(video_seg_id) + '/'
    path_stage2 = subdir + "stage2_key_list.json"
    output = []
    output_debug = []
    weights = [0.5, 1, 1]
    noun_ch_count = defaultdict(int)
    for dic in stage1_results:  # counting number of times this noun chunk occurs in text
        sent_id = dic['id']
        sent_span = dic['text']
        for noun_chunk in sent_span.noun_chunks:
            if not is_stop(noun_chunk.text.lower()):
                noun_ch_count[noun_chunk.text.lower()] += 1
    for dic in stage1_results:  # computing score for every noun chunk
        sent_id = dic['id']
        sent_span = dic['text']
        key_list = []
        for noun_chunk in sent_span.noun_chunks:
            if not is_stop(noun_chunk.text.lower()):  # not counting stop words as a noun chunk
                features = []
                score = 0
                details = []
                f1 = noun_ch_count[noun_chunk.text.lower()]
                f2 = 0
                f3 = 0
                tokens_in_nch = [t.lemma_.strip().lower() for t in noun_chunk if not is_stop(t.text.lower())]
                if any(x in video_seg_words for x in set(tokens_in_nch)):
                    f2 += 1
                f3 = 1/token_dep_height([noun_chunk.root])
                features.append(f1)
                features.append(f2)
                features.append(f3)
                score = round(np.dot(weights, features), 3)
                key_list.append((score, noun_chunk, features))
        out_dic = {'sent_id': sent_id, 'key_list': sorted(key_list, reverse=True)}
        out_dic_debug = {'sent_id': sent_id, 'key_list': sorted(key_list, reverse=True)}
        output_debug.append(out_dic_debug)
        output.append(out_dic)
    with open(path_stage2, 'w') as f:
        for o in output_debug:
            o['key_list'] = [(x[0], x[1].text.lower(), x[2]) for x in o['key_list']]
            f.write(json.dumps(o) + '\n')
    return output


def distractor_selection(video_seg_id, key_phrase, poten_distr):
    subdir = 'GFQG_data/seg' + str(video_seg_id) + '/'
    path_stage3 = subdir + "stage3_distractors.json"
    set_distractors = poten_distr
    output = []
    exists = set()
    for d in set_distractors:
        sim_score = 0
        sim_score += key_phrase.root.similarity(d.root)
        sim_score += key_phrase.similarity(d)
        if sim_score != 2 and d.root.lemma_ not in exists and key_phrase.root.lower_ != d.root.lower_:
            exists.add(d.root.lemma_)
            output.append((round(sim_score, 2), d))
    output.sort(reverse=True)
    with open(path_stage3, 'w') as f:
        for o in output:
            di = {'Score': o[0], 'Distractor': o[1].text.lower()}
            f.write(json.dumps(di) + '\n')
    return output


if __name__ == '__main__':
    pass
