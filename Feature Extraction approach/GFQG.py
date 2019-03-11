import spacy
import os
import numpy as np
import json

nlp = spacy.load('en_core_web_sm')


def is_stop(word):
    """
    Overwriting SpaCy stop word, since it doesn't support lower/upper case words yet
    :param word: string
    :return: True if it is stop word, False otherwise
    """
    return nlp.vocab[word.lower()].is_stop


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
    # if not os.path.exists('GFQG_data'):
    #     os.mkdir('GFQG_data')
    # if not os.path.exists('GFQG_data/seg' + str(video_seg_id)):
    #     os.mkdir('GFQG_data/seg' + str(video_seg_id))
    weights = [1.5, 1, 1, 1, 1, 1]
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
            f1 = len(set([str(i.lemma_).lower() for i in sent if
                          str(i.lemma_).lower() in video_seg_text])) / number_of_words
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
                                                                        'The following', 'For example', 'So', 'Above',
                                                                        'Figure', 'like this one', 'fig.', 'These',
                                                                        'This', 'That', 'However', 'Thus', 'Although',
                                                                        'Since', 'As a result']):
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
                    or not sent[-1].text == '.':
                score = score * -1
            sent_scores.append(score)
            details.append(features)
            good_sent.append(sent)
    # in this step we do selection based on the score. At this moment max score selected
    selection = [(y, x, z) for y, x, z in sorted(zip(sent_scores, good_sent, details), reverse=True)]
    id = 0
    with open(path_stage1, 'w') as f:
        for res in selection:
            id += 1
            dic = {"id": id, "score": round(res[0], 2), "text": res[1].text, "features": res[2]}
            f.write(json.dumps(dic) + '\n')



if __name__ == '__main__':
    pass
