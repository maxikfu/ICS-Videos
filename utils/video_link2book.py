import spacy
import json
import os
from collections import defaultdict
nlp = spacy.load('en_core_web_sm')
# to speed up process I need only lemmas of the word count task, so i disable other parts of pipeline
nlp_light = spacy.load('en', disable=['tagger', 'ner'])


def video_2book(video_id, path_to_book):
    folder = os.path.dirname(os.path.abspath(path_to_book)) + '/v' + str(video_id) + '/'
    # loading video  OCR
    try:
        with open(folder + 'v' + str(video_id) + '.json', 'r') as f:
            seg_list = f.readlines()
    except IOError:
        print("Error: File " + folder + 'v' + str(video_id) + '.json' + " does not appear to exist.")
        exit()
    # loading book
    with open(path_to_book, 'r') as f:
        book_seg = f.readlines()
    seg_scores = []
    # each segment of the book comparing with words from video
    for video_seg in seg_list:
        video_seg_id = json.loads(video_seg)['id']
        video_words = set(json.loads(video_seg)['text'].split(','))
        seg_id_list = []
        seg_score_list = []
        for seg in book_seg:
            id = json.loads(seg)['id']
            text = nlp_light(json.loads(seg)['text'])
            book_words = set()
            for token in text:
                book_words.add(token.lemma_)
            common_words = len(book_words.intersection(video_words))
            # if common_words != 0:
            seg_score_list.append(common_words)
            seg_id_list.append(id)
        # at this moment we will choose 3 max score, can adapt it later
        seg_scores.append([(x, y, video_seg_id) for y, x in sorted(zip(seg_score_list, seg_id_list), reverse=True)][:4])
        # print(str(video_seg_id), seg_scores[-1])
    res = []
    for i in range(len(seg_scores)):
        for j in seg_scores[i]:
            res.append((j[1], j[0], j[2]))
    res.sort(reverse=True)
    d = {}
    used = set()
    for r in res:
        if r[1] not in used and r[2] not in d:
            d[r[2]] = (r[0], r[1])
            used.add(r[1])
    res = []
    for k, v in d.items():
        j = {"video_seg": k, "book_seg": v[1], "score": v[0]}
        res.append((k, json.dumps(j)))
    with open(folder + 'v' + str(video_id) + '_2book.json', 'w') as f:
        for e in sorted(res):
            f.write(e[1] + '\n')


if __name__ == '__main__':
    #videos_id = [4853, 4887, 4916, 4954, 4984, 4998, 5019, 5030, 5039, 5056, 5063, 5072, 5088]
    videos_id = [4588]
    for v in videos_id:
        print(v)
        video_2book(v, '../data/GEOL1330Fall18_Jinny/tt_Earth_cleaned.json')
