import spacy
import json
import pprint
from collections import defaultdict
nlp = spacy.load('en_core_web_sm')
# to speed up process I need only lemmas of the word count task, so i disable other parts of pipeline
nlp_light = spacy.load('en', disable=['tagger', 'ner'])


def video_2book(video_id, path_to_book):
    folder = '../data/GEOL1330Fall18_Jinny/v' + str(video_id) + '/'
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
            if common_words != 0:
                seg_score_list.append(common_words)
                seg_id_list.append(id)
        # at this moment we will choose 3 max score, can adapt it later
        seg_scores.append([(x, y) for y, x in sorted(zip(seg_score_list, seg_id_list), reverse=True)][:4])
        # print(str(video_seg_id), seg_scores[-1])
    d = defaultdict(list)
    # trying to identify unique book segment for video segment with max score
    for i in range(len(seg_scores)-1, -1, -1):
        for j in seg_scores[i]:
                d[j[0]].append((j[1], i+1))
    dd = defaultdict(list)
    for k,v in d.items():
            d[k] = sorted(d[k], reverse=True)[0]
    for k,v in d.items():
        dd[v[1]].append((v[0], k))
    res = []
    for k, v in dd.items():
        j = {"video_seg": k, "book_seg": v[0][1]}
        res.append((k, json.dumps(j)))
    with open(folder + 'v'+ str(video_id) + '_2book.json', 'w') as f:
        for e in sorted(res):
            f.write(e[1] + '\n')


if __name__ == '__main__':
    video_2book(4588, 'tt_Earth_cleaned.json')
