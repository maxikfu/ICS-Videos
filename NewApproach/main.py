import spacy
import ocr2dic
import GFQG
import json
[(134, 5), (133, 5), (317, 4), (313, 4)]
[(134, 5), (133, 5), (317, 4), (313, 4)]
[(166, 11), (153, 10), (126, 10), (157, 9)]
[(129, 11), (126, 11), (166, 9), (159, 8)]
[(153, 7), (319, 6), (962, 4), (317, 4)]
[(157, 9), (153, 8), (962, 7), (305, 7)]
[(148, 13), (463, 10), (149, 10), (303, 9)]
[(157, 19), (153, 16), (159, 15), (305, 13)]
[(153, 16), (157, 15), (319, 13), (313, 12)]
[(153, 21), (135, 21), (159, 17), (975, 16)]
nlp = spacy.load('en_core_web_sm')
# to speed up process I need only lemmas of the word count task, so i disable other parts of pipeline
nlp_light = spacy.load('en', disable=['tagger', 'ner'])



if __name__ == '__main__':
    video_id = 4623
    folder = '../data/GEOL1330Fall18_Jinny/v' + str(video_id) + '/'
    # loading video  OCR
    try:
        with open(folder + 'v' + str(video_id) + '.json', 'r') as f:
            seg_list = f.readlines()
    except IOError:
        print("Error: File " + folder + 'v' + str(video_id) + '.json' + " does not appear to exist.")
        exit()
    # loading book
    with open('tt_Earth_cleaned.json', 'r') as f:
        book_seg = f.readlines()
    # each segment of the book comparing with words from video
    for video_seg in seg_list:
        video_words = set(json.loads(video_seg)['text'].split(','))
        seg_id_list = []
        seg_score_list = []
        for seg in book_seg:
            id = json.loads(seg)['id']
            text = nlp_light(json.loads(seg)['text'])
            book_words = set()
            for token in text:
                book_words.add(token.lemma_)
            score = len(book_words.intersection(video_words))
            if score != 0:
                seg_score_list.append(score)
                seg_id_list.append(id)
        # at this moment we will choose 3 max score, can adapt it later
        scores = [(x, y) for y, x in sorted(zip(seg_score_list, seg_id_list), reverse=True)][:4]
        print(scores)
