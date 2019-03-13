from collections import OrderedDict
import json
import os
from Feature_Extraction_Approach import GFQG
from random import shuffle


if __name__ == '__main__':
    video_id = 4623
    video_directory = '../data/GEOL1330Fall18_Jinny/v' + str(video_id) + '/'
    os.chdir(video_directory)
    # loading file containing video segment to book segment link
    try:
        with open('v' + str(video_id) + '_2book.json', 'r') as f:
            raw = f.readlines()
    except IOError:
        print("Error: File with video " + str(video_id) + " 2 book links does not exist.")
        exit()
    video_book_link = OrderedDict()
    for line in raw:
        l_j = json.loads(line)
        video_book_link[int(l_j['video_seg'])] = int(l_j['book_seg'])

    # load OCR words for every video segment
    try:
        with open('v' + str(video_id) + '.json', 'r') as f:
            raw = f.readlines()
    except IOError:
        print("Error: File with video " + str(video_id) + " OCR does not exist.")
        exit()
    video_OCR = OrderedDict()
    for line in raw:
        l_j = json.loads(line)
        video_OCR[int(l_j["id"])] = set(l_j["text"].split(','))

    # load book
    try:
        with open('../tt_Earth_cleaned.json', 'r') as f:
            raw_book_segs = f.readlines()
    except IOError:
        print("Error: Text book is not found.")
        exit()

    for v_seg_id, book_seg_id in video_book_link.items():
        book_seg_json = json.loads(raw_book_segs[book_seg_id - 1])
        video_seg_text = video_OCR[v_seg_id]
        # SENTENCE SELECTION PART
        sentences = GFQG.sentence_selection(v_seg_id, video_seg_text, book_seg_json)
        # GAP SELECTION PART
        key_list = GFQG.key_list_formation(v_seg_id, sentences, video_seg_text)
        # DISTRACTOR SELECTION PART
        key_phrase = key_list[0]['key_list'][0][1]
        distractors = GFQG.distractor_selection(v_seg_id, key_phrase, key_list)
        # QUESTION FORMATION
        quest_sentence = sentences[0]['text']
        answers = [d[1].text.lower() for d in distractors[:3]]
        answers.append(key_phrase.text.lower())
        shuffle(answers)
        gap_question = str(quest_sentence.text).replace(str(key_phrase.text), '______________')
        subdir = 'GFQG_data/seg' + str(v_seg_id) + '/'
        final_stage = subdir + "final_stage.txt"
        result = 'Question: ' + gap_question + '\n' \
                 + 'a) ' + answers[0] + '\n' \
                 + 'b) ' + answers[1] + '\n' \
                 + 'c) ' + answers[2] + '\n' \
                 + 'd) ' + answers[3] + '\n' \
                 + 'Answer: ' + key_phrase.text + '\n' + '\n'
        with open(final_stage, 'w') as f:
            f.write(result)

