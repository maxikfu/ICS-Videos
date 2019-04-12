from collections import OrderedDict
import json
import os
from Feature_Extraction_Approach import GFQG
from random import shuffle
from utils.noun_chunk_extraction import distr_extraction


if __name__ == '__main__':
    DEBUG = True
    if DEBUG:
        video_index = [4588, 4608, 4609, 4618, 4623]
    else:
        video_index = [4853, 4887, 4916, 4954, 4984, 4998, 5019, 5030, 5039, 5056, 5063, 5072, 5088]

    # video_index = [4608]
    main_directory = os.getcwd()
    all_questions = []
    # video_index = [4916]
    if DEBUG:
        tt_book = '../data/GEOL1330Fall18_Jinny/tt_Earth_cleaned.json'
    else:
        tt_book = '../data/Evaluation/tt_Microbiology_full_cleaned.json'
    # load book
    try:
        with open(tt_book, 'r') as f:
            raw_book_segs = f.readlines()
    except IOError:
        print("Error: Text book is not found.")
        exit()

    potential_distr = distr_extraction(raw_book_segs)
    total_number_of_segments = 0
    for video_id in video_index:
        print(video_id)
        if DEBUG:
            video_directory = '../data/GEOL1330Fall18_Jinny/v' + str(video_id) + '/'
        else:
            video_directory = '../data/Evaluation/v' + str(video_id) + '/'
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
        # iterating over video segments
        for v_seg_id, book_seg_id in video_book_link.items():
            total_number_of_segments += 1
            # print(v_seg_id)
            book_seg_json = json.loads(raw_book_segs[book_seg_id - 1])
            video_seg_text = video_OCR[v_seg_id]
            # SENTENCE SELECTION PART
            sentences = GFQG.sentence_selection(v_seg_id, video_seg_text, book_seg_json)

            # if there are no relevant sentences we will not proceed
            if [s for s in sentences if s["relevant"] == 'Yes' and s['score'] > 0]:
                # GAP SELECTION PART
                for s in sentences:
                    if s["relevant"] == 'Yes':
                        quest_sentence_id = s['id']
                        break
                key_list = GFQG.key_list_formation(v_seg_id, sentences, video_seg_text)
                # DISTRACTOR SELECTION PART
                key_phrase = key_list[quest_sentence_id - 1]['key_list'][0][1]
                distractors = GFQG.distractor_selection(v_seg_id, key_phrase, potential_distr)
                # QUESTION FORMATION
                answers = [d[1].text.lower() for d in distractors[:3]]
                answers.append(key_phrase.text.lower())
                shuffle(answers)
                quest_sentence = sentences[quest_sentence_id - 1]['text']
                gap_question = str(quest_sentence.text).replace(str(key_phrase.text), '______________')
                subdir = 'GFQG_data/seg' + str(v_seg_id) + '/'
                final_stage = subdir + "final_stage.txt"
                # print(answers)
                result = 'Question: ' + gap_question + '\n' \
                         + 'a) ' + answers[0] + '\n' \
                         + 'b) ' + answers[1] + '\n' \
                         + 'c) ' + answers[2] + '\n' \
                         + 'd) ' + answers[3] + '\n' \
                         + 'Answer: ' + key_phrase.text + '\n' + '\n'
                quest = {"video_id": video_id, "seg_id": v_seg_id, "text": result}
                all_questions.append(quest)
                with open(final_stage, 'w') as f:
                    f.write(result)
            else:
                subdir = 'GFQG_data/seg' + str(v_seg_id) + '/'
                final_stage = subdir + "final_stage.txt"
                with open(final_stage, 'w') as f:
                    f.write('No relevant to video sentences were found. :(')

        os.chdir(main_directory)
    if DEBUG:
        with open('../data/GEOL1330Fall18_Jinny/results.json', 'w') as f:
            for q in all_questions:
                f.write(json.dumps(q) + '\n')
                print('Video:', q['video_id'], 'Segment:', q['seg_id'])
                print(q['text'] + '\n')
    else:
        with open('../data/Evaluation/results.json', 'w') as f:
            for q in all_questions:
                f.write(json.dumps(q) + '\n')
                print('Video:', q['video_id'], 'Segment:', q['seg_id'])
                print(q['text'] + '\n')
    print(total_number_of_segments)
