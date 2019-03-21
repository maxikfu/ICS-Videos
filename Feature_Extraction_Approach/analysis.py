import os
import json
from Feature_Extraction_Approach.GFQG import min_max_normalize
from collections import defaultdict


def relevance_analysis():
    video_index = [4588, 4608, 4609, 4618, 4623]
    main_directory = os.getcwd()
    all_sent = []
    root = '../data/GEOL1330Fall18_Jinny/'
    for video_id in video_index:
        print(video_id)
        video_directory = '../data/GEOL1330Fall18_Jinny/v' + str(video_id) + '/GFQG_data/'
        for subdir, dir, files in os.walk(video_directory, topdown=True):
            break
        i = 0
        for d in sorted(dir):
            with open(os.path.join(video_directory, d + '/stage1_imp_sent.json')) as f:
                raw = f.readlines()
            seg_id = int(''.join([x for x in d if x.isdigit()]))
            print(seg_id)
            for line in raw:
                dd = json.loads(line)
                dd['video_id'] = str(video_id)
                dd['seg_id'] = seg_id
                all_sent.append(dd)
        with open(root + 'relevance_new.json', 'w') as f:
            for l in all_sent:

                f.write(json.dumps(l) + '\n')


def sentence_analysis():
    video_index = [4588, 4608, 4609, 4618, 4623]
    main_directory = os.getcwd()
    all_sent = []
    all_scores = []
    root = '../data/GEOL1330Fall18_Jinny/'
    for video_id in video_index:
        print(video_id)
        video_directory = '../data/GEOL1330Fall18_Jinny/v' + str(video_id) + '/GFQG_data/'
        for subdir, dir, files in os.walk(video_directory, topdown=True):
            break
        for d in sorted(dir):
            with open(os.path.join(video_directory, d + '/stage1_imp_sent.json')) as f:
                raw = f.readlines()
            for line in raw:
                dd = json.loads(line)
                dd['video_id'] = video_id
                if dd['text'] not in all_sent:
                    all_sent.append(dd['text'])
                    all_scores.append(dd['score'])
    min_max_normalize(all_scores)
    data = set(sorted([(x, y) for x, y in zip(all_scores, all_sent)], reverse=True))
    with open(root + 'all_sent.json', 'w') as f:
        for l in data:
            if l[0] > 0:
                d = {'rating': '', 'text': l[1], 'score': round(l[0], 2)}
                f.write(json.dumps(d) + '\n')


def data_prep():
    with open('../data/GEOL1330Fall18_Jinny/relevance_new.json', 'r') as f:
        raw = f.readlines()
    data = {}
    csv_out = [['video_id', 'video_seg_id', 'sent_id', 'relevant', 'score', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'common words', 'text']]
    for line in raw:
        j = json.loads(line)
        csv_out.append([j['video_id'], j['seg_id'], j['id'], j['relevant'], j['score'], j['features'][0], j['features'][1], j['features'][2], j['features'][3], j['features'][4], j['features'][5], j['common_words'], j['text'].strip()])
    with open('../data/GEOL1330Fall18_Jinny/relevance_new_data.csv', 'w') as f:
        for l in csv_out:
            f.write('$'.join([str(x) for x in l[:-1]]) + '$' + l[-1].strip().replace('\n', '') + '\n')


if __name__ == '__main__':
    data_prep()

