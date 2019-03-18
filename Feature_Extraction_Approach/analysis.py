import os
import json
from Feature_Extraction_Approach.GFQG import min_max_normalize


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
            i += 1
            for line in raw:
                dd = json.loads(line)
                dd['video_id'] = str(video_id)
                dd['seg_id'] = str(i)
                all_sent.append(dd)
        with open(root + 'relevance.csv', 'w') as f:
            for l in all_sent:
                row = [l['video_id'], l['seg_id'], l['relevant'], str(l['score']), str(l['common_words']), str(l['features'][0])]
                f.write(','.join(row) + '\n')


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
                all_sent.append(dd['text'])
                all_scores.append(dd['score'])
    min_max_normalize(all_scores)
    data = sorted([(x, y) for x, y in zip(all_scores, all_sent)], reverse=True)
    with open(root + 'all_sent_relevance_rated.json', 'w') as f:
        for l in data:
            d = {'rating': '', 'score': l[0], 'text': l[1]}
            f.write(json.dumps(d) + '\n')


if __name__ == '__main__':
    relevance_analysis()
