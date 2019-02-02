import GFQG
import ocr2dic
import pprint
if __name__ == '__main__':
    dict_ocr = ocr2dic.ocr2dict('data/v4557/Modi_all_4557.csv', 'data/v4557/v4557_segments.csv')
    # list_of_words = []
    #some_dic, w_count = GFQG.data_pre_processing('data/v4557/Human_Physiology.txt')

    with open('data/v4557/text_tiling_info.txt', 'r') as f:
        segmented_text = f.readlines()
    text_tiling_dict = {}
    for line in segmented_text:
        if ('_TT') in line:  # new topic starts
            topic_segment = line.strip()
            text_tiling_dict[topic_segment] = ''
        elif line.strip():  # keep adding lines to the previous topic
            text_tiling_dict[topic_segment] += line
    video_segment_1 = dict_ocr[1]
    import_words = set()
    for frame in video_segment_1:
        for seq in video_segment_1[frame]:
            for word in video_segment_1[frame][seq][0]:
                if not GFQG.is_stop(word):
                    import_words.add(word)
    for key, value in text_tiling_dict.items():
        i = 0
        for imp in import_words:
            if imp in value:
                i += 1
        #print(key, i)
    print(text_tiling_dict['_TT62'])
    # for slide, clusters in dict_ocr.items():
    #     list_of_words = list_of_words + [clusters[s][0][0] for s in clusters]
    # GFQG.rawtext2question('data/history/IE_chapter17_cleaned.txt', list_of_words)
