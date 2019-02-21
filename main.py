import GFQG
import ocr2dic
import spacy
from spacy import displacy
import pprint
import re


nlp = spacy.load('en_core_web_sm')
# to speed up process I need only lemmas of the word count task, so i disable other parts of pipeline
nlp_light = spacy.load('en', disable=['parser', 'tagger', 'ner'])


def texttiling_file_read(path_to_segmented_book):
    # reading file containing book by segments
    with open(path_to_segmented_book, 'r') as f:
        segmented_text = f.readlines()
    text_tiling_dict = {}
    book_seg_number = None
    # each segment of the book separated by custom line. in my case _TT# of the line
    # below I convert it to the dictionary key - custom line, value - actual text of the segment
    # In the future instead of custom line develop Topic extraction algorithm
    for line in segmented_text:
        if '_TT' in line:  # new topic starts
            book_seg_number = line.strip().replace('_TT', '')
            text_tiling_dict[int(book_seg_number)] = ''
        elif line.strip():  # keep adding lines to the previous topic
            text_tiling_dict[int(book_seg_number)] += ' ' + line.strip()
    total_count_in_book = {}
    for k,v in text_tiling_dict.items():
        doc = nlp(v)
        for token in doc:
            if not GFQG.is_stop(token.text) and not token.is_punct and token.text not in ['\n', ' '] \
                    and token.text.isalpha():
                if token.lemma_.lower() in total_count_in_book:
                    total_count_in_book[token.lemma_.lower()] += 1
                else:
                    total_count_in_book[token.lemma_.lower()] = 1
        text_tiling_dict[k] = doc
    return text_tiling_dict, total_count_in_book


if __name__ == '__main__':
    # doc = nlp(u'Crystallization of a molten mass generates Figure 3.11 Geodes like this one form when silica dissolved in groundwater precipitates to form quartz crystals that grow within cavities in rocks.')
    # with open('visual.html','w') as f:
    #     f.write(displacy.render(doc,style='dep'))
    # print(doc[0].is_alpha)
    # # exit()
    dict_ocr = ocr2dic.ocr2dict('data/v4588/img_txt/Modi_all_4588.csv', 'data/v4588/v4588_segments.csv')
    # for segment, frames in dict_ocr.items():
    #     for frame, region_ids in frames.items():
    #         for region_id, words in region_ids.items():
    #             pass
    already_selected = set()
    book_dict, total_w_count = texttiling_file_read('data/v4588/tt_Earth.txt')
    for seg in dict_ocr:
        print(seg)
        segment = dict_ocr[seg]
        segment_text = []
        for frame, region_id in segment.items():
            for region, region_text in region_id.items():
                segment_text = segment_text + region_text[0]
        doc = nlp(' '.join(segment_text))
        video_words = set()
        for token in doc:
            if not GFQG.is_stop(token.text) and not token.is_punct and token.tag_ in ['NN', 'NNS', 'NNP', 'NNPS']:
                video_words.add(token.lemma_.lower())
        seg_number_list = []
        seg_score_list = []
        # each segment of the book comparing with words from video
        for seg, text in book_dict.items():
            book_words = set()
            for token in text:
                book_words.add(token.lemma_)
            score = len(book_words.intersection(video_words))
            if score != 0:
                seg_score_list.append(score)
                seg_number_list.append(seg)
        # at this moment we will choose 3 max score, can adapt it later
        scores = [(x, y) for y, x in sorted(zip(seg_score_list, seg_number_list), reverse=True)]
        max_score_seg = [scores[0][0], scores[1][0], scores[2][0]]
        segment_text = nlp(book_dict[max_score_seg[0]].text + book_dict[max_score_seg[1]].text + \
                    book_dict[max_score_seg[2]].text)

        # for slide, clusters in dict_ocr.items():
        #     list_of_words = list_of_words + [clusters[s][0][0] for s in clusters]
        already_selected = GFQG.rawtext2question(segment_text, video_words, already_selected, total_w_count, book_dict)
