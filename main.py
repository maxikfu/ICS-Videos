import GFQG
import ocr2dic
import pprint
if __name__ == '__main__':
    dict_ocr = ocr2dic.ocr2dict('data/v4653/Modi_all_4653.csv', 'data/v4653/v4653_segments.csv')
    list_of_words = []
    pprint.pprint(dict_ocr)
    # for slide, clusters in dict_ocr.items():
    #     list_of_words = list_of_words + [clusters[s][0][0] for s in clusters]
    # GFQG.rawtext2question('data/history/IE_chapter17_cleaned.txt', list_of_words)
