import GFQG
import ocr2dic
if __name__ == '__main__':
    dict_ocr = ocr2dic.ocr2dict('data/TestData/ocr_Inf_extraction.csv', 'path')
    list_of_words = []
    for slide, clusters in dict_ocr.items():
        list_of_words = list_of_words + [clusters[s][0][0] for s in clusters]
    GFQG.rawtext2question('data/history/IE_chapter17_cleaned.txt', list_of_words)
