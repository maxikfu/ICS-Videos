import GFQG
import ocr2dic
import spacy

nlp = spacy.load('en_core_web_sm')


if __name__ == '__main__':

    dict_ocr = ocr2dic.ocr2dict('data/v4557/Modi_all_4557.csv', 'data/v4557/v4557_segments.csv')
    # for segment, frames in dict_ocr.items():
    #     for frame, region_ids in frames.items():
    #         for region_id, words in region_ids.items():
    #             pass
    already_selected = set()
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
                video_words.add(token.lemma_)
        # for slide, clusters in dict_ocr.items():
        #     list_of_words = list_of_words + [clusters[s][0][0] for s in clusters]
        already_selected = GFQG.rawtext2question('data/v4557/tt_anatomy_physiology_1.txt', video_words, already_selected)
