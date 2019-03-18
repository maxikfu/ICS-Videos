import spacy
import json
from Feature_Extraction_Approach.GFQG import is_stop


nlp = spacy.load('en_core_web_lg')


def distr_extraction(tt_book_data):
    potential_distractors = set()
    for book_seg in tt_book_data:
        text = json.loads(book_seg)['text']
        doc = nlp(text)
        for noun_phrase in doc.noun_chunks:
            if not is_stop(noun_phrase.lower_):
                potential_distractors.add(noun_phrase)
    return potential_distractors


if __name__ == '__main__':
    distr_extraction('../data/GEOL1330Fall18_Jinny/tt_Earth_cleaned.json')
