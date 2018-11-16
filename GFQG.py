import utility

import re
import spacy
from spacy.lang.en import English
from spacy.tokens import Doc
import pprint
nlp = spacy.load('en')


def data_cleaning(raw_text):  # cleaning text
    # sentences = sent_tokenize(text)
    # looking for the titles of subchapters and related to that subchapter content
    doc = nlp(raw_text)
    tokenizer = English().Defaults.create_tokenizer(nlp)
    sentences = [sent.string.strip() for sent in doc.sents]
    # deleting sentences ending with ,
    # sentences = [re.sub(r"[\n]", " ", s) for s in sentences if s[-1] != ',']
    sent_containing_chapter = [s[s.find('CHAPTER')+len('CHAPTER'):] for s in sentences if 'CHAPTER' in s]
    chapter_name = re.sub(r"[0-9]", "", sent_containing_chapter[0]).strip()
    chapter_number = re.sub(r"[a-zA-Z]", "", sent_containing_chapter[0]).strip()
    # sub_topics = [s for s in sentences if '17.' in s] # todo: figure out method to extract subtopics
    tokenized_sentences = []
    for doc in tokenizer.pipe(sentences, batch_size=50):
        tokenized_sentences.append(doc)
    # tokenized_sentences = [s for s in tokenized_sentences if len(s) > 3]
    return tokenized_sentences, chapter_name


def sentence_selection(data, chapter):  # returns selected sentences with biggest score
    # Here we will calculate all features needed for sentence selection
    # f1 - number of tokens similar in the title/ length of sentence (excluding punctuation marks)
    # f2 - does sentence containes any abbreviation
    # f3 - contain a word in its superlative degree
    chapter = chapter.lower()
    chapter = chapter.split()
    for s in data:

        # si = [re.sub(r"[^0-9a-zA-Z\s]", "", str(i)) for i in s if not i.is_punct]
        # f1 = len([i for i in si if str(i).lower() in chapter])/len(si)
        # f2 = len([i for i in si if str(i).isupper() and len(i) > 1])
        # f3 = len([token for token in s if token.tag_ == 'JJS'])
        print(s)
        print([token.pos_ for token in s])
    return data


if __name__ == '__main__':
    # utility.pdf2text('data/IE_chapter17.pdf')
    # after converting pdf to txt we need to clean up data
    with open('data/IE_chapter17.txt', 'r') as f:
        book_text = f.read()
    sentences, c_name = data_cleaning(book_text)
    # sentence_selection(sentences, c_name)