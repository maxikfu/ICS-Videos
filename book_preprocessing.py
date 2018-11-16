import utility

import re
import spacy
from spacy.lang.en import English
import pprint
nlp = spacy.load('en')


def data_cleaning(raw_text): # cleaning text from misspelled words such as (I NFORMATION E XTRACTION)
    # sentences = sent_tokenize(text)
    # looking for the titles of subchapters and related to that subchapter content
    doc = nlp(raw_text)
    tokenizer = English().Defaults.create_tokenizer(nlp)
    sentences = [sent.string.strip() for sent in doc.sents]
    # deleting sentences ending with ,
    sentences = [re.sub(r"[\n]", " ", s) for s in sentences if s[-1] != ',']
    sent_containing_chapter = [s[s.find('CHAPTER')+len('CHAPTER'):] for s in sentences if 'CHAPTER' in s]
    chapter_name = re.sub(r"[0-9]", "", sent_containing_chapter[0]).strip()
    chapter_number = re.sub(r"[a-zA-Z]", "", sent_containing_chapter[0]).strip()
    # sub_topics = [s for s in sentences if '17.' in s] # todo: figure out method to extract subtopics
    tokenized_sentences = []
    for doc in tokenizer.pipe(sentences, batch_size=50):
        tokenized_sentences.append(doc)
    tokenized_sentences = [s for s in tokenized_sentences if len(s) > 3]
    print(tokenized_sentences[0][0])

if __name__ == '__main__':
    # utility.pdf2text('data/IE_chapter17.pdf')
    # after converting pdf to txt we need to clean up data
    with open('data/IE_chapter17.txt', 'r') as f:
        book_text = f.read()
    data_cleaning(book_text)