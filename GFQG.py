import utility
import collections as coll
import re
import spacy
import numpy as np
from spacy.lang.en import English
from spacy.tokens import Doc
import pprint
nlp = spacy.load('en')


def data_cleaning(raw_text):  # cleaning text
    # Returns - lis of Span objects sentences from Doc object
    # and chapter name
    print(raw_text)
    exit()
    raw_text = re.sub(r"[\n]", " ", raw_text)
    doc = nlp(raw_text)
    c = 1
    for s in doc.sents:
        print(c, str(s).strip())
        c += 1
    exit()
    raw_text = raw_text.split()
    raw_text = ' '.join(raw_text)
    text = re.sub(r"[\n]", " ", raw_text)
    text = re.sub(r"[0-9]", "", text)
    raw_text = re.sub(r"[\W\n]", " ", raw_text)
    raw_text = re.sub(r"[\n]", "", raw_text)
    doc = nlp(text.lower())
    freq_word = {}
    for t in doc:
        if t.is_stop or t.is_punct:
            pass
        else:
            if str(t).strip() in freq_word:
                freq_word[str(t).strip()] += 1
            else:
                freq_word[str(t).strip()] = 1
    s = [(k, freq_word[k]) for k in sorted(freq_word, key=freq_word.get, reverse=True)]
    for k, v in s:
        print(k,v)
    exit()
    chapter = False
    for sentence in doc.sents:
        if chapter:
            chapter_name = sentence
            break
        elif 'CHAPTER' in str(sentence):  # next sentence suppose to be name of the chapter
            chapter = True
            chapter_number = re.sub(r"[a-zA-Z]", "", str(sentence))
    doc = nlp(raw_text)
    tokenized_sentences = [s for s in doc.sents]
    return tokenized_sentences, chapter_name


def sentence_selection(data, chapter):  # returns selected sentences with biggest score
    # Here we will calculate all features needed for sentence selection
    # Parameters - data - list of sentences (Span objects of Doc type)
    # f1 - number of tokens similar in the title/ length of sentence (excluding punctuation marks)
    # f2 - does sentence containes any abbreviation
    # f3 - contain a word in its superlative degree
    # f4 - beginning with discorse connective TODO: figure out how to identify them
    # f5 - number of words in sentence
    # f6= number of nouns / length of the sentence
    # f7 - number of pronouns/ length of the sentence
    chapter = str(chapter).lower()
    chapter = chapter.split()
    counter = 0
    weights = [1, 0.1, 1, 1, 0.0001, 1, 1]
    score = []
    for s in data:
        print(s)
    #     features = []
    #     f3 = 0
    #     pos_tags = [token.tag_ for token in s]
    #     number_of_words = len([pos for pos in pos_tags if pos != 'PUNCT'])
    #     f1 = 1/(len([i for i in s if str(i).lower() in chapter])/len(s)+1)
    #     features.append(f1)
    #     f2 = len([i for i in s if i.text.isupper() and len(i.text) > 1 and i.pos_ == 'PROPN'])
    #     features.append(f2)
    #     if 'JJS' in pos_tags:
    #         f3 = 1
    #     features.append(f3)
    #     if any(discorse in str(s) for discorse in ['the following', 'The following', 'example', 'So', 'above', 'Figure', 'like this one']):
    #         f4 = 0
    #     else:
    #         f4 = 1
    #     features.append(f4)
    #     f5 = number_of_words
    #     features.append(f5)
    #     f6 = len([p for p in pos_tags if p in ['NN', 'NNS']])/number_of_words
    #     features.append(f6)
    #     f7 = len([p for p in pos_tags if p in ['NNP', 'NNPS']])/number_of_words
    #     features.append(f7)
    #     score.append(np.dot(weights, features))
    # z = [print(y, x) for y, x in sorted(zip(score, data))]
    return data


if __name__ == '__main__':
    # utility.pdf2text('data/IE_chapter17.pdf')
    # after converting pdf to txt we need to clean up data
    with open('data/IE_chapter17.txt', 'r') as f:
        book_text = f.read()
    sentences, c_name = data_cleaning(book_text)
    sentence_selection(sentences, c_name)