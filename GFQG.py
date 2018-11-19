import utility
import collections as coll
import re
import spacy
import numpy as np
from spacy.lang.en import English
from spacy.tokens import Doc
import pprint
nlp = spacy.load('en')


def data_pre_processing(raw_text):  # cleaning text
    # returns Dictionary of spans with topic as a key
    # identifying topic and related content of the topic
    doc = nlp(raw_text)
    document_content_dict = {}  # dict: key - topic name, value - list of Span objects
    for sentence in doc.sents:
        if len(sentence) > 1:
            if any(x in str(sentence) for x in ['Topic:', 'Subtopic:']):  # topic found
                topic = str(sentence).replace('Topic:', '')
                topic = topic.replace('Subtopic:', '').strip()
                topic = topic.replace('.', '').strip().lower()
                document_content_dict[topic] = []
            else:
                document_content_dict[topic].append(sentence)
    return document_content_dict


def sentence_selection(data_dict):  # returns selected sentences depending on score
    # Here we will calculate all features needed for sentence selection.
    # For each chapter we will select sentences what are good for gap fill questions and most informative
    # Parameters - data_dict - spans of sentences with topic as key  (Span objects of Doc type)
    # f1 - number of tokens similar in the title/ length of sentence (excluding punctuation marks) I look for words in all titles
    # f2 - does sentence contains any abbreviation
    # f3 - contain a word in its superlative degree
    # f4 - beginning with discourse connective TODO: figure out how to identify them
    # f5 - number of words in sentence (excluding stop words)
    # f6= number of nouns / length of the sentence
    # f7 - number of pronouns/ length of the sentence

    # lemmatizing and removing stop words from names of all topics
    all_the_topics = ' '.join([t for t in data_dict])
    topics_doc = nlp(all_the_topics)
    all_the_topics = set(token.lemma_ for token in topics_doc if not token.is_stop and not token.is_punct)
    weights = [1.5, 0.1, 0.2, 0.5, 0.001, 0.2, 0.1]
    result_sel_sent = {}
    for topic, sentences in data_dict.items():
        result_sel_sent[topic] = []
        # topic = 'named entity recognition'
        # sentences = data_dict['named entity recognition']
        for span in sentences:
            number_of_words = len(set(token.lemma_ for token in span if not token.is_stop and not token.is_punct))
            score = 0
            if number_of_words > 4:
                features = []
                pos_tags = [token.tag_ for token in span]
                f1 = len([i for i in span if str(i.lemma_).lower() in all_the_topics])/number_of_words
                features.append(f1)
                f2 = 1/np.exp(len([i for i in span if i.text.isupper() and len(i.text) > 1 and i.pos_ == 'PROPN']))
                features.append(f2)
                f3 = 0
                if 'JJS' in pos_tags:
                    f3 = 1
                features.append(f3)
                if any(discourse in str(span) for discourse in ['the following', 'The following', 'example', 'So', 'above', 'Figure', 'like this one', 'Fig.', 'These', 'This', 'That']):
                    f4 = 0
                else:
                    f4 = 1
                features.append(f4)
                f5 = number_of_words
                features.append(f5)
                f6 = len([p for p in pos_tags if p in ['NN', 'NNS']])/number_of_words
                features.append(f6)
                f7 = len([p for p in pos_tags if p in ['NNP', 'NNPS']])/number_of_words
                features.append(f7)
                score = np.dot(weights, features)
                # print(score, features, span)
            result_sel_sent[topic].append(score)
        # z = [print(y, x) for y, x in sorted(zip(score, data))]

    # in this step we do selection based on the score. At this moment boundary set to 1.1
    result_sentences = {}
    for topic in data_dict:
        selection = set(score for score in result_sel_sent[topic] if score > 1.2)
        if selection:
            result_sentences[topic] = [data_dict[topic][result_sel_sent[topic].index(elem)] for elem in selection]
    return result_sentences


if __name__ == '__main__':
    # utility.pdf2text('data/IE_chapter17.pdf')
    # after converting pdf to txt we need to clean up data
    with open('data/IE_chapter17_cleaned.txt', 'r') as f:
        book_text = f.read()
    data = data_pre_processing(book_text)
    score = sentence_selection(data)
    # for i in range(0, len(data['named entity recognition'])):
    #     print(score['named entity recognition'][i], data['named entity recognition'][i])
