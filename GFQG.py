import utility
import collections as coll
import re
import spacy
import numpy as np
from spacy.lang.en import English
from spacy.tokens import Doc
import pprint
nlp = spacy.load('en')


# overwriting spaCy, making it not case sensitive
def is_stop(word):
    return nlp.vocab[word.lower()].is_stop


def data_pre_processing(raw_text):  # cleaning text
    # returns Dictionary of spans with topic as a key and count of the words in the document
    # identifying topic and related content of the topic
    doc = nlp(raw_text)
    document_content_dict = {}  # dict: key - topic name, value - list of Span objects
    word_count = {}
    for sentence in doc.sents:
        if len(sentence) > 1:
            for token in sentence:
                if not is_stop(token.text) and not token.is_punct and not token.is_space:
                    if token.text.strip().lower() in word_count:
                        word_count[token.text.strip().lower()] += 1
                    else:
                        word_count[token.text.strip().lower()] = 1
            if any(x in str(sentence) for x in ['Topic:', 'Subtopic:']):  # topic found
                topic = str(sentence).replace('Topic:', '')
                topic = topic.replace('Subtopic:', '').strip()
                topic = topic.replace('.', '').strip().lower()
                document_content_dict[topic] = []
            else:
                document_content_dict[topic].append(sentence)
    return document_content_dict, word_count


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
    all_the_topics = set(token.lemma_ for token in topics_doc if not is_stop(token.text) and not token.is_punct)
    weights = [1.5, 0.1, 0.2, 0.5, 0.001, 0.2, 0.1]
    result_sel_sent = {}
    for topic, sentences in data_dict.items():
        result_sel_sent[topic] = []
        # topic = 'named entity recognition'
        # sentences = data_dict['named entity recognition']
        for span in sentences:
            number_of_words = len(set(token.lemma_ for token in span if not is_stop(token.text) and not token.is_punct))
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
                if any(discourse in str(span).lower() for discourse in ['the following', 'example', 'so', 'above', 'figure', 'like this one', 'fig.', 'these', 'this', 'that', 'however', 'thus', 'although', 'since']):
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
        selection = set(score for score in result_sel_sent[topic] if score > 1.3)
        if selection:
            result_sentences[topic] = [data_dict[topic][result_sel_sent[topic].index(elem)] for elem in selection]
    return result_sentences, all_the_topics


def token_dep_height(tokens):  # height of the token in syntactic tree
    # input - list of tokens
    nodes_on_level = []
    depth = 1
    for token in tokens:
        nodes_on_level = nodes_on_level + [t for t in token.children]
    if nodes_on_level:
        depth += token_dep_height(nodes_on_level)
    return depth


def questions_formation(sentences, word_count, topic_words):
    # creating all possible key lists from selected sentences
    # need to identify noun_chunks
    # List of features:
    # f1 - number of occurrences in document
    # f2 - contains in the title
    # f3 - height in syntactic tree
    weights = [1, 100, 1]
    for span in sentences['information extraction']:
        all_noun_chunks = []
        print(span)
        # Step 1: saving all noun chinks
        for chunk in span.noun_chunks:
            all_noun_chunks.append(chunk)
        # Step 2: Selecting the most important noun chunk
        for chunk in all_noun_chunks:
            features = []
            f1 = 0
            f2 = 0
            f3 = 0
            for token in chunk:
                if token.text.strip().lower() in word_count:
                    f1 += word_count[token.text.strip().lower()]
                if token.lemma_.strip().lower() in topic_words:
                    f2 += 1
                f3 += token_dep_height([token]) - 1
            features.append(f1)
            features.append(f2)
            features.append(f3)
            score = np.dot(weights, features)
            print(score, chunk, features)



if __name__ == '__main__':
    # utility.pdf2text('data/IE_chapter17.pdf')
    # after converting pdf to txt we need to clean up data
    with open('data/IE_chapter17_cleaned.txt', 'r') as f:
        book_text = f.read()
    data, word_dict = data_pre_processing(book_text)
    selected_sent, topic_words = sentence_selection(data)
    questions_formation(selected_sent, word_dict, topic_words)
    # for i in range(0, len(data['named entity recognition'])):
    #     print(score['named entity recognition'][i], data['named entity recognition'][i])
