import spacy
import numpy as np
from random import shuffle

nlp = spacy.load('en')  # make sure to use larger model!

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
                # TODO: this list need to be filled with more examples or figure out something easier
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
        selection = set(score for score in result_sel_sent[topic] if score > 1.5)
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


def distractor_selection(key, key_sentence, document, word_count):
    """
    Distractor is a concept semantically close to the key which, howerver, cannot serve as the right answer itself

    :features:
    f1 - context - measure of contextual similarity
    f2 - dice coefficient score between question sentence and sentence containing distractor
    f3 - difference in term frequencies of distractor and key

    :param key: noun chunk answer for the gap question
    :param document: document itself used to find distractors
    :param word_count: count of words
    :return: list of distractors at this moment we will set it to 3 distractors
    """
    sent_similarity_score = []
    chunk_similarity_score = []
    similar_chunks = []
    similarity_sentence = []
    for key, value in document.items():
        # first we will look for similar sentences
        for span_sentence in value:
            score = key_sentence.similarity(span_sentence)
            if score != 1 and score > 0.4:
                sent_similarity_score.append(score)
                similarity_sentence.append(span_sentence)
    three_max_elem = np.array(sent_similarity_score).argsort()[-3:][::-1]  # returns indices of three max elements in  list

    # Second step we will look for most similar noun chunks in those sentences
    for sim_sent in similarity_sentence:
        for noun_chunk in sim_sent.noun_chunks:
            score = key_chunk.similarity(noun_chunk)
            if 0.8 > score > 0.4:
                chunk_similarity_score.append(score)
                similar_chunks.append(noun_chunk)
                # print(score, noun_chunk)
    three_max_elem = np.array(chunk_similarity_score).argsort()[-3:][::-1]
    return [similar_chunks[three_max_elem[0]], similar_chunks[three_max_elem[1]], similar_chunks[three_max_elem[2]]]


def questions_formation(sentences, word_count, topic_words):
    # creating all possible key lists from selected sentences
    # need to identify noun_chunks
    # List of features:
    # f1 - number of occurrences in document
    # f2 - contains in the title
    # f3 - height in syntactic tree
    weights = [1, 100, 1]
    chunk_span_dict = {}
    for topic in sentences:
        for span in sentences[topic]:
            all_noun_chunks = []
            # print(span)
            # Step 1: saving all noun chinks
            for chunk in span.noun_chunks:
                all_noun_chunks.append(chunk)
            # Step 2: Selecting the most important noun chunk
            score = []
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
                score.append(np.dot(weights, features))

            # At this moment we choose max score chunk only, even though we can choose couple chunks with score > 100
            gap_chunk_index = np.argmax(score)
            if score:
                if all_noun_chunks[gap_chunk_index] not in chunk_span_dict:
                    chunk_span_dict[all_noun_chunks[gap_chunk_index]] = [span]
                else:
                    chunk_span_dict[all_noun_chunks[gap_chunk_index]].append(span)
    return chunk_span_dict


if __name__ == '__main__':
    # utility.pdf2text('data/IE_chapter17.pdf')
    # after converting pdf to txt we need to clean up data
    with open('data/IE_chapter17_cleaned.txt', 'r') as f:
        book_text = f.read()
    data, word_dict = data_pre_processing(book_text)
    selected_sent, topic_words = sentence_selection(data)
    questions = questions_formation(selected_sent, word_dict, topic_words)
    for key_chunk, value in questions.items():
        for q in value:
            distractor_list = distractor_selection(key_chunk, q, data, word_dict)
            distractor_list.append(str(key_chunk))
            shuffle(distractor_list)
            # Printing question and multiple answers to this question:
            gap_question = str(q).replace(str(key_chunk), '______________')
            print('Question: ', gap_question)
            print('a) ', distractor_list[0])
            print('b) ', distractor_list[1])
            print('c) ', distractor_list[2])
            print('d) ', distractor_list[3])
            print('Answer: ', key_chunk)
            print('\n')
    # for i in range(0, len(data['named entity recognition'])):
    #     print(score['named entity recognition'][i], data['named entity recognition'][i])
