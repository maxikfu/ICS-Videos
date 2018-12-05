import spacy
import numpy as np
from random import shuffle
import utility

nlp = spacy.load('en')  # make sure to use larger model!


def is_stop(word):
    """
    Overwriting SpaCy stop word, since it doesn't support lower/upper case words yet
    :param word: string
    :return: True if it is stop word, False otherwise
    """
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


def sentence_selection(data_dict, external_key_words):
    """
    Calculating features for each sentence and then calculating sentence score based on assigned weights to each feature
    Features description:
    f1 - number of tokens similar in the title/ length of sentence (excluding punctuation marks) I look for words in all
    titles
    f2 - does sentence contains any abbreviation
    f3 - contain a word in its superlative degree
    f4 - beginning with discourse connective TODO: figure out how to identify them better
    f5 - number of words in sentence (excluding stop words)
    f6= number of nouns / length of the sentence
    f7 - number of pronouns/ length of the sentence
    :param data_dict: key - topic, value - list of SpaCy.span (sentences) related to this topic
    :type data_dict: dictionary
    :param external_key_words: external words what may help for selecting good sentences for sentence generation
    :type external_key_words: list
    :return: dictionary (key - name of the topic, value - list of selected sentences) and list of lemmatized words
    from all topics and external help words
    """
    # lemmatizing and removing stop words from names of all topics
    all_the_topics = ' '.join([t for t in data_dict])
    all_the_topics = all_the_topics + ' ' + ' '.join(external_key_words)
    topics_doc = nlp(all_the_topics)
    all_the_topics = set(token.lemma_ for token in topics_doc if not is_stop(token.text) and not token.is_punct and not token.is_space)
    weights = [1.5, 0.1, 0.2, 0.5, 0.01, 0.2, 0.1]
    result_sel_sent = {}
    # with open('data/results/all_sentences.txt', "w"):
    #     pass
    for topic, sentences in data_dict.items():
        result_sel_sent[topic] = []
        # topic = 'named entity recognition'
        # sentences = data_dict['syntactic parsing']
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
                if any(discourse in str(span).lower() for discourse in ['because', 'then', 'here', 'Here’s',
                                                                        'Ultimately', 'chapter', 'finally', 'described',
                                                                        'the following', 'example', 'so', 'above',
                                                                        'figure', 'like this one', 'fig.', 'these',
                                                                        'this', 'that', 'however', 'thus', 'although',
                                                                        'since']):
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
                # f8 = token_dep_height([span.root])
                # features.append(f8)
                score = np.dot(weights, features)
                # with open('data/results/all_sentences.txt','a') as f:
                #     f.write(str(score)+' '+str(span)+'\n')
            result_sel_sent[topic].append(score)
        # exit()
        # z = [print(y, x) for y, x in sorted(zip(score, data))]

    # in this step we do selection based on the score. At this moment boundary set to 1.1
    result_sentences = {}
    for topic in data_dict:
        selection = set(score for score in result_sel_sent[topic] if score > 1.8)
        if selection:
            result_sentences[topic] = [data_dict[topic][result_sel_sent[topic].index(elem)] for elem in selection]
    return result_sentences, all_the_topics


def token_dep_height(tokens):
    """
    Calculates how height provided token in the syntactic tree
    :param tokens: collection of SpaCy objects (token)
    :type tokens: list
    :return: int level
    """
    nodes_on_level = []
    depth = 1
    for token in tokens:
        nodes_on_level = nodes_on_level + [t for t in token.children]
    if nodes_on_level:
        depth += token_dep_height(nodes_on_level)
    return depth


def distractor_selection(key_sentence, document, key_chunk):
    """
    Selecting distractors from all noun chunks from all sentences from document based on their score
    Features:
    f1 - context - measure of contextual similarity
    f2 - dice coefficient score between question sentence and sentence containing distractor
    f3 - difference in term frequencies of distractor and key
    :param key_chunk: for this chunk we are looking for distractors
    :param key_sentence: gap-fill question
    :param document: all sentences
    :type document: SpaCy object Doc
    :return: list containing three distractors for the question
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
    three_max_elem = np.array(sent_similarity_score).argsort()[-3:][::-1]  # returns indices of three max elements in

    # Second step we will look for most similar noun chunks in those sentences
    for sim_sent in similarity_sentence:
        for noun_chunk in sim_sent.noun_chunks:
            score = key_chunk.similarity(noun_chunk)
            if 0.7 > score > 0.5:
                chunk_similarity_score.append(score)
                similar_chunks.append(noun_chunk)
                # print(score, noun_chunk)
    three_max_elem = np.array(chunk_similarity_score).argsort()[-3:][::-1]
    return [similar_chunks[three_max_elem[0]], similar_chunks[three_max_elem[1]], similar_chunks[three_max_elem[2]]]


def questions_formation(sentences, word_count, topic_words):
    """
    Here we choosing noun chunk what will be replaces with blank space in the sentence ergo it will be correct answer
    to this question. Calculating score based on features:
    f1 - number of occurrences in document
    f2 - contains in the title
    f3 - height in syntactic tree
    :param sentences: key - topic, values - list of SpaCy.span sentences
    :type sentences: dictionary
    :param word_count: key - lemmatized word, value - number of times occurred in the document
    :type word_count: dictionary
    :param topic_words: list of lemmatized words occurring in all topics and externally provided lists
    :return: dictionary, key - noun chunk chosen to be a gap, value - sentences SpaCy.span object what will be a question
    """
    weights = [1, 100, 1]
    chunk_span_dict = {}
    # with open('data/results/chunk_selection.txt', "w"):
    #     pass
    for topic in sentences:
        for span in sentences[topic]:
            # with open('data/results/chunk_selection.txt', "a") as f:
            #     f.write(str(span)+'\n')
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
                with open('data/results/chunk_selection.txt', "a") as f:
                    f.write(str(np.dot(weights, features))+' '+str(features)+' '+str(chunk) + '\n')

            # At this moment we choose max score chunk only, even though we can choose couple chunks with score > 100
            gap_chunk_index = np.argmax(score)
            if score:
                # noinspection PyTypeChecker
                if all_noun_chunks[gap_chunk_index] not in chunk_span_dict:
                    chunk_span_dict[all_noun_chunks[gap_chunk_index]] = [span]
                else:
                    chunk_span_dict[all_noun_chunks[gap_chunk_index]].append(span)
    return chunk_span_dict


def rawtext2question(path_to_raw_text):
    """
    Main function what generates gap-fill questions from text book
    :param path_to_raw_text: self explanatory
    :return: at this moment nothing. Prints to stdout questions with multiple answers
    """
    with open(path_to_raw_text, 'r') as f:
        book_text = f.read()
    data, word_dict = data_pre_processing(book_text)
    with open('data/key_words.txt', 'r') as f:
        key_words = f.read()
    key_words = key_words.lower().split(',')
    selected_sent, topic_words = sentence_selection(data, key_words)
    questions = questions_formation(selected_sent, word_dict, topic_words)
    for key_chunk, value in questions.items():
        for q in value:
            distractor_list = distractor_selection(q, data, key_chunk)
            distractor_list.append(str(key_chunk))
            shuffle(distractor_list)
            # Printing question and multiple answers to this question:
            gap_question = str(q).replace(str(key_chunk), '______________')
            print('Question: ', gap_question)
            print('a) ', str(distractor_list[0]).lower())
            print('b) ', str(distractor_list[1]).lower())
            print('c) ', str(distractor_list[2]).lower())
            print('d) ', str(distractor_list[3]).lower())
            print('Answer: ', key_chunk)
            print('\n')


if __name__ == '__main__':
    # utility.pdf2text('data/syntactic_parsing.pdf')
    # after converting pdf to txt we need to clean up data
    rawtext2question('data/IE_chapter17_cleaned.txt')
