import spacy
import numpy as np
from random import shuffle
import utility

nlp = spacy.load('en_core_web_sm')  # make sure to use larger model!


def is_stop(word):
    """
    Overwriting SpaCy stop word, since it doesn't support lower/upper case words yet
    :param word: string
    :return: True if it is stop word, False otherwise
    """
    return nlp.vocab[word.lower()].is_stop


def book_pre_processing(raw_text):
    """
    Counting word occupancies in the text. Feeding text to SpaCy default pipeline for tokenizing, tagging etc.
    :param raw_text: text
    :return: document_content_dict {key - nothing important at this moment, value- text}
    """
    doc = nlp(raw_text)
    document_content_dict = {'topic': []}  # dict: key - topic name, value - list of Span objects
    word_count = {}
    for sentence in doc.sents:
        if len(sentence) > 1:
            for token in sentence:
                if not is_stop(token.text) and not token.is_punct and not token.is_space:
                    if token.text.strip().lower() in word_count:
                        word_count[token.text.strip().lower()] += 1
                    else:
                        word_count[token.text.strip().lower()] = 1
            document_content_dict['topic'].append(sentence)
    return document_content_dict, word_count


def sentence_selection(data_dict, external_key_words):
    """
    Calculating features for each sentence S and then calculating sentence score based on assigned weights to each feature
    Features description:
    F1 - number of tokens common in video segment and S/ length(S) TODO: it is not working yet
    F2 - does S contains any abbreviation (1/0)
    F3 - does S contains words in superlative degree (POS - ‘JJS’)
    F4 - does S beginning with a discourse connective (because, since, when, thus, however etc.) TODO: figure out how to identify them better
    F5 - number of words in S (excluding stop words)
    F6 - number of nouns in S/ length(S)
    F7 - number of pronouns in S/ length(S)
    :param data_dict: key - topic, value - list of SpaCy.span (sentences) related to this topic
    :type data_dict: dictionary
    :param external_key_words: words from video lecture segment
    :type external_key_words: set
    :return: dictionary (key - name of the topic, value - list of selected sentences) and list of words lemmas
    from video segment
    """

    important_words = ' '.join(external_key_words)
    topics_doc = nlp(important_words)
    important_words = set(
        token.lemma_ for token in topics_doc if not is_stop(token.text) and not token.is_punct and not token.is_space)
    weights = [1.5, 0.1, 0.2, 0.5, 0.01, 0.2, 0.1]
    result_sel_sent = {}
    # finding features
    for topic, sentences in data_dict.items():
        result_sel_sent[topic] = []
        for span in sentences:
            number_of_words = len(set(token.lemma_ for token in span if not is_stop(token.text) and not token.is_punct))
            score = 0
            # only for sentences more then 4 tokens
            if number_of_words > 4:
                features = []
                pos_tags = [token.tag_ for token in span]
                f1 = len([i for i in span if str(i.lemma_).lower() in important_words]) / number_of_words
                features.append(f1)
                f2 = 1 / np.exp(len([i for i in span if i.text.isupper() and len(i.text) > 1 and i.pos_ == 'PROPN']))
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
                                                                        'since', 'it is']):
                    f4 = 0
                else:
                    f4 = 1
                features.append(f4)
                f5 = number_of_words
                features.append(f5)
                f6 = len([p for p in pos_tags if p in ['NN', 'NNS']]) / number_of_words
                features.append(f6)
                f7 = len([p for p in pos_tags if p in ['NNP', 'NNPS']]) / number_of_words
                features.append(f7)
                score = np.dot(weights, features)
            result_sel_sent[topic].append(score)
    # in this step we do selection based on the score. At this moment max score selected
    result_sentences = {}
    for topic in data_dict:
        # selection = set(score for score in result_sel_sent[topic] if score > 1.2)
        selection = {max(result_sel_sent[topic])}
        if selection:
            result_sentences[topic] = [data_dict[topic][result_sel_sent[topic].index(elem)] for elem in selection]
    return result_sentences, important_words


def token_dep_height(tokens):
    """
    Calculates how height provided token in the syntactic tree of the sentence.
    The height of the tree is the length of the path from the deepest node in the tree to the root.
    :param tokens: collection of SpaCy objects (token)
    :type tokens: list
    :return: int level
    """
    nodes_on_level = []
    level = 1
    for token in tokens:
        nodes_on_level = nodes_on_level + [t for t in token.children]
    if nodes_on_level:
        level += token_dep_height(nodes_on_level)
    return level


def distractor_selection(key_sentence, document, key_chunk):
    """
    Selecting distractors from all noun chunks from all sentences from textbook segment based on their score
    Features:
    F1 - similarity score. Similarity is determined by comparing word vectors or "word embeddings", multi-dimensional
    meaning representations of a word.
    :param key_chunk: for this chunk we are looking distractors
    :param key_sentence: gap-fill question
    :param document: all sentences in segment TODO: use all book not just segment
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
    Here we choosing noun chunk as a key what will be replaces with blank space in the sentence ergo
    it will be correct answer to this question. Calculating score based on features:
    F1 - number of occurrences of the key in the textbook segment.
    F2 - does video lecture segment contain key
    F3 - height of the key in the syntactic tree of the sentence.
    :param sentences: key - topic, values - list of SpaCy.span sentences
    :type sentences: dictionary
    :param word_count: key - word lemma, value - number of times occurred in the document
    :type word_count: dictionary
    :param topic_words: list of word lemmas occurring in video segment
    :return: dictionary, key - noun chunk chosen to be a key, value - sentences SpaCy.span object what will be a question
    """
    weights = [1, 100, 1]
    chunk_span_dict = {}
    for topic in sentences:
        for span in sentences[topic]:
            all_noun_chunks = []
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
                # with open('data/results/chunk_selection.txt', "a") as f:
                #     f.write(str(np.dot(weights, features)) + ' ' + str(features) + ' ' + str(chunk) + '\n')

            # At this moment we choose max score chunk only, even though we can choose couple chunks with score > 100
            gap_chunk_index = np.argmax(score)
            if score:
                # noinspection PyTypeChecker
                if all_noun_chunks[gap_chunk_index] not in chunk_span_dict:
                    chunk_span_dict[all_noun_chunks[gap_chunk_index]] = [span]
                else:
                    chunk_span_dict[all_noun_chunks[gap_chunk_index]].append(span)
    return chunk_span_dict


def rawtext2question(path_to_segmented_book, video_lecture_words):
    """
    Main function what generates gap-fill questions from text book
    :param video_lecture_words:
    :param path_to_segmented_book: self explanatory
    :return: at this moment nothing. Prints to stdout questions with multiple answers
    """
    # reading file containing book by segments
    with open(path_to_segmented_book, 'r') as f:
        segmented_text = f.readlines()
    text_tiling_dict = {}
    book_seg_number = None
    # each segment of the book separated by custom line. in my case _TT# of the line
    #TODO: topic extraction
    # below I convert it to the dictionary key - custom line, value - actual text of the segment
    # In the future instead of custom line develop Topic extraction algorithm
    for line in segmented_text:
        if '_TT' in line:  # new topic starts
            book_seg_number = line.strip().replace('_TT', '')
            text_tiling_dict[int(book_seg_number)] = ''
        elif line.strip():  # keep adding lines to the previous topic
            text_tiling_dict[int(book_seg_number)] += line
    seg_number_list = []
    seg_score_list = []
    # each segment of the book comparing with words from video
    # recording number of common words in both segments
    # to speed up process I need only lemmas of the word, so i disable other parts of pipeline
    nlp_local = spacy.load('en', disable=['parser', 'tagger', 'ner'])
    for seg, text in text_tiling_dict.items():
        doc_book = nlp_local(text)
        book_words = set()
        for token in doc_book:
            book_words.add(token.lemma_)
        score = len(book_words.intersection(video_lecture_words))
        if score != 0:
            seg_score_list.append(score)
            seg_number_list.append(seg)
    # at this moment we will choose 3 max score, can adapt it later
    scores = [(x, y) for y, x in sorted(zip(seg_score_list, seg_number_list), reverse=True)]
    max_score_seg = [scores[0][0], scores[1][0], scores[2][0]]
    book_text = text_tiling_dict[max_score_seg[0]] + text_tiling_dict[max_score_seg[1]] + \
                text_tiling_dict[max_score_seg[2]]
    data, word_dict = book_pre_processing(book_text)
    selected_sent, topic_words = sentence_selection(data, video_lecture_words)
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
    # rawtext2question('data/IE_chapter17_cleaned.txt')
    pass
