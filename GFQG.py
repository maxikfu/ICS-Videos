import spacy
import numpy as np
import random
import string
from random import shuffle

nlp = spacy.load('en_core_web_sm')  # make sure to use larger model!


def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == '.':
            doc[token.i+1].is_sent_start = True
    return doc


# nlp.add_pipe(set_custom_boundaries, before='parser')



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
    :return: document_content  text}
    """
    document_content = []  # list of Span objects
    for sentence in raw_text.sents:
        if len(sentence) > 1:
            document_content.append(sentence)
    return document_content


def sentence_selection(data, important_words, al_sel):
    """
    Calculating features for each sentence S and then calculating sentence score based on assigned weights to each feature
    Features description:
    F1 - number of tokens common in video segment and S/ length(S) TODO: it is not working yet
    F2 - number of common words in video segment and S
    F3 - number S contains any abbreviation/ length(S)
    F4 - does S contains words in superlative degree (POS - ‘JJS’)
    F5 - does S beginning with a discourse connective (because, since, when, thus, however etc.) TODO: figure out how to identify them better
    F6 - number of nouns in S/ length(S)
    F7 - number of pronouns in S/ length(S)
    :param al_sel:
    :param data: list of SpaCy.span (sentences) related to this topic
    :type data: list
    :param important_words: words from video lecture segment
    :type important_words: set
    :return: dictionary (key - name of the topic, value - list of selected sentences) and list of words lemmas
    from video segment
    """
    # weights = [1.5, 0.1, 1, 1, 0.2, 0.2]
    weights = [2, 1, 0.5, 1, 1, 1, 1]
    # finding features
    sent_scores = []
    details = []
    good_sent = []
    for sentence in data:
        number_of_words = len(set(token.lemma_ for token in sentence if not is_stop(token.text) and not token.is_punct))
        score = 0
        # only for sentences more then 4 tokens
        if number_of_words > 8 \
                and sentence.text not in al_sel \
                and sentence[0].text[0].isupper() \
                and sentence[0].is_alpha \
                and sentence[-1].text == '.':
            features = []
            pos_tags = [token.tag_ for token in sentence]
            f1 = len(set([str(i.lemma_).lower() for i in sentence if str(i.lemma_).lower() in important_words])) / number_of_words
            features.append(f1)
            f2 = len(set([str(i.lemma_).lower() for i in sentence if str(i.lemma_).lower() in important_words]))
            features.append(f2)
            f3 = len([i for i in sentence if i.text.isupper() and len(i.text) > 1 and i.pos_ == 'PROPN'])/ number_of_words
            features.append(f3)
            f4 = 0
            if 'JJS' in pos_tags:
                f4 = 1
            features.append(f4)
            # TODO: this list need to be filled with more examples or figure out something easier
            if any(discourse in sentence.text.lower() for discourse in ['because', 'then', 'here', 'Here’s',
                                                                        'Ultimately', 'chapter', 'finally', 'described',
                                                                        'the following', 'example', 'so', 'above',
                                                                        'figure', 'like this one', 'fig.', 'these',
                                                                        'this', 'that', 'however', 'thus', 'although',
                                                                        'since']):


                # sentence[0].text.lower() in ['because', 'then', 'here', 'here’s',
                #                                                     'ultimately', 'chapter', 'finally', 'described',
                #                                                     'the following', 'example', 'so', 'above',
                #                                                     'figure', 'like this one', 'fig.', 'these',
                #                                                     'this', 'that', 'however', 'thus', 'although',
                #                                                     'since', 'it is'] and :
                f5 = -10
            else:
                f5 = 1
            features.append(f5)
            f6 = len([p for p in pos_tags if p in ['NN', 'NNS']]) / number_of_words
            features.append(f6)
            f7 = len([p for p in pos_tags if p in ['NNP', 'NNPS']]) / number_of_words
            features.append(f7)
            score = np.dot(weights, features)
            sent_scores.append(score)
            details.append(features)
            good_sent.append(sentence)
    # in this step we do selection based on the score. At this moment max score selected
    selection = [(y,x,z) for y, x, z in sorted(zip(sent_scores, good_sent, details), reverse=True)][:3]
    selection_out = [x for _, x in sorted(zip(sent_scores, good_sent), reverse=True)][:1]
    al_sel.add(selection_out[0].text)
    max_score = selection[0][0]
    # for i in selection:
    #     print('Overall score', i[0])
    #     print('Sentence', i[1])
    #     print('Features', i[2])
    #     print('Common words', [j for j in set([str(i.lemma_).lower() for i in i[1] if str(i.lemma_).lower() in important_words])])
    # pprint.pprint(al_sel)
    return selection_out, important_words, al_sel, max_score


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


def distractor_selection(key_sentence, key_chunk, full_book):
    """
    Selecting distractors from all noun chunks from all sentences from textbook segment based on their score
    Features:
    F1 - similarity score. Similarity is determined by comparing word vectors or "word embeddings", multi-dimensional
    meaning representations of a word.
    :param full_book:
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
    for k, text in full_book.items():
        # first we will look for similar sentences
        for span_sentence in text.sents:
            score = key_sentence.similarity(span_sentence)
            if score != 1 and score > 0.8:
                sent_similarity_score.append(score)
                similarity_sentence.append(span_sentence)
                similarity_sentence = [i for _, i in sorted(zip(sent_similarity_score, similarity_sentence), reverse=True)][:20]
                sent_similarity_score = sorted(sent_similarity_score, reverse=True)[:20]

    # Second step we will look for most similar noun chunks in those sentences
    dupl_noun_chunks = set()
    for sim_sent in similarity_sentence:
        for noun_chunk in sim_sent.noun_chunks:
            score = key_chunk.similarity(noun_chunk)
            if 0.9 > score > 0.4 and noun_chunk.text.strip().lower() not in dupl_noun_chunks:
                dupl_noun_chunks.add(noun_chunk.text.strip().lower())
                chunk_similarity_score.append(score)
                similar_chunks.append(noun_chunk)
                similar_chunks = [i for _, i in sorted(zip(chunk_similarity_score, similar_chunks), reverse=True)][:3]
                chunk_similarity_score = sorted(chunk_similarity_score, reverse=True)[:3]
                # print(score, noun_chunk)
    # print('Dupl noun', dupl_noun_chunks)
    # # TODO: in case we didnt find distr in range we need to make range bigger
    # if len(similar_chunks) < 3:
    #     for sim_sent in similarity_sentence:
    #         for noun_chunk in sim_sent.noun_chunks:
    #             score = key_chunk.similarity(noun_chunk)
    #             if 0.9 > score > 0.5:
    #                 chunk_similarity_score.append(score)
    #                 similar_chunks.append(noun_chunk)
    #                 similar_chunks = [i for _, i in sorted(zip(chunk_similarity_score, similar_chunks), reverse=True)][
    #                                  :3]
    #                 chunk_similarity_score = sorted(chunk_similarity_score, reverse=True)[:3]
    return similar_chunks


def questions_formation(sentences, word_count, topic_words):
    """
    Here we choosing noun chunk as a key what will be replaces with blank space in the sentence ergo
    it will be correct answer to this question. Calculating score based on features:
    F1 - number of occurrences of the key in the textbook segment.
    F2 - does video lecture segment contain key
    F3 - height of the key in the syntactic tree of the sentence.
    :param sentences: list of SpaCy.span sentences
    :type sentences: list
    :param word_count: key - word lemma, value - number of times occurred in the document
    :type word_count: dictionary
    :param topic_words: list of word lemmas occurring in video segment
    :return: dictionary, key - noun chunk chosen to be a key, value - sentences SpaCy.span object what will be a question
    """
    weights = [1, 1.5, 1]
    chunk_span_dict = {}
    details = []
    total_number_words = 0
    for k,v in word_count.items():
        total_number_words += v
    best_score = 0
    for span in sentences:
        # better question will be creating by deleting Proper Noun I think
        # because most of the time it is abbreviation
        abbrev = [t for t in span if t.tag_ == 'NNP' and t.is_alpha and t.is_upper]
        if len(abbrev) == 1 and '('+abbrev[0].text+')' not in span.text and len(abbrev[0]) > 2:
            chunk_span_dict[abbrev[0]] = [span]
        else:
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
                        f2 = 1
                    f3 += token_dep_height([token]) - 1
                if f1 > 0:
                    f1 = 1/(f1)
                features.append(f1)
                features.append(f2)
                if f3 > 0:
                    f3 = f3
                features.append(f3)
                score.append(np.dot(weights, features))
                details.append(features)
                # with open('data/results/chunk_selection.txt', "a") as f:
                #     f.write(str(np.dot(weights, features)) + ' ' + str(features) + ' ' + str(chunk) + '\n')

            # At this moment we choose max score chunk only, even though we can choose couple chunks with score > 100
            best_noun_ch = [x for _, x in sorted(zip(score, all_noun_chunks), reverse=True)][:1]
            best_noun_ch_debug = [(y,x,z) for y, x, z in sorted(zip(score, all_noun_chunks, details), reverse=True)][:3]
            best_score = best_noun_ch_debug[0][0]
            # print(span)
            # print([(i,word_count[i]) for i in set([str(i.lemma_).lower() for i in span if str(i.lemma_).lower() in topic_words])])
            # for c in best_noun_ch_debug:
            #     print('Score:', c[0], 'Chunk:', c[1], 'Feature:', c[2])
            chunk_span_dict[best_noun_ch[0]] = [span]
            # chunk_span_dict=0
    return chunk_span_dict, best_score


def rawtext2question(book_text, video_lecture_words, already_sel, word_dict, full_book, work_folder, video_seg):
    """
    Main function what generates gap-fill questions from text book
    :param work_folder:
    :param full_book:
    :param word_dict:
    :param already_sel:
    :param video_lecture_words:
    :param book_text: text from 3 segments of the book
    :return: at this moment nothing. Prints to stdout questions with multiple answers
    """
    data = book_pre_processing(book_text)
    selected_sent, topic_words, already_sel, sent_score = sentence_selection(data, video_lecture_words, already_sel)
    questions, key_score = questions_formation(selected_sent, word_dict, topic_words)
    open(work_folder + 'results.txt', 'w').close()
    for key_chunk, value in questions.items():
        q = value[0]
        if not key_chunk.text.isupper():
            distractor_list = distractor_selection(q, key_chunk, full_book)
            distractor_list.append(str(key_chunk).lower())
        else:
            distractor_list = ["".join(random.choices(string.ascii_uppercase, k=len(key_chunk))) for _ in range(3)]
            distractor_list.append(key_chunk)
        shuffle(distractor_list)
        # Printing question and multiple answers to this question:
        gap_question = str(q).replace(str(key_chunk), '______________')
        # print('Score:', score)
        # print('Question: ', gap_question)
        # print('a) ', str(distractor_list[0]))
        # print('b) ', str(distractor_list[1]))
        # print('c) ', str(distractor_list[2]))
        # print('d) ', str(distractor_list[3]))
        # print('Answer: ', key_chunk)
        threshold = 0
        with open(work_folder + 'results.txt', 'a') as out:
            if (sent_score + key_score) >= threshold:
                result = str(video_seg) + '\n' + 'Score:' + str(sent_score + key_score) + '\n' \
                         + 'Question: ' + gap_question + '\n' \
                         + 'a) ' + str(distractor_list[0]).lower() + '\n' \
                         + 'b) ' + str(distractor_list[1]).lower() + '\n' \
                         + 'c) ' + str(distractor_list[2]).lower() + '\n' \
                         + 'd) ' + str(distractor_list[3]).lower() + '\n' \
                         + 'Answer: ' + key_chunk.text + '\n' + '\n'
            else:
                result = str(video_seg) + ' No questions with the score more then' + str(threshold) + '\n \n'
            out.write(result)
    return already_sel


if __name__ == '__main__':
    # utility.pdf2text('data/syntactic_parsing.pdf')
    # after converting pdf to txt we need to clean up data
    # rawtext2question('data/IE_chapter17_cleaned.txt')
    pass
