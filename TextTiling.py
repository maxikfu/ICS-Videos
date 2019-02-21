import nltk
import pprint


def book_texttiling():
    """
    cleaning up book, deleting lines with number of tokens less then 5
    saving to the file since textbook big and takes long time to perform texttiling
    :param path_to_the_book:
    :return: file with text divided on segment
    """
    # with open(path_to_the_book, 'r') as f:
    #     orig_book_lines = f.readlines()
    # we need to clean this book up a little bit
    # if line contains only 1-4 tokens deleting that line
    # with open('data/v4557/Anatomy_Physiology_cleaned_1.txt', 'w') as f:
    #     for line in orig_book_lines:
    #         if line == '\n' or len(line.split()) > 4:
    #             f.write(line)
    with open('data/v4588/Earth_An_Introduction_to_Physical_Geology.txt', 'r') as f:
        raw_text = f.read()
    tt = nltk.TextTilingTokenizer(w=500)
    tokens = tt.tokenize(raw_text)
    with open('data/v4588/tt_Earth.txt', 'w') as f:
        i = 1
        for token in tokens:
            f.write('\n_TT'+str(i)+'\n')
            f.write(token)
            i += 1


if __name__ == '__main__':
    book_texttiling()

    # exit()
    # with open('data/v4557/tt_anatomy_physiology.txt', 'r') as f:
    #     segmented_text = f.readlines()
    # text_tiling_dict = {}
    # for line in segmented_text:
    #     if ('_TT') in line:  # new topic starts
    #         topic_segment = line.strip().replace('_TT', '')
    #         text_tiling_dict[int(topic_segment)] = ''
    #     elif line.strip():  # keep adding lines to the previous topic
    #         text_tiling_dict[int(topic_segment)] += line
    # print(text_tiling_dict[1])
