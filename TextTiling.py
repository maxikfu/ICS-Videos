import nltk

if __name__ == '__main__':
    # with open('data/v4557/Human_physiology.txt', 'r') as f:
    #     raw_text = f.read()
    # tt = nltk.TextTilingTokenizer(w=90)
    # tokens = tt.tokenize(raw_text)
    # with open('data/v4557/text_tiling_info.txt', 'w') as f:
    #     i = 1
    #     for token in tokens:
    #         f.write('\n_TT'+str(i)+'\n')
    #         f.write(token)
    #         i += 1
    # print(i)
    with open('data/v4557/text_tiling_info.txt', 'r') as f:
        segmented_text = f.readlines()
    text_tiling_dict = {}
    for line in segmented_text:
        if ('_TT') in line:  # new topic starts
            topic_segment = line.strip()
            text_tiling_dict[topic_segment] = ''
        elif line.strip():  # keep adding lines to the previous topic
            text_tiling_dict[topic_segment] += line
    print(text_tiling_dict['_TT200'])
