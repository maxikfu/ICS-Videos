import pandas as pd
from nltk.corpus import stopwords


def extract_sentences_from_ocr(data):
    """
    Putting words together from OCR output to create sentences
    Data Frame columns in the specific order: word,Fontsize,FontFamily,FontFaceStyle,Left,Top,Right,Bottom,
    RecognitionConfidence,Id,RegionId,LineId,imageFile
    :param data: data frame from csv file of the single slide
    :return:
    """
    file_dict = {}
    file_names = set(data['imageFile'])
    for file_name in file_names:
        rows = data.loc[data['imageFile'] == file_name]
        sentence = []
        region_id = None
        line_id = None
        file_dict[file_name] = {}
        for index, row in rows.iterrows():
            if region_id != row['RegionId']:  # checking if it is in the different slide
                if sentence:  # writing last sentence from previous file
                    file_dict[file_name][region_id].append(sentence)
                    sentence = []
                region_id = row['RegionId']
                file_dict[file_name][region_id] = []  # list of sentences
                line_id = row['LineId']
            sentence.append(row['Word'])
        if sentence:  # need ability to add last sentence
            file_dict[file_name][region_id].append(sentence)
    return file_dict


def segmentation(input_dict, path_to_segments):
    """
    Every slide assigned to specific segment in the video lecture.
    Returns new dictionary
    :param path_to_segments:
    :param input_dict:
    :return: output_dict dictionary: key - segment, value -
    dictionary {key - slide, value - dictionary {key -# sequence, value - list of words}}
    """
    segment_df = pd.read_csv(path_to_segments)
    segment_index = segment_df['belongs_to_index_no']
    segment_file = segment_df['filename']
    new_dic = {}
    for seg_id, filename in zip(segment_index, segment_file):
        if seg_id not in new_dic:
            new_dic[seg_id] = {filename: input_dict[filename]}
        else:
            new_dic[seg_id][filename] = input_dict[filename]
    return new_dic


def load_ocr_output(file_path):  # removing stopwords in this step
    data = pd.read_csv(file_path)
    # removing spaces in file names
    id = list(data.index.values)
    for i in id:
        data.at[i, 'imageFile'] = str(data.at[i, 'imageFile']).strip()
    stop_words = set(stopwords.words('english'))
    # data = data[~data.word.isin(stop_words)]
    return data


def conv_to_dict(data, path_to_segments):
    """
    Main function converts OCR output to dictionary for further question generation tasks
    :param path_to_segments:
    :param data: dataframe of original OCR output from csv file
    :return: dictionary {key-slide number, value-dict{key-cluster, value list of words}}
    """
    data_dict = {}
    file_names = set(data['imageFile'])
    for file_name in file_names:
        # print('working on file', file_name)
        rows = data.loc[data['imageFile'] == file_name]
    data_dict = segmentation(extract_sentences_from_ocr(data), path_to_segments)
    return data_dict, data


def ocr2dict(path_to_ocr_output, path_to_segments):
    """
    Organizes OCR output by segments and word sequences
    :param path_to_ocr_output:
    :param path_to_segments:
    :return: dictionary: key - segment, value -
    dictionary {key - slide, value - dictionary {key -# sequence, value - list of words}}
    """
    ocr_result_df = load_ocr_output(path_to_ocr_output)
    ocr_dict, ocr_data_frame = conv_to_dict(ocr_result_df, path_to_segments)
    return ocr_dict


if __name__ == '__main__':
    dict_ocr = ocr2dict('data/TestData/ocr_Inf_extraction.csv')

