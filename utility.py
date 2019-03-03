# useful functions
import pandas as pd
from nltk.corpus import stopwords
import codecs
import os, subprocess


def load_ocr_output(file_path):  # removing stopwords in this step
    data = pd.read_csv(file_path)
    # removing spaces in file names
    id = list(data.index.values)
    for i in id:
        data.at[i, 'imageFile'] = str(data.at[i, 'imageFile']).strip()
    stop_words = set(stopwords.words('english'))
    # data = data[~data.word.isin(stop_words)]
    return data


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


def segmentation(input_dict, path_to_segments):
    #TODO: remake for txt file BIOL_4557.txt when I will have time
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


def pdf2text(pdf_file_path):  # converts pdf file to txt file
    env = dict(os.environ)
    env['LC_ALL'] = 'en_US.UTF-8'
    out_fname = os.path.splitext(os.path.basename(pdf_file_path))[0]
    working_dir = os.path.dirname(os.path.abspath(pdf_file_path))
    cmd = ['pdftotext', pdf_file_path, os.path.join(working_dir, out_fname+'.txt')]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    out, err = p.communicate()
    with codecs.open(os.path.join(working_dir, out_fname+'.txt'), 'r', encoding='utf-8') as f_in:
        content = f_in.read()
    content.replace('\r', '').replace('\x0C', '')
    return content
