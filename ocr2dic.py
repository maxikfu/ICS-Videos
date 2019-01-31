import utility
import pprint


def ocr2dict(path_to_ocr_output, path_to_segments):
    """
    Organizes OCR output by segments and word sequences
    :param path_to_ocr_output:
    :param path_to_segments:
    :return: dictionary: key - segment, value -
    dictionary {key - slide, value - dictionary {key -# sequence, value - list of words}}
    """
    ocr_result_df = utility.load_ocr_output(path_to_ocr_output)
    ocr_dict, ocr_data_frame = utility.cluster_upgrade(ocr_result_df, path_to_segments)
    return ocr_dict


if __name__ == '__main__':
    dict_ocr = ocr2dict('data/TestData/ocr_Inf_extraction.csv')
    pprint.pprint(dict_ocr)
