import utility
import pprint


def ocr2dict(path_to_ocr_output):
    ocr_result_df = utility.load_ocr_output(path_to_ocr_output)
    ocr_dict, ocr_data_frame = utility.cluster_upgrade(ocr_result_df)
    return ocr_dict


if __name__ == '__main__':
    dict_ocr = ocr2dict('data/TestData/ocr_Inf_extraction.csv')
    pprint.pprint(dict_ocr)
