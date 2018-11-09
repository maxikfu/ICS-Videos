import utility
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



parsing_gold_dict = utility.load_test_data('data/TestData/parsing1_pptx/precise_clusters_parsing_1.csv')
ie_gold_dict = utility.load_test_data('data/TestData/information_extraction/precise_clusters_IE.csv')

parsing_ocr_result_df = utility.load_ocr_output('data/TestData/ocr_parsing.csv')
ocr_parsing_dict = utility.extract_sentences_from_ocr(parsing_ocr_result_df)

ie_ocr_result_df = utility.load_ocr_output('data/TestData/ocr_Inf_extraction.csv')
ocr_ie_dict = utility.extract_sentences_from_ocr(ie_ocr_result_df)


perfect_parsing_slides = utility.perfect_ocr(parsing_gold_dict, ocr_parsing_dict)

perfect_IE_slides = utility.perfect_ocr(ie_gold_dict, ocr_ie_dict)
# dropping not perfect slides from GOLD dataset
perfect_gold_parsing = {}
perfect_gold_ie = {}
for p in perfect_parsing_slides:
    perfect_gold_parsing[p] = parsing_gold_dict[p]
for i in perfect_IE_slides:
    perfect_gold_ie[i] = ie_gold_dict[i]

# just ocr accuracy
# correct, total = utility.evaluation(ocr_ie_dict, perfect_gold_ie)
# correct1, total1 = utility.evaluation(ocr_parsing_dict, perfect_gold_parsing)
# print('Accuracy just OCR: ', (correct+correct1)/(total+total1))

# dropping not perfect ocr recognized slides
# for parsing
origin_ids = parsing_ocr_result_df.index.values
perfect_ids = parsing_ocr_result_df.loc[parsing_ocr_result_df['imageFile'].isin(perfect_parsing_slides)].index.values
drop_ids = set(origin_ids)-set(perfect_ids)
parsing_ocr_result_df = parsing_ocr_result_df.drop(drop_ids)
ocr_parsing_dict = utility.extract_sentences_from_ocr(parsing_ocr_result_df)
# for IE
origin_ids = ie_ocr_result_df.index.values
perfect_ids = ie_ocr_result_df.loc[ie_ocr_result_df['imageFile'].isin(perfect_IE_slides)].index.values
drop_ids = set(origin_ids)-set(perfect_ids)
ie_ocr_result_df = ie_ocr_result_df.drop(drop_ids)
#ie_ocr_result_df = utility.load_ocr_output('data/i_0088.csv')
ocr_ie_dict = utility.extract_sentences_from_ocr(ie_ocr_result_df)


# avg_acc = []
# for i in range(0, 10):
ocr_parsing_dict, pars_data_frame = utility.cluster_upgrade(parsing_ocr_result_df)
ocr_ie_dict, ie_data_frame = utility.cluster_upgrade(ie_ocr_result_df)

    # Algorithm accuracy
# correct, total = utility.evaluation(ocr_ie_dict, perfect_gold_ie)
# correct1, total1 = utility.evaluation(ocr_parsing_dict, perfect_gold_parsing)
    # print('Parsing: ', correct1, total1)
    # print('IE: ', correct, total)
# accuracy = (correct+correct1)/(total+total1)
#     avg_acc.append(accuracy)
# print('Accuracy of algorithm on step '+str(i)+': ', accuracy)
# print('Avg acc: ', np.mean(avg_acc))

# extracting word sequence dependencies
parsing_dependencies_dict = utility.extract_dependencies(pars_data_frame)
ie_dependencies_dict = utility.extract_dependencies(ie_data_frame)
print(parsing_dependencies_dict[56], ocr_parsing_dict[56])