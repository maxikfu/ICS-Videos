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
ocr_ie_dict = utility.extract_sentences_from_ocr(ie_ocr_result_df)



# Parsing
ocr_parsing_dict = utility.clusterize_upgrade(parsing_ocr_result_df)
ocr_ie_dict = utility.clusterize_upgrade(ie_ocr_result_df)
# Algorithm accuracy
correct, total = utility.evaluation(ocr_ie_dict, perfect_gold_ie)
correct1, total1 = utility.evaluation(ocr_parsing_dict, perfect_gold_parsing)
print('Parsing: ',correct1, total1)
print('IE: ', correct, total)
print('Accuracy of algorithm: ', (correct+correct1)/(total+total1))

# # removing word length between points to bring words closer to each other for better clustering performance MAYBE
# x, y = utility.ocr_coordinates_pre_processing(ie_ocr_result_df)
# x_y = [[x1, y1] for x1, y1 in zip(x, y)]
# # estimating number of cluster with gap statistic
# k, linkage = utility.estimate_n_clusters(x_y)
# labels = utility.clustering(x, y, k, linkage)  # clustering for 2D data
# ie_ocr_result_df = utility.update_ocr_results(ie_ocr_result_df, labels)
# ocr_ie_dict = utility.extract_sentences_from_ocr(ie_ocr_result_df)


# Parsing slides
# x, y = utility.ocr_coordinates_pre_processing(parsing_ocr_result_df)
# x_y = [[x1, y1] for x1, y1 in zip(x, y)]
# # estimating number of cluster with gap statistic
# k, linkage = utility.estimate_n_clusters(x_y)
# labels = utility.clustering(x, y, k, linkage)  # clustering for 2D data
# parsing_ocr_result_df = utility.update_ocr_results(parsing_ocr_result_df, labels)
# ocr_parsing_dict = utility.extract_sentences_from_ocr(parsing_ocr_result_df)

# ax = plt.gca()  # get the axis
# ax.invert_yaxis()  # invert the axis
# plt.scatter(x, y, c=labels)
# plt.show()
# # plt.scatter(np.zeros(len(x)), y, c=labels_1)
# # plt.show()


# perfect_gold_parsing = {}
# perfect_gold_ie = {}
# perfect_ocr_ie = {}
# perfect_ocr_parsing ={}
# for p in perfect_parsing_slides:
#     perfect_gold_parsing[p] = parsing_gold_dict[p]
#     perfect_ocr_parsing[p] = ocr_parsing_dict['Slide'+str(p)+'.jpg']
# for i in perfect_IE_slides:
#     perfect_gold_ie[i] = ie_gold_dict[i]
#     perfect_ocr_ie[i] = ocr_ie_dict['Slide'+str(i)+'.jpg']





