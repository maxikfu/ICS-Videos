import utility
import numpy as np
import matplotlib.pyplot as plt


# parsing_data = utility.load_test_data('data/TestData/parsing1_pptx/parsing_1pptx.csv')
actual_data = utility.load_test_data('data/gold_i_0088.csv')
ocr_result = utility.load_ocr_output('data/i_0088.csv')
ocr_data = utility.extract_sentences_from_ocr(ocr_result)

print(ocr_data)


# removing word length between points to bring words closer to each other for better clustering performance MAYBE
x, y = utility.ocr_coordinates_pre_processing(ocr_result)
x_y = [[x1, y1] for x1, y1 in zip(x, y)]
# estimating number of cluster with gap statistic
k, linkage = utility.estimate_n_clusters(x_y)
labels = utility.clustering(x, y, k, linkage)  # clustering for 2D data
new_ocr_results = utility.update_ocr_results(ocr_result, labels)
# ax = plt.gca()  # get the axis
# ax.invert_yaxis()  # invert the axis
# plt.scatter(x, y, c=labels_1)
# plt.show()
# # plt.scatter(np.zeros(len(x)), y, c=labels_1)
# # plt.show()
