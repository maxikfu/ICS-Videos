import utility
import numpy as np
import matplotlib.pyplot as plt


parsing_data = utility.load_test_data('data/TestData/parsing1_pptx/parsing_1pptx.csv')
ie_data = utility.load_test_data('data/TestData/information_extraction/inform_extraction.csv')
ocr_result = utility.load_ocr_output('data/Slide1.csv')
x_data = ocr_result['Left'].tolist()
y_data = ocr_result['Top'].tolist()
x_y_combined = [[x, y] for x, y in zip(x_data, y_data)]
x, y = utility.ocr_coordinates_pre_processing(ocr_result)
x_y = [[x1, y1] for x1, y1 in zip(x, y)]
k, df = utility.gap_statistic(x_y)
labels_1 = utility.clustering(x, y, k, 'single')  # clustering for 2D data
labels_2 = utility.clustering(np.zeros(len(x)), y, 5, 'single')
print(k,df)
# print(labels_1)
# print(labels_2)
ax = plt.gca()  # get the axis
ax.invert_yaxis()  # invert the axis
plt.scatter(x, y, c=labels_1)
plt.show()
# plt.scatter(np.zeros(len(x)), y, c=labels_1)
# plt.show()
