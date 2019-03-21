import os
from collections import defaultdict
import shutil


def group(video_id):
    path_to_seg = '../data/GEOL1330Fall18_Jinny/v' + str(video_id) + '/v' + str(video_id) + '_segments.csv'
    with open(path_to_seg, 'r') as f:
        raw = f.readlines()
    seg_data = defaultdict(list)
    for line in raw[1:]:
        l = line.split(',')
        seg_data[int(l[6])].append(l[7])
    for seg, frames in seg_data.items():
        destination = '../data/GEOL1330Fall18_Jinny/v' + str(video_id) + '/GFQG_data/seg' + str(seg) + '/frames'
        if not os.path.exists(destination):
            os.mkdir(destination)
        src = '../data/GEOL1330Fall18_Jinny/v' + str(video_id) + '/img_txt'
        for frame in frames:
            full_frame_name = os.path.join(src, frame)
            shutil.copy(full_frame_name, destination)


if __name__ == '__main__':
    video_index = [4588, 4608, 4609, 4618, 4623]
    for video_id in video_index:
        group(video_id)
