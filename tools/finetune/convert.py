"""Convert BIT/highfive/ut120 datasets to COCO format.
"""
import warnings
import argparse
from pathlib import Path

import mmcv


DATA_ROOT = {
    'bit': './data/BIT',
    'UT': './data/ut120',
    'highfive': './data/highfive'
}

FRAMES_ROOT = {
    'bit': 'BIT-frames',
}

ANNO_ROOT = {
    'bit': 'Bit-anno/tidy_anno'
}


def convert2coco(dataset):
    """Only used to convert BIT"""
    frame_root = Path(DATA_ROOT[dataset]) / Path(FRAMES_ROOT[dataset])
    anno_root = Path(DATA_ROOT[dataset]) / Path(ANNO_ROOT[dataset])
    coco_format_anno = dict()
    images, annotations, categories = [], [], []
    img_id, anno_id = 0, 0
    # step 1:
    for cate in frame_root.iterdir():  # cate:bend
        # now in bend Folder
        for video in cate.iterdir():  # video : bend_0001  anno: bend_0001.txt
            # now in video folder
            # format the anno into a structure list
            anno = (anno_root / cate.name / video.name).with_suffix('.txt')
            anno_list = []
            temp_a = []
            with open(anno, 'r') as f:
                for i, line in enumerate(f.readlines()):
                    temp_b = {}
                    # if i == 0:
                    #     line = line.strip().split(' ')
                    #     num_frames = int(line[-1])
                    #     assert num_frames > 0, 'number of frames must be positive.'
                    #     continue
                    line = line.strip().split()
                    if i != 0 and line[0] == 'frame':
                        anno_list.append(temp_a)
                        temp_a = []
                        continue
                    xmin, ymin, xmax, ymax = map(lambda x:int(x), line[1:5])
                    temp_b['area'] = (xmax - xmin) * (ymax - ymin)
                    temp_b['iscrowd'] = 0
                    temp_b['bbox'] = [xmin, ymin, xmax, ymax]
                    temp_b['category_id'] = 0
                    temp_a.append(temp_b)
                    if i == len(f.readlines()) - 1:
                        anno_list.append(temp_a)

            for frame in video.iterdir():  # frame : 0001.jpg
                # now single frame
                img = mmcv.load(frame)
                width, height = img.shape()[:-1]
                image = {'file_name': frame.name, 'height': height, 'width': width, 'id': img_id}  # todo: modify the file_name
                img_id += 1









def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert datasets into MS-COCO format"
    )
    parser.add_argument('--dataset_type', default='bit', choices=['bit', 'ut', 'highfive'], nargs='+',
                        help='Dataset type')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.dataset_type.lower() in ['bit', 'ut', 'highfive'], 'The dataset you want ' \
                                                                   'to convert does not belong ' \
                                                                   'to ["bit", "ut", "highfive"]'
    if not isinstance(args.dataset_type, (list, tuple)):
        args.dataset_type = [args.dataset_type]

    for dataset in args.dataset_type:
        convert2coco(dataset)


if __name__ == '__main__':
    main()
