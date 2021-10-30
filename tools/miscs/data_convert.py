"""Convert BIT/highfive/ut120 datasets to specific format like follows.
    [
        {
            'filename': 'bend/bend_0001/0001.jpg',
            'width': 320,
            'height': 240,
            'ann': {
                'bboxes' : <np.ndarray> (n, 4) in (x1, y1, x2, y2) order.
                'action_group': {
                    interaction_labels: <np.ndarray> (k, )
                    action_labels: <np.ndarray> (k, n)
                }
            }

        }
    ]
"""
import os
import argparse
from pathlib import Path

import mmcv

from mmhid.datasets import BIT

DATA_ROOT = {
    'bit': './data/BIT',
    'UT': './data/ut120',
    'highfive': './data/highfive'
}

FRAMES_ROOT = {
    'bit': 'Bit-frames',
}

ANNO_ROOT = {
    'bit': 'BIT-anno/tidy_anno'
}

DATASET_CLASS = {
    'bit': BIT
}


def convert(dataset):
    """Only used to convert BIT"""
    frame_root = Path(DATA_ROOT[dataset]) / Path(FRAMES_ROOT[dataset])
    anno_root = Path(DATA_ROOT[dataset]) / Path(ANNO_ROOT[dataset])
    annotations = []

    prog_bar = mmcv.ProgressBar(sum(1 for _ in frame_root.iterdir()))
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
                    index = 0
                    line = line.strip().split()
                    if len(line) == 4:
                        interaction = 'no_action'
                        frame_id = int(line[1])
                        box_num = int(line[-1])
                        frame_annotation = {
                            'box_num': box_num,
                            'bboxes': [],
                            'actions': [],
                            'interaction_label': []
                        }
                    else:
                        action = line[-1]
                        if action != 'no_action':
                            interaction = action
                        xmin, ymin, xmax, ymax = map(lambda x: int(x), line[1:5])
                        width = xmax - xmin
                        height = ymax - ymin
                        frame_annotation['bboxes'].append([xmin, ymin, width, height])
                        frame_annotation['actions'].append(action)
                        index += 1
                    if index == box_num:
                        if interaction != 'no_action':
                            frame_annotation['interaction_label'].append(action)

                    temp_b['action'] = line[-1]
                    temp_a.append(temp_b)
            anno_list.append(temp_a)

            for idx, frame in enumerate(video.iterdir()):  # frame : 0001.jpg
                # now single frame
                img = mmcv.imread(frame)
                height, width = img.shape[:-1]
                # todo: modify the file_name
                image = {'file_name': os.path.join(*frame.parts[-3:]), 'height': height, 'width': width}
                ann = {}
                ann['bboxes'] = [] # todo
                ann['interaction_group'] = [] # todo

                images.append(image)
                annotation = anno_list[idx]
                for item in annotation:
                    item['image_id'] = img_id
                    item['id'] = anno_id
                    anno_id += 1
                annotations.extend(annotation)
                img_id += 1
        prog_bar.update()


    # mmcv.dump(coco_format_anno, './bit_coco.json')
    mmcv.dump(coco_format_anno, Path(DATA_ROOT[dataset]) / f"{dataset}_coco.json")


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
