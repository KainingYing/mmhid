"""Convert BIT/highfive/ut120 datasets to COCO format.
"""
import os
import argparse
from pathlib import Path

import mmcv

DATA_ROOT = {
    'BIT': './data/BIT',
    'UT': './data/ut120',
    'highfive': './data/highfive'
}

FRAMES_ROOT = {
    'BIT': 'Bit-frames',
}

ANNO_ROOT = {
    'BIT': 'BIT-anno/tidy_anno'
}


def convert2coco(dataset):
    """Only used to convert BIT"""
    frame_root = Path(DATA_ROOT[dataset]) / Path(FRAMES_ROOT[dataset])
    anno_root = Path(DATA_ROOT[dataset]) / Path(ANNO_ROOT[dataset])
    coco_format_anno = dict()
    images, annotations, categories = [], [], []
    img_id, anno_id = 0, 0

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
                    line = line.strip().split()
                    if i == 0:
                        continue
                    if i != 0 and 'frame' in line[0]:
                        anno_list.append(temp_a)
                        temp_a = []
                        continue
                    xmin, ymin, xmax, ymax = map(lambda x: int(x), line[1:5])
                    width = xmax - xmin
                    height = ymax - ymin
                    temp_b['area'] = width * height
                    temp_b['iscrowd'] = 0
                    temp_b['bbox'] = [xmin, ymin, width, height]
                    temp_b['category_id'] = 0
                    temp_a.append(temp_b)
            anno_list.append(temp_a)

            for idx, frame in enumerate(video.iterdir()):  # frame : 0001.jpg
                # now single frame
                img = mmcv.imread(frame)
                height, width = img.shape[:-1]
                # todo: modify the file_name
                image = {'file_name': os.path.join(*frame.parts[-3:]), 'height': height, 'width': width, 'id': img_id}

                images.append(image)
                annotation = anno_list[idx]
                for item in annotation:
                    item['image_id'] = img_id
                    item['id'] = anno_id
                    anno_id += 1
                annotations.extend(annotation)
                img_id += 1
        prog_bar.update()

    categories.append({'id': 0, 'name': 'person'})
    coco_format_anno['images'] = images
    coco_format_anno['annotations'] = annotations
    coco_format_anno['categories'] = categories

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
