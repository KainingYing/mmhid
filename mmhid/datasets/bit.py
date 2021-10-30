from pathlib import Path
import os

import mmcv
import numpy as np
from PIL import Image

from .builder import HID_PIPELINES, HID_DATASETS, DATA_ROOT, FRAMES_ROOT, ANNO_ROOT
from .custom import CustomDataset


@HID_DATASETS.register_module()
class BIT(CustomDataset):
    """BIT dataset.
    The annotation format is shown as follows.
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

    ACTION_CLASSES = (
        'bned', 'box', 'handshake', 'hifive', 'hug', 'kick', 'pat', 'push',
        'be_bended', 'be_boxed', 'be_kicked', 'be_pated', 'be_pushed'
    )

    MAX_FRAME = 50

    def __init__(self,
                 ann_file,
                 pipeline,
                 train_frame=34,
                 img_prefix='',
                 test_mode=False,
                 action_classes=None,
                 interaction_classes=None):

        """
        Args:
            train_frame:
            ann_file:
            pipeline:
            img_prefix:
            test_mode:
            action_classes:
            interaction_classes:
        """
        self.ann_file = ann_file
        self.pipeline = pipeline
        self.img_prefix = img_prefix
        self.test_mode = test_mode

        self.data_infos = self.load_annotations(self.ann_file)
        assert 0 < train_frame < self.MAX_FRAME and isinstance(train_frame, int), 'train_frame must be an ' \
                                                                                  'integer in range (0, 50)'
        self.train_frame = train_frame
        self.train_seq, self.test_seq = self.get_train_test_seq()

    def load_annotations(self):
        """
        load annotations from annotation file. The annotation is formatted offline.
        Args:
            self:
            ann_file:

        Returns:

        """
        assert python
        return 0

    @staticmethod
    def convert2coco():
        """Only used to convert BIT"""
        frame_root = Path(DATA_ROOT[BIT.__name__]) / Path(FRAMES_ROOT[BIT.__name__])
        anno_root = Path(DATA_ROOT[BIT.__name__]) / Path(ANNO_ROOT[BIT.__name__])
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
                    img = Image.open(frame)
                    width, height = img.size
                    image = {'file_name': os.path.join(*frame.parts[-3:]), 'height': height, 'width': width,
                             'id': img_id}

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
        mmcv.dump(coco_format_anno, Path(DATA_ROOT[BIT.__name__]) / f"{BIT.__name__}_coco.json")
        print(f'\nConvert {len(images)} images into COCO format.')

    def get_train_test_seq(self):
        action_names = ['bned', 'box', 'handshake', 'hifive', 'hug', 'kick', 'pat', 'push']
        train_seq = ['{}_{:04d}'.format(i, j) for i in action_names for j in range(1, self.train_frame)]
        test_seq = ['{}_{:04d}'.format(i, j) for i in action_names for j in range(self.train_frame + 1,
                                                                                  self.MAX_FRAME + 1)]
        return train_seq, test_seq






