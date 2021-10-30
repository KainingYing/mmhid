import os
import re
from pathlib import Path

from setuptools import find_packages, setup


def get_readme():
    with open('README.md', encoding='utf-8') as f:
        content = re.sub(r'## Model([\s\S]*)respectively.\n\n', '', f.read())
    return content


if __name__ == '__main__':
    setup(
        name='mmhid',
        author='kaining',
        author_email='kennying99@gmail.com',
        license='GPLv2',
        url='https://github.com/noobying/mmhid',
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Topic :: Utilities',
        ],
        python_requires='>=3.8',
        install_requires=['scipy>=1.6', 'torch>=1.10.0', 'mmcv-full>=1.3,<1.4', 'torchvision>=0.7.0', 'matplotlib',
                          'tensorboard', 'terminaltables', 'mmdet>=2.17.0', 'numpy', 'pillow'],
        packages=find_packages(include=['mmhid']))
