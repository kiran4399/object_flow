# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Set up paths for Fast R-CNN."""
import os.path as osp
import sys
import platform

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        print 'added {}'.format(path)

this_dir = osp.dirname(__file__)

# Use this when you want to compute object detection output, for example when
# using scripts experiments/test_pretrained_models.sh experiments/train_models.sh
if False:
  caffe_path = osp.join(this_dir, 'caffe-fast-rcnn', 'python')
  add_path(caffe_path)

# Use this when you are computing SDS output, for example when using scripts
# experiments/sds_test_pretrained_models.sh experiments/sds_train_models.sh  
if True:
  caffe_path = osp.join(this_dir, 'caffe-sds', 'python')
  add_path(caffe_path)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)

lib_path = osp.join(this_dir, 'python_utils')
add_path(lib_path)

root_path = osp.join(this_dir, '.')
add_path(root_path)
