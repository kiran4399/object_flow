import _init_paths

import cv2
from IPython.display import clear_output
import numpy as np
import os, sys, cPickle
import matplotlib.pyplot as plt
from time import sleep
import argparse, pprint

import caffe
from lib.fast_rcnn.config import cfg
import lib.datasets.nyud2_voc
import vis_neurons as vis_neurons
import python_utils.sg_utils as sg_utils
import lib.fast_rcnn.test
import lib.utils.vis_utils as vu
import lib.sds.test_hypercolumns as sds_test
from lib.fast_rcnn.config import cfg

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--imdb_name', type=str)
    parser.add_argument('--sds_sp_dir', default=None, type=str)
    parser.add_argument('--sx', type=float)
    parser.add_argument('--sy', type=float)
    parser.add_argument('--count_f', type=float)
    parser.add_argument('--nx', type=int)
    parser.add_argument('--ny', type=int)
    parser.add_argument('--sds_output_dir', type=str)
    parser.add_argument('--ext', type=str)
    
    parser.add_argument('--sds_detection_file', default=None, type=str)

    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    args.sds_detection_file = args.sds_detection_file.format(args.imdb_name)
    if args.sds_output_dir is None:
      args.sds_output_dir = os.path.join(os.path.dirname(args.sds_detection_file), 'vis')
      sg_utils.mkdir_if_missing(args.sds_output_dir)
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    imdb = lib.datasets.factory.get_imdb(args.imdb_name)
    a = sg_utils.load_variables(args.sds_detection_file)
    
    gt_roidb = imdb.gt_roidb()
    count = np.zeros(imdb.num_classes)
    for gtr in gt_roidb:
        count = count + np.bincount(gtr['gt_classes'], minlength=imdb.num_classes)
        
    for cls_id in range(1,imdb.num_classes):
      tog = []
      for i in range(imdb.num_images):
        if a['nms_boxes'][cls_id-1][i] != []:
          sc = a['nms_boxes'][cls_id-1][i][:,4]*1
          sc = sc[:,np.newaxis]
          tog.append(np.hstack((i*np.ones_like(sc), np.arange(sc.shape[0])[:,np.newaxis], sc)))
      tog = np.vstack(tog)
      ind = np.argsort(tog[:,2])[::-1]
      tog = tog[ind,:]
      cnt = 0;

      f, axes = vu.subplot(plt, (args.ny, args.nx), (args.sy, args.sx)); axes = axes.ravel()[::-1].tolist()
      plt.subplots_adjust(hspace = 0.01, wspace = 0.01);

      ind_show = np.linspace(start=0, stop=count[cls_id]*args.count_f, num=args.nx*args.ny, dtype=int).ravel().tolist()
      for i in ind_show:
        im_id = int(tog[i,0]); box_id = int(tog[i,1])
        img = cv2.imread(imdb.image_path_at(im_id)[0])
        img = img[:,:,[2,1,0]]
        sp = cv2.imread(os.path.join(args.sds_sp_dir, imdb.image_index[im_id] + '.png'), cv2.CV_16U)
        sds = a['nms_sds'][cls_id-1][im_id]; det = a['nms_boxes'][cls_id-1][im_id]
        mask = sds[box_id,:][sp-1]; box = det[[box_id],:4]; sc = det[box_id,4]
        ax = axes.pop()
        im_with_mask = vu.show_mask_on_image(img.astype(np.float32)/255., mask.astype(np.float32), [1.0,0.0,1.0], f=0.4)
        ax.imshow((255.*im_with_mask).astype(np.uint8)); ax.set_axis_off();
        ax.set_title('{:s} det# {:d}: {:0.3f}'.format(imdb.classes[cls_id], i, sc), fontdict={'fontsize':20})
        vu.draw_bbox(plt, ax, box, fill=False, linewidth=2, edgecolor='b')
      
      if args.sds_output_dir is not None:
        file_name = os.path.join(args.sds_output_dir, imdb.classes[cls_id] + '.' + args.ext)
        print file_name
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0.)
