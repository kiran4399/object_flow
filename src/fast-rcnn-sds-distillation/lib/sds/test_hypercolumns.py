import numpy as np
import sys
import _init_paths
import caffe
import cv2, os, time, math, evaluation
import scipy.misc as scipymisc
from sds.prepare_blobs import get_blobs
import superpixel_representation as sprep
import my_accumarray as accum
from scipy.io import savemat, loadmat

def get_hypercolumn_prediction(net, imgs, boxes, categids, img_blob_names=['image'], output_blob_name='loss'):
  boxes = boxes.copy()
  #clip boxes to image
  boxes = clip_boxes(boxes-1, imgs[0].shape) 
  
  im_new = [[] for _ in range(len(imgs))]
  for i in range(len(imgs)):
    im_new[i], sppboxes, normalizedboxes, categids = get_blobs(imgs[i], boxes, categids)
  
  # make blobs
  blobs = {};
  for i in range(len(imgs)):
    blobs[img_blob_names[i]] = im_new[i]
  blobs['normalizedboxes'] = normalizedboxes
  blobs['sppboxes'] = sppboxes
  blobs['categids'] = categids

  #reshape the network
  for blob_name in blobs.keys():
    net.blobs[blob_name].reshape(*(blobs[blob_name].shape))
  
  #forward
  output_blobs = net.forward(**blobs)
  
  output = output_blobs[output_blob_name]
  return output

def clip_boxes(boxes, im_shape):
  clipped_boxes = boxes.copy()
  clipped_boxes[:,[0,2]] = np.maximum(0, np.minimum(im_shape[1]-1, clipped_boxes[:,[0,2]]))
  clipped_boxes[:,[1,3]] = np.maximum(0, np.minimum(im_shape[0]-1, clipped_boxes[:,[1,3]]))
  return clipped_boxes


def paste_output(output, boxes, im_shape, target_output_size = [50, 50]):
  pasted_output =np.zeros((output.shape[0], output.shape[1], im_shape[0], im_shape[1]))
  #clip and round
  new_boxes = np.hstack((np.ceil(boxes[:,0:2]),np.ceil(boxes[:,2:4])))
  new_boxes = clip_boxes(new_boxes, im_shape)
  xmin = new_boxes[:,0]
  ymin = new_boxes[:,1]
  w = new_boxes[:,2]-xmin+1
  h = new_boxes[:,3]-ymin+1
  for item in range(output.shape[0]):
    xmin_ = xmin[item]
    ymin_ = ymin[item]
    w_ = w[item]
    h_ = h[item]
    y_all = np.arange(ymin_, ymin_+h_)
    Y_all = (y_all-ymin_)*target_output_size[0]/h_
    Y_all = np.maximum(0, np.minimum(target_output_size[0]-1, np.floor(Y_all)))

    x_all = np.arange(xmin_, xmin_+w_)
    X_all = (x_all-xmin_)*target_output_size[1]/w_
    X_all = np.maximum(0, np.minimum(target_output_size[1]-1, np.floor(X_all)))


    

    for channel in range(output.shape[1]):
      pasted_output_this = pasted_output[item,channel]
      output_this = output[item, channel]
      for i in range(len(Y_all)):
        pasted_output_this_row = pasted_output_this[y_all[i]]
        output_this_row = output_this[Y_all[i]]
        for j in range(len(X_all)):
          #X = (x-xmin_)*target_output_size[1]/w_
          #X = math.floor(X)
          #X = max(0, min(target_output_size[1]-1,X))
          pasted_output_this_row[x_all[j]] = output_this_row[X_all[j]]

  return pasted_output
   
def paste_output_sp(output, boxes, im_shape, sp, target_output_size = [50, 50]):
  pasted_output =np.zeros((output.shape[0], output.shape[1], np.max(sp)))
  #clip and round
  new_boxes = np.hstack((np.ceil(boxes[:,0:2]),np.ceil(boxes[:,2:4])))
  new_boxes = clip_boxes(new_boxes, im_shape)
  xmin = new_boxes[:,0]
  ymin = new_boxes[:,1]
  w = new_boxes[:,2]-xmin+1.
  h = new_boxes[:,3]-ymin+1.
  for item in range(output.shape[0]):
    xmin_ = xmin[item]
    ymin_ = ymin[item]
    w_ = w[item]
    h_ = h[item]
    y_all = np.arange(ymin_, ymin_+h_)
    Y_all = (y_all-ymin_)*target_output_size[0]/h_
    Y_all = np.maximum(0, np.minimum(target_output_size[0]-1, np.floor(Y_all)))

    x_all = np.arange(xmin_, xmin_+w_)
    X_all = (x_all-xmin_)*target_output_size[1]/w_
    X_all = np.maximum(0, np.minimum(target_output_size[1]-1, np.floor(X_all)))


    x_all = x_all.astype(np.intp)
    y_all = y_all.astype(np.intp)
    X_all = X_all.astype(np.intp)
    Y_all = Y_all.astype(np.intp)

    spind = sp[np.ix_(y_all,x_all)]-1
       

    for channel in range(output.shape[1]):
      pasted_output_this = pasted_output[item,channel]
      output_this = output[item, channel]
      vals = output_this[np.ix_(Y_all,X_all)]
      np.add.at(pasted_output_this, spind, vals)


  #finally divide by sum
  counts = accum.my_accumarray(sp.reshape(-1)-1,1,np.max(sp))
  pasted_output = pasted_output/counts.reshape((1,1,-1))
  return pasted_output
 
def get_all_outputs(net, imdb, nms_boxes, sp_dir, 
  img_blob_names, output_blob_name, det_salt='', eval_salt='', 
  sp_thresh=0.4, out_dir = None, do_eval = True, 
  eval_thresh = [0.5, 0.7], save_output=False):

  numcategs = imdb.num_classes-1
  
  if save_output:
    nms_sds = [[[] for _ in a] for a in nms_boxes]

  
  if do_eval:
    # we will accumulate the overlaps and the classes of the gt
    all_ov=[]
    gt = []
    for j in range(numcategs):
      all_ov.append([])
  
  ap_file = None
  if out_dir is not None:
    ap_file = out_dir + '_eval'

  #a dictionary of times
  times = {}
  times['boxes']=0.
  times['pred']=0.
  times['sp']=0.
  times['ov']=0.
  times['total']=0.
  for i in range(imdb.num_images):
    t1=time.time()
    
    im_names = imdb.image_path_at(i)
    imgs = []
    for j in range(len(im_names)):
      imgs.append(cv2.imread(imdb.image_path_at(i)[j]))
    
    #get all boxes for this image
    boxes_img = np.zeros((0,4))
    cids_img = np.zeros((0,1))
    for j in range(numcategs):
      if len(nms_boxes[j][i]) > 0:
        boxes_img = np.vstack((boxes_img, 1*nms_boxes[j][i][:,:4]))
        cids_img = np.vstack((cids_img, j*np.ones((nms_boxes[j][i].shape[0],1))))
    
    t2=time.time()
    times['boxes']=times['boxes']+t2-t1


    #get the predictions
    output = get_hypercolumn_prediction(net, imgs, boxes_img.astype(np.float32), cids_img,
      img_blob_names=img_blob_names, output_blob_name=output_blob_name)
    
    t3=time.time()
    times['pred']=times['pred']+t3-t2
    
    #project to sp
    sp = cv2.imread(os.path.join(sp_dir, imdb.image_index[i] + '.png'), cv2.CV_16U)
    newreg2sp_all = paste_output_sp(output, boxes_img.astype(np.float32), sp.shape, sp)
    newreg2sp_all = np.squeeze(newreg2sp_all)
    newreg2sp_all = newreg2sp_all >= sp_thresh
    newreg2sp_all = newreg2sp_all.T
    t4 = time.time()
    times['sp'] = times['sp']+t4-t3
    
    #save if needed
    if save_output:
      for j in range(numcategs):
        ind = np.where(cids_img[:,0] == j)[0]
        nms_sds[j][i] = (newreg2sp_all[:,ind].copy()).T
        
      # savemat(os.path.join(out_dir, imdb.image_index[i] + '.mat'), {'output': output})
    
    #evaluate
    if do_eval:
      inst, categories = imdb._sds_eval_format(i)
      
      ov = evaluation.compute_overlap_sprep(sp, newreg2sp_all, inst)
      #separate according to categories
      for j in range(numcategs):
        all_ov[j].append(ov[:,cids_img.reshape(-1)==j])
      
      #append categories
      gt.append(np.squeeze(categories-1))
    
    t5=time.time()
    times['ov'] = times['ov']+t5-t4
    
    if i % 10 == 0:
      total = float(i+1)
      print 'Doing : {:4d} / {:4d}, get boxes:{:.2f} s, get pred:{:.2f} s, get sp:{:.2f} s, get ov:{:.2f} s'.\
        format(i, imdb.num_images, times['boxes']/total, times['pred']/total, times['sp']/total, times['ov']/total)

  ap = [[] for _ in eval_thresh]
  prec = [[] for _ in eval_thresh]
  rec = [[] for _ in eval_thresh]

  if save_output:
    import general_utils as sg_utils 
    sds_file = os.path.join(out_dir, 'detections' + det_salt + '.pkl')
    sg_utils.save_variables(sds_file, [nms_boxes, nms_sds], ['nms_boxes', 'nms_sds'], overwrite = True)
  
  if ap_file is not None:
    f = open(ap_file + det_salt + eval_salt + '.txt', 'wt')

  for i in range(numcategs):
    for t, thr in enumerate(eval_thresh):
      sc = []
      for b in nms_boxes[i]:
        if b == []:
          sc.append(np.zeros(0))
        else:
          sc.append(b[:,4]*1)
      ap_, prec_, rec_ = evaluation.generalized_det_eval_simple(sc, all_ov[i], gt, i, thr)
      ap[t].append(ap_)
      prec[t].append(prec_)
      rec[t].append(rec_)
    
    ap_str = '{:>20s} : {:10f}, {:10f}'.format(imdb.classes[i+1], ap[0][i]*100, ap[1][i]*100)
    print ap_str
    if ap_file is not None:
      f.write(ap_str + '\n')
  

  ap_str = '{:>20s} : {:10f}, {:10f}'.format('mean', np.mean(ap[0])*100, np.mean(ap[1])*100)
  print ap_str
  if ap_file is not None:
    f.write(ap_str + '\n')
    f.close()
  
  if ap_file is not None:
    import general_utils as sg_utils 
    eval_file = os.path.join(ap_file + det_salt + eval_salt + '.pkl')
    sg_utils.save_variables(eval_file, [ap, prec, rec, imdb.classes[1:]], \
        ['ap', 'prec', 'rec', 'classes'], overwrite = True)
    eval_file = os.path.join(ap_file + det_salt + eval_salt + '.mat')
    savemat(eval_file, 
      {'ap': ap, 'prec': prec, 'rec': rec, 'classes': imdb.classes[1:]}, 
      do_compression = True);
  return ap, prec, rec, all_ov, gt
