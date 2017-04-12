import cv2
import numpy as np

def paste_boxes(im_sz, rois, conf=None, im = None, col1 = [1.0, 0.0, 0.0]):
  #print rois.shape
  #print im_sz
  out = np.zeros(im_sz, dtype = np.float32)
  if conf is None:
    conf = np.ones(rois.shape[0])
  #print out.shape
  for i in xrange(rois.shape[0]):
    out[rois[i,1]:rois[i,3], rois[i,0]:rois[i,2]][:] += conf[i]/rois.shape[0]
  out = out/np.max(out)
  if im is not None:
    out2 = im*0.25
    col2 = [1.0,1.0,1.0]
    for k in range(3):
      out2[:,:,k] += 0.75*(col1[k]*out + col2[k]*(1.-out))
    out = out2
  return out

def paste_bboxes(im, rois, col1=[1.0, 0.0, 0.0], w=10):
  out = im.copy()
  x = np.arange(im.shape[1]).reshape((1,-1))
  y = np.arange(im.shape[0]).reshape((-1,1))
  for i in range(rois.shape[0]):
    idx_xmin = np.logical_and(np.logical_and(x>=rois[i,0]-w/2, x<=rois[i,0]+w/2), np.logical_and(y>=rois[i,1]-w/2, y<=rois[i,3]+w/2))
    idx_ymin =  np.logical_and(np.logical_and(y>=rois[i,1]-w/2, y<=rois[i,1]+w/2), np.logical_and(x>=rois[i,0]-w/2, x<=rois[i,2]+w/2))
    idx_xmax =  np.logical_and(np.logical_and(x>=rois[i,2]-w/2, x<=rois[i,2]+w/2), np.logical_and(y>=rois[i,1]-w/2, y<=rois[i,3]+w/2))
    idx_ymax =  np.logical_and(np.logical_and(y>=rois[i,3]-w/2, y<=rois[i,3]+w/2), np.logical_and(x>=rois[i,0]-w/2, x<=rois[i,2]+w/2))

    idx = np.logical_or(np.logical_or(idx_xmin, idx_ymin),np.logical_or(idx_xmax, idx_ymax))
    for j in range(3):
      tmp=out[:,:,j]
      tmp[idx] = col1[j]
      out[:,:,j]=tmp
  return out
   


def vis_best_box(imdb, roidb, cat_id, all_boxes):
  import roi_data_layer.context_utils as context_utils 
  # From the set of generated proposals for each image, show the best
  # overlapping box to the ground truth
  info = []
  for i, r in enumerate(roidb):
    gt_ind = r['gt_classes'] == cat_id
    if any(gt_ind):
      gt_boxes = r['boxes'][gt_ind, :].reshape((-1, 4))
      ab = all_boxes[i]
      # Compute overlap between boxes
      ov = context_utils.compute_box_overlap(ab, gt_boxes)
      ind = np.argmax(ov, axis = 0).reshape((-1))
      ov = np.max(ov, axis = 0).reshape((-1))
      for j in range(gt_boxes.shape[0]):
        # print ind, imdb.image_path_at(i), gt_boxes[j,:], ind[j], ab[ind[j], :],ov[j]
        info.append((imdb.image_path_at(i), gt_boxes[j, :], ab[ind[j], :], ov[j]))
  return info
# def vis_boxes():

def subplot(plt, (Y, X), (sz_y, sz_x) = (10, 10)):
  plt.rcParams['figure.figsize'] = (X*sz_x, Y*sz_y)
  fig, axes = plt.subplots(Y, X)
  return fig, axes

def vis_detections(plt, dt, gt, details, imdb, filter_str=True, num_output=30, out_file_name=None, RC=None):
    # walk down the detections and visualize things which pass the filter
    cols = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    tog = []
    for i, sc in enumerate(details['score']):
        tog.append(np.hstack((i*np.ones_like(sc), np.arange(sc.shape[0])[:,np.newaxis], sc)))
    tog = np.vstack(tog)
    ind = np.argsort(tog[:,2])[::-1]
    tog = tog[ind,:]
    cnt = 0;

    if RC is None:
      fig, axes = subplot(plt, (np.int(np.ceil(num_output/3.0)),3))
    else: 
      fig, axes = subplot(plt, RC)

    for j in xrange(tog.shape[0]):
        im_id = int(tog[j,0]); box_id = int(tog[j,1]);
        p = {}
        for t in ['fp', 'score', 'instId', 'ov', 'dupDet', 'tp']:
            p[t] = details[t][im_id][box_id,0]
        vis_it = eval(filter_str, p)
        if vis_it:
            # load the image
            im = cv2.imread(imdb.image_path_at(im_id))[:,:,::-1]

            # draw the detection
            ax = axes.ravel()[cnt]
            ax.imshow(im)
            ax.set_axis_off()
            roi = dt[im_id]['boxInfo'][box_id,:].astype(np.int)

            # draw the ground truth box if there is
            ax.add_patch(plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0], roi[3] - roi[1], 
                                       fill=False, edgecolor=cols[2-p['fp']], linewidth=3))
            title_str = '[{:d} {:d} {:d}], {:3.2f} #{:d}, ov: {:3.2f}'.format(
                p['tp'], p['fp'], p['dupDet'], p['score'], j, p['ov'])

            if p['ov'] > 0:
                roi = gt[im_id]['boxInfo'][p['instId'],:].astype(np.int)
                ax.add_patch(plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0], roi[3] - roi[1], 
                               fill=False, edgecolor=cols[0], linewidth=2))
            ax.set_title(title_str, {'fontsize': 20})
            cnt = cnt+1
        if cnt >= num_output:
            break

    if out_file_name is not None:
        plt.savefig(out_file_name)
    return fig, axes

def draw_bbox(plt, ax, rois, fill=False, linewidth=2, edgecolor=[1.0, 0.0, 0.0], **kwargs):
  for i in range(rois.shape[0]):
    roi = rois[i,:].astype(np.int)
    ax.add_patch(plt.Rectangle((roi[0], roi[1]), 
      roi[2] - roi[0], roi[3] - roi[1],
      fill=False, linewidth=linewidth, edgecolor=edgecolor, **kwargs))

def color_seg(mask, img):
  #convert to single
  img2 = img.copy().astype(np.float32)
  mask2 = mask.copy().astype(np.float32)[:,:,np.newaxis]
  color = np.array([255., 0., 0.])
  color = color[np.newaxis, np.newaxis,:]
  colorbg = np.array([255., 255., 255.])
  colorbg = colorbg[np.newaxis, np.newaxis,:]
  img2 = 0.5*img2 + 0.5*mask2*color + 0.5*(1.-mask2)*colorbg
  return img2.astype(np.uint8)


def show_mask_on_image(image, mask, col, f=0.25):
  out = image*f
  white = [1.0,1.0,1.0]
  for k in range(3):
    out[:,:,k] += (1-f)*(col[k]*mask + white[k]*(1.-mask))
  return out
