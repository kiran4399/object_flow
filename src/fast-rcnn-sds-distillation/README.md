# Cross Modal Distillation for Supervision Transfer
Saurabh Gupta, Judy Hoffman, Jitendra Malik

This codebase allows use of RGB-D object detection models from this [arXiv tech report](http://arxiv.org/abs/1507.00448). 

### License

This code base is built on Fast R-CNN. License for Fast R-CNN can be found in LICENSE_fast_rcnn.

### Citing 

If you find this code base and models useful in your research, please consider citing an appropriate sub-set of the following papers:

    @article{gupta2015cross,
      title={Cross Modal Distillation for Supervision Transfer},
      author={Gupta, Saurabh and Hoffman, Judy and Malik, Jitendra},
      journal={arXiv preprint arXiv:1507.00448},
      year={2015}
    }

    @incollection{gupta2014learning,
      title={Learning rich features from RGB-D images for object detection and segmentation},
      author={Gupta, Saurabh and Girshick, Ross and Arbel{\'a}ez, Pablo and Malik, Jitendra},
      booktitle={Computer Vision--ECCV 2014},
      pages={345--360},
      year={2014},
      publisher={Springer}
    }

    @article{girshick15fastrcnn,
        Author = {Ross Girshick},
        Title = {Fast R-CNN},
        Journal = {arXiv preprint arXiv:1504.08083},
        Year = {2015}
    }
    
    @phdthesis{Hariharan:EECS-2015-193,
    	Author = {Hariharan, Bharath},
        Title = {Beyond Bounding Boxes: Precise Localization of Objects in
        Images},
        School = {EECS Department, University of California, Berkeley},
        Year = {2015},
        Month = {Aug},
        URL = {http://www.eecs.berkeley.edu/Pubs/TechRpts/2015/EECS-2015-193.html},
       	Number = {UCB/EECS-2015-193}
    }

### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Basic installation](#installation-sufficient-for-the-demo)

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  ```

2. Python packages you might not have: `cython`, `python-opencv`, `easydict`

### Requirements: hardware

1. For training smaller networks (CaffeNet, VGG_CNN_M_1024) a good GPU (e.g., Titan, K20, K40, ...) with at least 3G of memory suffices
2. For training with VGG16, you'll need a K40 (~11G of memory)

### Installation (sufficient for the demo)

1. Clone the repository
  ```Shell
  # Clone the python code
  git clone https://github.com/s-gupta/fast-rcnn.git
  git checkout sds-distillation
  ```
  We'll call the directory that you cloned Fast R-CNN into `FRCN_ROOT`. 

2. Clone Caffe with roi_pooling_layers:

    ```Shell
    cd $FRCNN_ROOT
    git clone https://github.com/rbgirshick/caffe-fast-rcnn.git caffe-fast-rcnn
    cd caffe-fast-rcnn
    # caffe-fast-rcnn needs to be on the fast-rcnn branch (or equivalent detached state).
    git checkout fast-rcnn
    ```
    
3. Clone Caffe with SDS layers:

    ```Shell
    cd $FRCNN_ROOT
    git clone https://github.com/bharath272/caffe-sds.git caffe-sds
    ```
    
3. Build the Cython modules
    ```Shell
    cd $FRCN_ROOT/lib
    make
    ```
    
4. Build Caffe and pycaffe, for both caffe-fast-rcnn and caffe-sds:
    ```Shell
    cd $FRCN_ROOT/caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do.
    # Make sure caffe is built with PYTHON layers.
    make -j8 && make pycaffe
    
    cd $FRCN_ROOT/caffe-sds
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do.
    # Make sure caffe is built with PYTHON layers.
    make -j8 && make pycaffe
    ```
    
### Download models and data
1. Download the NYUD2 data

  ```Shell
  cd $FRCN_ROOT
  ./data/scripts/fetch_nyud2_data.sh
  ```
	
2. Download the NYUD2 MCG boxes

  ```Shell
  cd $FRCN_ROOT
  ./data/scripts/fetch_nyud2_mcg_boxes.sh
  ```
  
3. Download the NYUD2 Superpixels

  ```Shell
  cd $FRCN_ROOT
  ./data/scripts/fetch_nyud2_sp.sh
  ```

3. Download the ImageNet and Supervision Transfer Models 

  ```Shell
  cd $FRCN_ROOT
  ./data/scripts/fetch_init_models.sh
  ```

4. Fetch NYUD2 Object Detector Models.

  ```Shell
  cd $FRCN_ROOT
  ./output/scripts/fetch_nyud2_detectors.sh
  ```

5. Fetch NYUD2 SDS Models.

  ```Shell
  cd $FRCN_ROOT
  ./output/scripts/fetch_nyud2_sds.sh
  ```

### Usage

1. Look at experiments/sds_test_pretrained_models.sh and experiments/sds_train_models.sh to use pretrained SDS models and train SDS models yourself. Additionally, look at experiments/test_pretrained_models.sh and experiments/train_models.sh to use pretrained models and train your models yourself.
2. The current code base uses two different versions of caffe (one with roi_pooling layer and another with sds layers). Currently, these two versions aren't merged, and hence when using the detector please change to the roi_pooling layer caffe (``caffe-fast-rcnn``), and when using the sds model on the detections please change to the sds layer caffe (``caffe-sds``) in ``_init_paths.py``. I am sorry for this, and will soon merge these two.
