# This script needs to use caffe-sds. Uncomment appropriately in _init_paths.py


# Train the RGB Alexnet model finetuning from the alexnet rgb model finetuned for detection
PYTHONUNBUFFERED="True" PYTHONPATH=:.:caffe-sds/build/python/ GLOG_logtostderr=1 \
caffe-sds/build/tools/caffe.bin train \
  -gpu 1 \
  -solver output/sds/training_demo/alexnet_rgb/solver.prototxt \
  -weights output/alexnet_rgb_alexnet_hha/nyud2_images+hha_2015_trainval/fast_rcnn_iter_40000.caffemodel \
  2>&1 | tee output/sds/training_demo/alexnet_rgb/train.log

# Train the RGB Alexnet model finetuning from the alexnet rgb model finetuned for detection
PYTHONUNBUFFERED="True" PYTHONPATH=:.:caffe-sds/build/python/ GLOG_logtostderr=1 \
caffe-sds/build/tools/caffe.bin train \
  -gpu 0 \
  -solver output/sds/training_demo/alexnet_hha/solver.prototxt \
  -weights output/alexnet_rgb_alexnet_hha/nyud2_images+hha_2015_trainval/fast_rcnn_iter_40000.caffemodel \
  2>&1 | tee output/sds/training_demo/alexnet_hha/train.log

# Train a VGG RGB models for instance segmentation again starting from the finetuned for detection model
PYTHONUNBUFFERED="True" PYTHONPATH=:caffe-sds/python/ GLOG_logtostderr=1 \
caffe-sds/build/tools/caffe.bin train \
  -gpu 0 \
  -solver output/sds/training_demo/vgg_rgb/solver.prototxt \
  -weights output/vgg_rgb_alexnet_hha/nyud2_images+hha_2015_trainval/fast_rcnn_iter_40000.caffemodel \
  2>&1 | tee output/sds/training_demo/vgg_rgb/train.log


# test rgb only models
modality="images"
for model in "alexnet_rgb" "vgg_rgb"; do
  PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/test_net.py --gpu 1 \
    --def output/sds/training_demo/$model/test.prototxt.$modality \
    --net output/sds/training_demo/$model/nyud2_$modality""_2015_trainval/inst_segm_iter_10000.caffemodel \
    --imdb nyud2_$modality"_2015_test" \
    --sds 1 --sds_sp_dir data/nyud2_sp --sds_detection_file output/$model/nyud2_$modality"_2015_test"/fast_rcnn_iter_40000/detections.pkl \
    --sds_img_blob_name image --sds_output_blob_name loss --sds_sp_thresh 0.4 \
    --set EXP_DIR sds/training_demo/$model SDS.TARGET_SIZE 688 SDS.DET_SALT _rgbd_det
done

# test hha only models
modality="hha"
for model in "alexnet_hha"; do
  PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/test_net.py --gpu 1 \
    --def output/sds/training_demo/$model/test.prototxt.$modality \
    --net output/sds/training_demo/$model/nyud2_$modality""_2015_trainval/inst_segm_iter_10000.caffemodel \
    --imdb nyud2_$modality"_2015_test" \
    --sds 1 --sds_sp_dir data/nyud2_sp --sds_detection_file output/$model/nyud2_$modality"_2015_test"/fast_rcnn_iter_40000/detections.pkl \
    --sds_img_blob_name image --sds_output_blob_name loss --sds_sp_thresh 0.4 \
    --set EXP_DIR sds/training_demo/$model SDS.TARGET_SIZE 688
done

# test hha + rgb models
modality="images+hha"
hha_model="alexnet_hha"
# for rgb_model in "alexnet_rgb" "vgg_rgb"; do
for rgb_model in "vgg_rgb"; do
  PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/test_net.py --gpu 1 \
    --def output/sds/training_demo/$rgb_model"_"$hha_model/test.prototxt.$modality \
    --net output/sds/training_demo/$rgb_model/nyud2_images_2015_trainval/inst_segm_iter_10000.caffemodel output/sds/training_demo/$hha_model/nyud2_hha_2015_trainval/inst_segm_iter_10000.caffemodel \
    --imdb nyud2_$modality"_2015_test" \
    --sds 1 --sds_sp_dir data/nyud2_sp --sds_detection_file output/$rgb_model"_"$hha_model/nyud2_$modality"_2015_test"/fast_rcnn_iter_40000/detections.pkl \
    --sds_img_blob_name image da_image --sds_output_blob_name fusion_loss --sds_sp_thresh 0.4 --sds_save_output 1 \
    --set EXP_DIR sds/training_demo/$rgb_model"_"$hha_model SDS.TARGET_SIZE 688 
done
