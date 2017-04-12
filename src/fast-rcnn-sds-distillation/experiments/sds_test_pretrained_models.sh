# This script needs to use caffe-sds. Uncomment appropriately in _init_paths.py
set -x

# To test any of these models you will need detections on the test set from the
# various detection models. Please run experiments/test_pretrained_models.sh to
# generate the detections. Note that to run the detection script, you need to
# use caffe-fast-rcnn and not caffe-sds. To use caffe-fast-rcnn, uncomment
# appropriately in _init_paths.py.

# Testing a RGB + HHA model
modality="images+hha"
hha_model="alexnet_hha"
for rgb_model in "vgg_rgb"; do
  PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/test_net.py --gpu 0 \
    --def output/sds/$rgb_model"_"$hha_model/test.prototxt.$modality \
    --net output/sds/$rgb_model/nyud2_images_2015_trainval/inst_segm_iter_10000.caffemodel output/sds/$hha_model/nyud2_hha_2015_trainval/inst_segm_iter_10000.caffemodel \
    --imdb nyud2_$modality"_2015_test" \
    --sds 1 --sds_sp_dir data/nyud2_sp --sds_detection_file output/$rgb_model"_"$hha_model/nyud2_images+hha_2015_test/fast_rcnn_iter_40000/detections.pkl \
    --sds_img_blob_name image da_image --sds_output_blob_name fusion_loss --sds_sp_thresh 0.4 --sds_save_output 1 \
    --set EXP_DIR sds/$rgb_model"_"$hha_model SDS.TARGET_SIZE 688 
done

# Testing RGB only models
modality="images"
for model in "alexnet_rgb" "vgg_rgb"; do
  PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/test_net.py --gpu 1 \
    --def output/sds/$model/test.prototxt.$modality \
    --net output/sds/$model/nyud2_$modality""_2015_trainval/inst_segm_iter_10000.caffemodel \
    --imdb nyud2_$modality"_2015_test" \
    --sds 1 --sds_sp_dir data/nyud2_sp --sds_detection_file output/$model/nyud2_$modality"_2015_test"/fast_rcnn_iter_40000/detections.pkl \
    --sds_img_blob_name image --sds_output_blob_name loss --sds_sp_thresh 0.4 \
    --set EXP_DIR sds/$model SDS.TARGET_SIZE 688
done

# Testing HHA only model
modality="hha"
for model in "alexnet_hha"; do
  PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/test_net.py --gpu 1 \
    --def output/sds/$model/test.prototxt.$modality \
    --net output/sds/$model/nyud2_$modality""_2015_trainval/inst_segm_iter_10000.caffemodel \
    --imdb nyud2_$modality"_2015_test" \
    --sds 1 --sds_sp_dir data/nyud2_sp --sds_detection_file output/$model/nyud2_$modality"_2015_test"/fast_rcnn_iter_40000/detections.pkl \
    --sds_img_blob_name image --sds_output_blob_name loss --sds_sp_thresh 0.4 \
    --set EXP_DIR sds/$model SDS.TARGET_SIZE 688
done
