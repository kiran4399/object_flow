# This script needs to use caffe-fast-rcnn. Uncomment appropriately in _init_paths.py

set -x
tr_set='trainval'
test_set='test'

modality="images+hha"
for model in "alexnet_rgb_alexnet_hha" "vgg_rgb_alexnet_hha"; do
  PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/test_net.py --gpu 0 \
    --def output/$model/test.prototxt.$modality \
    --net output/$model/nyud2_images+hha_2015_$tr_set/fast_rcnn_iter_40000.caffemodel \
    --imdb nyud2_"$modality"_2015_"$test_set" \
    --cfg output/$model/config.prototxt
done

# test only alexnet RGB 
model="alexnet_rgb"
modality="images"
PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/test_net.py --gpu 1 \
  --def output/$model/test.prototxt.$modality \
  --net output/alexnet_rgb_alexnet_hha/nyud2_images+hha_2015_$tr_set/fast_rcnn_iter_40000.caffemodel \
  --imdb nyud2_"$modality"_2015_"$test_set" \
  --set EXP_DIR $model TEST.SCALES [688] 

# test only alexnet HHA 
model="alexnet_hha"
modality="hha"
PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/test_net.py --gpu 1 \
  --def output/$model/test.prototxt.$modality \
  --net output/alexnet_rgb_alexnet_hha/nyud2_images+hha_2015_$tr_set/fast_rcnn_iter_40000.caffemodel \
  --imdb nyud2_"$modality"_2015_"$test_set" \
  --set EXP_DIR $model TEST.SCALES [688] 


# test only vgg RGB
# vgg RGB model was trained with an image size of 600 and not 688. Testing this model
# with image size of 600 gives 38.8, with image size of 688 gives 36.0. 
model="vgg_rgb"
modality="images"
PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/test_net.py --gpu 1 \
  --def output/$model/test.prototxt.$modality \
  --net output/vgg_rgb_alexnet_hha/nyud2_images+hha_2015_$tr_set/fast_rcnn_iter_40000.caffemodel \
  --imdb nyud2_"$modality"_2015_"$test_set" \
  --set EXP_DIR $model TEST.SCALES [600]

# PYTHONPATH='.' PYTHONUNBUFFERED="True" python tools/test_net.py --gpu 1 \
#   --def output/$model/test.prototxt.$modality \
#   --net output/vgg_rgb_alexnet_hha/nyud2_images+hha_2015_$tr_set/fast_rcnn_iter_40000.caffemodel \
#   --imdb nyud2_"$modality"_2015_"$test_set" \
#   --set EXP_DIR $model TEST.SCALES [688] TEST.DET_SALT _688
