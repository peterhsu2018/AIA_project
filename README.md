## Coffee bean abnormal detection
   3 CNN models including Resnet50, Inception-Resnet v2 
   and InceptionV4 to prdict normal/abnormal or black/broken/fungus/insect/normal


| number of class   |  Inception V4     | Resnet50   | Inception-Resnet v2   |
| :---------------: | :---------------: | :--------: | :-------------------: |
|     2 class       |         92%       |   95%      |        97%            |
|     5 class       |         89%       |   96%      |        95%            |


# Inception V4 
## Run code
    python train.py \
    --dataroot datasets/train/ \
    --gpu_ids 0 \
    --batchsize 32 \
    --n_class 2 \
    --imsize 299

## Run validate set code
    python evaluate.py \
    --gpu_ids 0 \
    --batchsize 32 \
    --eval_type validate \
    --n_class 2 \
    --imsize 299

## Run test set code
    python evaluate.py \
    --dataroot datasets/test/ \
    --gpu_ids 0 \
    --batchsize 32 \
    --n_class 2 \
    --imsize 299
    
## Run predict one image by tensorflow model
    python testImage.py \
    --image_path datasets/test/normal/xxxx.jpg \
    --meta_graph_name model.ckpt-xxxx.meta \
    --n_class 2 \
    --imsize 299 
      
## Run predict one image by keras model
    python predictImage.py \
    --image_path datasets/test/normal/xxxx.jpg \
    --model_file_name xxxx.h5 \
    --n_class 2 \
    --imsize 299      
      