import options
import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.models import load_model
import time

def get_image(data_path, image_size):
    img = load_img(data_path, target_size=(image_size, image_size))
    tr_x = img_to_array(img)
    tr_x = tr_x[:, :, ::-1]
    tr_x = preprocess_input(tr_x)
    tr_x = np.expand_dims(np.array(tr_x), axis=0)

    return tr_x

if __name__ == '__main__':
    categorie = []
    opt = options.Options().parse()
    log_dir = opt.log_dir
    image_path = opt.image_path
    model = opt.model

    checkpoint_file = opt.model_file_name
    if len(checkpoint_file) < 1:
        raise Exception('please set model file name')
    outputDir = opt.outf
    channel = opt.nc
    size = opt.imsize
    n_class = opt.n_class
    if n_class == 2:
        categorie = ['abnormal', 'normal']
    elif n_class == 5:
        categorie = ['black', 'broken', 'fungus', 'insect', 'normal']
    else:
       raise ValueError('Only support to 2 or 5!')

    predict_dir = os.path.join(outputDir, 'predict')
    if not os.path.isdir(predict_dir):
        os.makedirs(predict_dir)

    start_time = time.time()
    model = load_model(os.path.join(log_dir, checkpoint_file))
    print('load model spendhnbvcxoi90 {:0.4f} s'.format(time.time() - start_time))
    start_time1 = time.time()
    testImages = get_image(image_path, size)
    print('load image spend {:0.4f} s'.format(time.time() - start_time1))
    start_time2 = time.time()
    result = model.predict(testImages, verbose=0, steps=None)
    print('predic spend {:0.4f} s'.format(time.time() - start_time2))

    print(categorie[np.argmax(result, axis=1)[0]])




