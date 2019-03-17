import os
# import sys

import h5py
import numpy as np
# import tensorflow as tf
from keras.applications.resnet50 import preprocess_input
# from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.models import load_model
from keras.preprocessing import image

from file_helper import get_imlist


def extract_feature(dir_path, net):
    img_list = get_imlist(dir_path)

    features = []
    names = []

    for i, image_path in enumerate(img_list):
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = net.predict(x)
        features.append(np.squeeze(feature))

        img_name = os.path.split(image_path)[1]  # 分割路径中最后一个‘/’与前面的部分，并取最后一个‘/’后面的文件名
        names.append(img_name.encode())  # 文件名字符串使用h5文件保存时必须用ascii编码
        print("extracting feature from image No. %d , %d images in total" % ((i + 1), len(img_list)))

    return np.array(features), np.array(names)


def write_to_h5(output, figures, deep_feats):

    fig_name = 'fig_name'
    deep_f_name = 'deep_f'

    h5f = h5py.File(output, 'w')
    h5f.create_dataset(fig_name, data=figures)
    h5f.create_dataset(deep_f_name, data=deep_feats)

    h5f.close()


if __name__ == '__main__':
    # path2 = os.path.abspath('..')
    # sys.path.extend([path2])

    # sources = ['market', 'mix', 'cuhk03', 'cuhk03']
    source = 'cuhk03'
    source_model_path = './model/' + source + '_pair_model.h5'
    model = load_model(source_model_path)

    # dataset = ['search_test_a', 'search_val']
    probe_path = './dataset/search_a/search_val/probe/'
    gallery_path = './dataset/search_a/search_val/gallery/'

    print('------------extracting deep features---------------')
    model = Model(inputs=[model.get_layer('resnet50').get_input_at(0)],
          outputs=[model.get_layer('resnet50').get_output_at(0)])
    net = Model(inputs=[model.input], outputs=[model.get_layer('avg_pool').output])
    gallery_deep, gallery_fig = extract_feature(gallery_path, net)
    probe_deep, probe_fig = extract_feature(probe_path, net)

    # write the extracted features to the h5 file
    print('------------writing features to the file-----------')
    output = './result/gallery.h5'
    write_to_h5(output, gallery_fig, gallery_deep)

    output = './result/probe.h5'
    write_to_h5(output, probe_fig, probe_deep)

    print('-----deep feature extration done-------')