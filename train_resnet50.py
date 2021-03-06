import os
from random import shuffle

import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.initializers import RandomNormal
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical


def load_data(LIST, TRAIN):
    images, labels = [], []
    with open(LIST, 'r') as f:
        last_label = -1
        label_cnt = -1
        for line in f:
            line = line.strip()
            img = line
            lbl = line.split('_')[0]
            if last_label != lbl:
                label_cnt += 1
            last_label = lbl
            img = image.load_img(os.path.join(TRAIN, img), target_size=[224, 224])
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            images.append(img[0])
            labels.append(label_cnt)

    img_cnt = len(labels)
    shuffle_idxes = list(range(img_cnt))
    shuffle(shuffle_idxes)
    shuffle_imgs = list()
    shuffle_labels = list()
    for idx in shuffle_idxes:
        shuffle_imgs.append(images[idx])
        shuffle_labels.append(labels[idx])
    images = np.array(shuffle_imgs)
    labels = to_categorical(shuffle_labels)
    return images, labels


def softmax_model_pretrain(train_list, train_dir, class_count, target_model_path):
    images, labels = load_data(train_list, train_dir)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    set_session(sess)

    # load pre-trained resnet50
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    x = base_model.output
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)
    x = Dense(class_count, activation='softmax', name='fc8', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(x)
    net = Model(inputs=[base_model.input], outputs=[x])

    for layer in net.layers:
        layer.trainable = True

    # pretrain
    batch_size = 16
    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2)

    net.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    net.fit_generator(
        train_datagen.flow(images, labels, batch_size=batch_size),
        steps_per_epoch=len(images) / batch_size + 1, epochs=40,
    )
    net.save(target_model_path)


def softmax_pretrain_on_dataset(source, project_path):
    if source == 'market':
        train_list = project_path + '/dataset/dataset-list/market_train.list'
        train_dir = project_path + '/dataset/Market-1501/train'
        class_count = 751
    elif source == 'cuhk01':
        train_list = project_path + '/dataset/dataset-list/cuhk_train.list'
        train_dir = project_path + '/dataset/cuhk01'
        class_count = 971
    elif source == 'cuhk03':
        train_list = project_path + '/dataset/dataset-list/cuhk03.list'
        train_dir = project_path + '/dataset/cuhk03'
        class_count = 767
    elif 'mix' == source:
        train_list = project_path + '/dataset/dataset-list/mix_train.list'
        train_dir = project_path + '/dataset/mix'
        class_count = 250 + 971 + 630
    elif 'mix2' == source:
        train_list = project_path + '/dataset/dataset-list/mix2.list'
        train_dir = project_path + '/dataset/mix2'
        class_count = 250 + 971 + 630 + 767
    else:
        train_list = 'unknown'
        train_dir = 'unknown'
        class_count = -1
    softmax_model_pretrain(train_list, train_dir, class_count, project_path + '/model/' + source + '_resnet_model.h5')


if __name__ == '__main__':
    path1 = os.path.abspath('.')  # 表示当前所处的文件夹的绝对路径 .../cnn-based-hash-image-retrieval
    # path2 = os.path.abspath('..')  # 表示当前所处的文件夹上一级文件夹的绝对路径

    # sources = ['market', 'grid', 'cuhk03', 'viper', 'mix']
    sources = ['mix2']
    for source in sources:
        softmax_pretrain_on_dataset(source, path1)
