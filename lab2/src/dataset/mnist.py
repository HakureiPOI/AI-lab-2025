import os
import pickle
import gzip
import numpy as np
import urllib.request

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    url = 'http://yann.lecun.com/exdb/mnist/'
    key_file = {
        'train_img': 'train-images-idx3-ubyte.gz',
        'train_label': 'train-labels-idx1-ubyte.gz',
        'test_img': 't10k-images-idx3-ubyte.gz',
        'test_label': 't10k-labels-idx1-ubyte.gz'
    }
    dataset_dir = os.path.dirname(__file__)  # 当前文件夹
    save_file = os.path.join(dataset_dir, 'mnist.pkl')

    # 如果 pkl 文件存在，直接读取
    if os.path.exists(save_file):
        with open(save_file, 'rb') as f:
            dataset = pickle.load(f)
    else:
        # 检查是否有数据文件，没有就下载
        for v in key_file.values():
            file_path = os.path.join(dataset_dir, v)
            if not os.path.exists(file_path):
                print(f"Downloading {v}...")
                urllib.request.urlretrieve(url + v, file_path)

        # 读取数据
        dataset = {}
        dataset['train_img'] = _load_imgs(os.path.join(dataset_dir, key_file['train_img']))
        dataset['train_label'] = _load_labels(os.path.join(dataset_dir, key_file['train_label']))
        dataset['test_img'] = _load_imgs(os.path.join(dataset_dir, key_file['test_img']))
        dataset['test_label'] = _load_labels(os.path.join(dataset_dir, key_file['test_label']))

        # 保存成 pkl 文件
        with open(save_file, 'wb') as f:
            pickle.dump(dataset, f, -1)

    # 处理 normalize / flatten / one_hot_label
    train_img, train_label = _preprocess(dataset['train_img'], dataset['train_label'], normalize, flatten, one_hot_label)
    test_img, test_label = _preprocess(dataset['test_img'], dataset['test_label'], normalize, flatten, one_hot_label)

    return (train_img, train_label), (test_img, test_label)

def _load_imgs(file_path):
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def _load_labels(file_path):
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return labels

def _preprocess(imgs, labels, normalize, flatten, one_hot_label):
    if normalize:
        imgs = imgs.astype(np.float32)
        imgs /= 255.0
    if flatten:
        imgs = imgs.reshape(imgs.shape[0], -1)  # (N, 784)
    if one_hot_label:
        labels = _to_one_hot(labels)
    return imgs, labels

def _to_one_hot(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    for idx, row in enumerate(one_hot):
        row[labels[idx]] = 1
    return one_hot
