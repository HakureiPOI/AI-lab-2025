import os
import pickle
import gzip
import urllib.request
import numpy as np

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    url = "https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz"
    save_path = "mnist.pkl.gz"

    if not os.path.exists(save_path):
        _download_mnist(url, save_path)

    try:
        with gzip.open(save_path, 'rb') as f:
            train_set, val_set, test_set = pickle.load(f, encoding='latin1')
    except Exception as e:
        print("读取 mnist.pkl.gz 出错，可能文件损坏，尝试重新下载...")
        os.remove(save_path)
        _download_mnist(url, save_path)
        with gzip.open(save_path, 'rb') as f:
            train_set, val_set, test_set = pickle.load(f, encoding='latin1')

    train_img, train_label = train_set
    test_img, test_label = test_set

    if normalize:
        train_img = train_img.astype(np.float32)
        test_img = test_img.astype(np.float32)

    if not flatten:
        train_img = train_img.reshape(-1, 28, 28)
        test_img = test_img.reshape(-1, 28, 28)

    if one_hot_label:
        train_label = _to_one_hot(train_label)
        test_label = _to_one_hot(test_label)

    return (train_img, train_label), (test_img, test_label)

def _to_one_hot(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

def _download_mnist(url, save_path):
    tmp_path = save_path + ".tmp"
    print(f"Downloading MNIST dataset from {url}...")
    urllib.request.urlretrieve(url, tmp_path)
    os.rename(tmp_path, save_path)
    print("Download finished!")
