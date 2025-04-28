import os
import pickle
import gzip
import urllib.request
import numpy as np

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    url = "https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz"
    save_path = "mnist.pkl.gz"
    
    # 如果本地没有，就下载
    if not os.path.exists(save_path):
        print(f"Downloading {save_path} from {url}...")
        urllib.request.urlretrieve(url, save_path)
        print("Download finished!")

    # 读取数据
    with gzip.open(save_path, 'rb') as f:
        train_set, val_set, test_set = pickle.load(f, encoding='latin1')
        
    # 只用训练集、测试集（忽略验证集）
    train_img, train_label = train_set
    test_img, test_label = test_set

    # 处理 normalize
    if normalize:
        train_img = train_img.astype(np.float32)
        test_img = test_img.astype(np.float32)

    # 处理 flatten
    if not flatten:
        train_img = train_img.reshape(-1, 28, 28)
        test_img = test_img.reshape(-1, 28, 28)

    # 处理 one_hot_label
    if one_hot_label:
        train_label = _to_one_hot(train_label)
        test_label = _to_one_hot(test_label)

    return (train_img, train_label), (test_img, test_label)

def _to_one_hot(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    for idx, label in enumerate(labels):
        one_hot[idx, label] = 1
    return one_hot
