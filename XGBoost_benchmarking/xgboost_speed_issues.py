import xgboost as xgb
import time
import os
import struct
import numpy as np

# Trying to benchmark the XGBoost speed issue on MNIST dataset
# https://datascience.stackexchange.com/questions/61181/xgboost-speed-issues
# Read MNIST files
def read_images(images_name):
    f = open(images_name, "rb")
    ds_images = []
    mw_32bit = f.read(4)
    n_numbers_32bit = f.read(4)
    n_rows_32bit = f.read(4)
    n_columns_32bit = f.read(4)
    mw = struct.unpack('>i', mw_32bit)[0]
    n_numbers = struct.unpack('>i', n_numbers_32bit)[0]
    n_rows = struct.unpack('>i', n_rows_32bit)[0]
    n_columns = struct.unpack('>i', n_columns_32bit)[0]
    try:
        for i in range(n_numbers):
            image = []
            for r in range(n_rows):
                for l in range(n_columns):
                    byte = f.read(1)
                    pixel = struct.unpack('B', byte)[0]
                    image.append(pixel)
            ds_images.append(image)
    finally:
        f.close()
    return ds_images


def read_labels(labels_name):
    f = open(labels_name, "rb")
    ds_labels = []
    mw_32bit = f.read(4)
    n_numbers_32bit = f.read(4)
    mw = struct.unpack('>i', mw_32bit)[0]
    n_numbers = struct.unpack('>i', n_numbers_32bit)[0]
    try:
        for i in range(n_numbers):
            byte = f.read(1)
            label = struct.unpack('B', byte)[0]
            ds_labels.append(label)
    finally:
        f.close()
    return ds_labels


def read_dataset(images_name, labels_name):
    images = read_images(images_name)
    labels = read_labels(labels_name)
    assert len(images) == len(labels)
    return (images, labels)


def create_datasets(sample_dir):
    image_file = os.path.join(sample_dir, 'train-images-idx3-ubyte')
    label_file = os.path.join(sample_dir, 'train-labels-idx1-ubyte')
    training_images, training_labels = read_dataset(image_file, label_file)
    image_file = os.path.join(sample_dir, 't10k-images-idx3-ubyte')
    label_file = os.path.join(sample_dir, 't10k-labels-idx1-ubyte')
    testing_images, testing_labels = read_dataset(image_file, label_file)
    dtrain = xgb.DMatrix(
        np.asmatrix(training_images),
        label=training_labels,
        nthread=2
    )
    dtest = xgb.DMatrix(
        np.asmatrix(testing_images),
        label=testing_labels,
        nthread=2
    )
    data_dict = {
        'dtrain': dtrain,
        'dtest': dtest,
        'training_labels': training_labels,
        'testing_labels': testing_labels
    }
    return data_dict


def main():
    data_dict = create_datasets("D:\Data\MNIST")
    parameters = {
        'learning_rate': 0.1,
        'max_depth': 3,
        'gamma': 1,
        'min_child_weight': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.4,
        'verbosity': 1,
        'objective': 'multi:softprob',
        'num_class': 10,
        'nthread': 2,
        'seed': 1,
    }
    start = time.time()
    model = xgb.train(
        parameters,
        data_dict['dtrain'],
        num_boost_round=100
    )
    end = time.time()
    print("Model training time: ", end-start)
    start2 = time.time()
    pred_train = model.predict(data_dict['dtrain'])
    end2 = time.time()
    print("Predict time: ", end2-start2)


if __name__ == '__main__':
    main()