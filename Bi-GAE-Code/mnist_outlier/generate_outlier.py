import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os


# gzip nist*/*ubyte and gzip *ubyte to compress the ubyte file

def float_to_int(images):
    n, d = images.shape
    #imagedata = numpy.zeros_like(images, dtype=numpy.uint8)
    images *= 255.0
    return images.astype(np.uint8)


def dense_to_one_hot(labels, num_classes):
    num_labels = labels.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot

def generate_outlier_mnist(mnist, ratio):
    train_images = mnist.train.images
    train_labels = mnist.train.labels
    label_numbers = np.where(train_labels == 1)[1]
    mixed_images = np.array([])
    mixed_labels = np.array([])
    true_labels = np.array([])

    for i in range(10):
        i_positions = np.where(label_numbers == i)[0].reshape(-1)
        i_images = train_images[i_positions]
        i_quantity = i_images.shape[0]
        outlier_quantity = int(i_quantity * ratio)
        no_outlier_quantity = i_quantity - outlier_quantity
        list_remove_i = range(10)
        list_remove_i.remove(i)
        outlier_labels = np.random.choice(list_remove_i, outlier_quantity, replace=True)
        if mixed_images.shape[0] == 0:
            mixed_images = i_images
            mixed_labels = np.concatenate((outlier_labels, np.array([i] * no_outlier_quantity)))
            true_labels = np.concatenate((np.ones_like(outlier_labels), np.zeros(no_outlier_quantity)))
        else:
            mixed_images = np.concatenate((mixed_images, i_images))
            mixed_labels = np.concatenate((mixed_labels, outlier_labels))
            mixed_labels = np.concatenate((mixed_labels, np.array([i] * no_outlier_quantity)))
            true_labels = np.concatenate((true_labels, np.ones_like(outlier_labels)))
            true_labels = np.concatenate((true_labels, np.zeros(no_outlier_quantity)))
    print(np.sum(true_labels))
    idx = np.random.permutation(len(mixed_labels))
    mixed_images, mixed_labels, true_labels = mixed_images[idx], mixed_labels[idx], true_labels[idx]
    return mixed_images, mixed_labels.astype(np.uint8), true_labels.astype(np.uint8)


def main():
    # load mnist
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    for ratio in [k / 10.0 for k in range(10)]:
        dir_name = 'mnist_outlier_' + str(ratio)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        mixed_images, mixed_labels, true_labels = generate_outlier_mnist(mnist=mnist, ratio=ratio)
        images_name = dir_name + '/images-' + str(ratio) + '-ubyte'
        labels_outlier = dir_name + '/labels-outlier-' + str(ratio) + '-ubyte'
        labels_true = dir_name + '/labels-true-' + str(ratio) + '-ubyte'
        print("start write")

        # write images
        header = np.array([0x0803, len(mixed_images), 28, 28], dtype='>i4')
        with open(images_name, 'wb') as f:
            f.write(header.tobytes())
            f.write(float_to_int(mixed_images).tobytes())

        # write labels
        header = np.array([0x0801, len(mixed_labels)], dtype='>i4')
        with open(labels_outlier, 'wb') as f:
            f.write(header.tobytes())
            f.write(mixed_labels.tobytes())

        header = np.array([0x0805, len(true_labels)], dtype='>i4')
        with open(labels_true, 'wb') as f:
            f.write(header.tobytes())
            f.write(true_labels.tobytes())

    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    header = np.array([0x0803, len(mnist.test.images), 28, 28], dtype='>i4')
    with open('test-images-ubyte', 'wb') as f:
        f.write(header.tobytes())
        f.write(float_to_int(mnist.test.images).tobytes())
    header = np.array([0x0801, len(mnist.test.labels)], dtype='>i4')
    with open('test-labels-ubyte', 'wb') as f:
        f.write(header.tobytes())
        f.write(mnist.test.labels.astype(np.uint8).tobytes())
    header = np.array([0x0803, len(mnist.validation.images), 28, 28], dtype='>i4')
    with open('validation-images-ubyte', 'wb') as f:
        f.write(header.tobytes())
        f.write(float_to_int(mnist.validation.images).tobytes())
    header = np.array([0x0801, len(mnist.validation.labels)], dtype='>i4')
    with open('validation-labels-ubyte', 'wb') as f:
        f.write(header.tobytes())
        f.write(mnist.validation.labels.astype(np.uint8).tobytes())

if __name__ == '__main__':
    main()
