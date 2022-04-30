import gzip

import numpy
from tensorflow.python.platform import gfile
from tensorflow.python.framework import dtypes

def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

  Args:
    f: A file object that can be passed into a gzip reader.

  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].

  Raises:
    ValueError: If the bytestream does not start with 2051.

  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_labels(f, one_hot=False, num_classes=10):
  """Extract the labels into a 1D uint8 numpy array [index].

  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.

  Returns:
    labels: a 1D uint8 numpy array.

  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels

def extract_if_outlier(f):
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2053:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    return labels

def get_outlier_with_ratio(ratio):
    ratio = int(ratio * 10) / 10.0
    # dir =mnist_outlier_' + str(ratio)
    dir = 'mnist_outlier/mnist_outlier_'+ str(ratio)
    images_gz = dir + '/images-' + str(ratio) + '-ubyte.gz'
    labels_outlier = dir + '/labels-outlier-' + str(ratio) + '-ubyte.gz'
    labels_true = dir + '/labels-true-' + str(ratio) + '-ubyte.gz'
    with gfile.Open(images_gz, 'rb') as f:
      train_images = extract_images(f)
    with gfile.Open(labels_outlier, 'rb') as f:
      train_labels = extract_labels(f, one_hot=True)
    with gfile.Open(labels_outlier, 'rb') as f:
        raw_labels = extract_labels(f, one_hot=False)
    with gfile.Open(labels_true, 'rb') as f:
      train_if_outlier = extract_if_outlier(f)
    return int_to_float(train_images), train_labels, train_if_outlier,raw_labels

def get_test():
    path = 'mnist_outlier/'
    images_gz = 'test-images-ubyte.gz'
    labels_gz = 'test-labels-ubyte.gz'
    with gfile.Open(path+images_gz, 'rb') as f:
      test_images = extract_images(f)
    with gfile.Open(path+labels_gz, 'rb') as f:
      test_labels = extract_labels(f, one_hot=True)
    with gfile.Open(path + labels_gz, 'rb') as f:
      raw_labels = extract_labels(f,one_hot=False)
    return int_to_float(test_images), test_labels,raw_labels


def get_validation():
    path = 'mnist_outlier/'
    images_gz = 'validation-images-ubyte.gz'
    labels_gz = 'validation-labels-ubyte.gz'
    with gfile.Open(path+images_gz, 'rb') as f:
      validation_images = extract_images(f)
    with gfile.Open(path+labels_gz, 'rb') as f:
      validation_labels = extract_labels(f, one_hot=True)
    with gfile.Open(path + labels_gz, 'rb') as f:
      raw_labels = extract_labels(f,one_hot=False)
    return int_to_float(validation_images), validation_labels,raw_labels


def int_to_float(images):
    #imagedata = numpy.zeros_like(images, dtype=numpy.uint8)
    images = images.astype(numpy.float32)
    return numpy.multiply(images, 1.0 / 255.0)


class MnistOutlier:
    def __init__(self, outlier_ratio):
        self.outlier_ratio = outlier_ratio
        self.train_images, self.train_labels, self.if_outlier,self.train_raw = get_outlier_with_ratio(outlier_ratio)
        self.test_images, self.test_labels,self.test_raw = get_test()
        self.validation_images, self.validation_labels,self.validation_raw = get_validation()
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._num_examples = self.train_images.shape[0]

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self.train_images[start:self._num_examples]
            labels_rest_part = self.train_labels[start:self._num_examples]
            raws_rest_part = self.train_raw[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self.train_images = self.train_images[perm]
                self.train_labels = self.train_labels[perm]
                self.train_raw = self.train_raw[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self.train_images[start:end]
            labels_new_part = self.train_labels[start:end]
            raws_new_part = self.train_raw[start:end]
            return numpy.concatenate((images_rest_part, images_new_part), axis=0), numpy.concatenate(
                (labels_rest_part, labels_new_part), axis=0),numpy.concatenate((raws_rest_part, raws_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self.train_images[start:end], self.train_labels[start:end],self.train_raw[start:end]

def main():
    outlier = MnistOutlier(0.1)
    print(outlier.train_images[0].transpose(2,0,1).shape)

# main()