# CUDA_VISIBLE_DEVICES=0,1,2,3 python make_encoder_celeba.py --num_exp=21 --iteration=12000 --image_size=512 --nlat=512

# python svm_celeba_classification.py --name=mmds --classifer=svm --num_exp=21 --image_size=512 --iteration=10000 --nlat=512

# next_i,next_l,next_t,next_f = sess.run([next_images, next_labels, next_text, next_filenames])
# print(next_i.shape)
# print(next_i.max())
# print(next_i.mean())
# print(next_i.min())
# item_image = torch.from_numpy(next_i)
# item_image = item_image.type(torch.FloatTensor)
# print(item_image.shape)
# print(next_t)
# print(next_l)
# print(next_f)
# print(type(next_t))

import os
import numpy as np
import tensorflow as tf
import torch
import json

imageWidth = 224
imageHeight = 224
imageDepth = 3
resize_min = 256




def _parse_function(example_proto):
    features = {"image": tf.FixedLenFeature([], tf.string, default_value=""),
                "height": tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                "width": tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                "channels": tf.FixedLenFeature([1], tf.int64, default_value=[3]),
                "colorspace": tf.FixedLenFeature([], tf.string, default_value=""),
                "img_format": tf.FixedLenFeature([], tf.string, default_value=""),
                "label": tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                "bbox_xmin": tf.VarLenFeature(tf.float32),
                "bbox_xmax": tf.VarLenFeature(tf.float32),
                "bbox_ymin": tf.VarLenFeature(tf.float32),
                "bbox_ymax": tf.VarLenFeature(tf.float32),
                "text": tf.FixedLenFeature([], tf.string, default_value=""),
                "filename": tf.FixedLenFeature([], tf.string, default_value="")
               }
    parsed_features = tf.parse_single_example(example_proto, features)
    xmin = tf.expand_dims(parsed_features["bbox_xmin"].values, 0)
    xmax = tf.expand_dims(parsed_features["bbox_xmax"].values, 0)
    ymin = tf.expand_dims(parsed_features["bbox_ymin"].values, 0)
    ymax = tf.expand_dims(parsed_features["bbox_ymax"].values, 0)
    bbox = tf.concat(axis=0, values=[ymin, xmin, ymax, xmax])
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])
    height = parsed_features["height"]
    width = parsed_features["width"]
    channels = parsed_features["channels"]
    bbox_begin, bbox_size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.concat(axis=0, values=[height, width, channels]),
        bounding_boxes=bbox,
        min_object_covered=0.1,
        use_image_if_no_bounding_boxes=True)
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.cast(tf.stack([offset_y, offset_x, target_height, target_width]), tf.int32)
    cropped = tf.image.decode_and_crop_jpeg(parsed_features["image"], crop_window, channels=3)
    image_train = tf.image.resize_images(cropped, [imageHeight, imageWidth],
                                         method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
    image_train = tf.cast(image_train, tf.uint8)
    image_train = tf.image.convert_image_dtype(image_train, tf.float32)
    return image_train, parsed_features["label"][0], parsed_features["text"], parsed_features["filename"]


def save_config(config,filename):
    config_content = {}
    for key,value in config.items():
        # if key != 'job' and key != 'ns':
        config_content[key] = value
        # task_content['task_id'] = tasks['task_id']
    fw = open(filename, 'w', encoding='utf-8')
    dic_json = json.dumps(config_content, ensure_ascii=False, indent=4)
    fw.write(dic_json)
    fw.close()

def load_config(config_file):
    f = open(config_file,encoding='utf-8')
    res = f.read()
    config_content = json.loads(res)
    return config_content

if __name__ == '__main__':
    data_path = '/data1/k8sdata/imagenet_data'
    train_files_names = os.listdir('/data1/k8sdata/imagenet_data/train_tf/')
    train_files = ['/data1/k8sdata/imagenet_data/train_tf/' + item for item in train_files_names]
    dataset_train = tf.data.TFRecordDataset(train_files)
    dataset_train = dataset_train.map(_parse_function, num_parallel_calls=4)
    dataset_train = dataset_train.batch(1)
    dataset_train = dataset_train.prefetch(128)
    iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
    train_init_op = iterator.make_initializer(dataset_train)
    next_images, next_labels, next_text, next_filenames = iterator.get_next()
    sess = tf.Session()
    sess.run(train_init_op)

    try:
        class_index = load_config("class_index.json")
        num_index = load_config("num_index.json")
    except Exception as eee:
        class_index = {}
        num_index = {}
        num_index['total'] = 0

    total_index = 0

    while (True):
        try:
            next_i, next_l, next_t, next_f = sess.run([next_images, next_labels, next_text, next_filenames])
            if str(int(next_l[0])) not in class_index.keys():
                num_index[str(int(next_l[0]))] = 0
                class_index[str(int(next_l[0]))] = []
            now_name = '%d_%d.npy' % (next_l[0], num_index[str(int(next_l[0]))])
            np.save(os.path.join(os.path.join(data_path, 'train_np'), now_name), next_i[0])
            total_index += 1
            num_index[str(int(next_l[0]))] += 1
            class_index[str(int(next_l[0]))].append([now_name, str("%s" % next_t[0]).strip()])
            num_index['total'] = total_index
            if total_index % 100 == 0:
                save_config(class_index, "class_index.json")
                save_config(num_index, "num_index.json")
                print(next_i.shape)
                print(next_l[0])
                print(total_index)
        except Exception as ee:
            print(ee)
            break