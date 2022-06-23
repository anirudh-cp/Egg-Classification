# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from:
     
# Cats and Dogs
# O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
# IEEE Conference on Computer Vision and Pattern Recognition, 2012


from fileinput import filename
import hashlib
import io
import logging
import os
import random
import re

import PIL.Image
from traitlets import default
import tensorflow.compat.v1 as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.compat.v1.flags
flags.DEFINE_string('data_dir', 'Images', 'Images Directory')
flags.DEFINE_string('output_dir', 'TFR', 'Output Directory')
flags.DEFINE_string('label_map_path', 'label_map.pb', 'Label map path')
FLAGS = flags.FLAGS


def dict_to_tf_example(data, label_map_dict, image_subdirectory, exampleName, classNames):
    """Convert YOLO format derived nested list to tf.Example proto. 

    Args:
      data: Nested list bounding box info for a single image
      label_map_dict: A map from string label names to integers ids.
      image_subdirectory: Directory holding the actual image data.

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """

    filename = exampleName.split('\\')[-1]
    img_path = exampleName + '.png'

    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    width, height = image.size

    if image.format != 'PNG':
        raise ValueError('Image format not PNG')

    key = hashlib.sha256(encoded_jpg).hexdigest()

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []

    for obj in data:
        x_center = obj[1]
        y_center = obj[2]
        box_wt = obj[3]
        box_ht = obj[4]

        xmin.append(x_center - box_wt/2)
        xmax.append(x_center + box_wt/2)
        ymin.append(y_center - box_ht/2)
        ymax.append(y_center + box_ht/2)
        
        class_name = classNames[int(obj[0])]
        classes_text.append(class_name.encode('utf8'))
        classes.append(label_map_dict[class_name])


    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            filename.encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return example


def create_tf_record(output_filename, label_map_dict, image_dir, examples, classes):
    """Creates a TFRecord file from examples.

    Args:
      output_filename: Path to where output file is saved.
      label_map_dict: The label map dictionary.
      image_dir: Directory where image files are stored.
      examples: Examples to parse and save to tf record.
    """

    writer = tf.python_io.TFRecordWriter(output_filename)
    for idx, example in enumerate(examples):

        path = example + '.txt'

        data = []
        if not os.path.exists(path):
            logging.warning(f'Could not find {path}, ignoring example.')
            continue
        with tf.gfile.GFile(path, 'r') as file:
            for line in file:
                data.append([float(x) for x in line.split()] )

        tf_example = dict_to_tf_example(data, label_map_dict, image_dir, example, classes)
        writer.write(tf_example.SerializeToString())

    writer.close()


def main(_):
    data_dir = FLAGS.data_dir
    label_map_dict = label_map_util.get_label_map_dict(os.path.join(FLAGS.output_dir, FLAGS.label_map_path))

    image_dir = os.path.join(data_dir, '/')
    annotations_dir = os.path.join(data_dir, '/')

    classes = []
    with open(os.path.join(data_dir, 'classes.txt'), 'r') as file:
        classes.extend([x.strip() for x in file.readlines()])

    # Perform split of data
    fileNames = [os.path.join(data_dir, f'image{index}') for index in range(1, 251)]
    print(fileNames[0:5])
    
    random.shuffle(fileNames)
    num_train = int(0.7 * len(fileNames))
    train_examples = fileNames[:num_train]
    val_examples = fileNames[num_train:]

    train_output_path = os.path.join(FLAGS.output_dir, 'train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'val.record')
    create_tf_record(train_output_path, label_map_dict, image_dir, train_examples, classes)
    create_tf_record(val_output_path, label_map_dict, image_dir, val_examples, classes)


if __name__ == '__main__':
    tf.compat.v1.app.run()
