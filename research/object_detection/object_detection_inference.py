import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import time

from distutils.version import StrictVersion

sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

from utils import label_map_util
from utils import visualization_utils as vis_util


class ObjectDetect(object):
    def __init__(self, graph_path):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            self.tensor_dict = tensor_dict
            self.image_tensor = image_tensor
            self.sess = tf.Session()

    def predict(self, image):
        # Run inference
        output_dict = self.sess.run(self.tensor_dict, feed_dict={self.image_tensor: image})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict


def inference_show(detector, label_map, image_path):
    cv2.namedWindow('pic', 0)
    cv2.resizeWindow('pic', 1000, 1000)

    num = 0
    num_all = 0
    time_all = 0
    for file_name in os.listdir(image_path):
        if file_name.endswith('jpg'):
            num_all += 1
            print(file_name)
            image_file = os.path.join(image_path, file_name)
            image = cv2.imread(image_file)
            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            start = time.time()
            output_dict = detector.predict(image_np_expanded)
            end = time.time()
            print('time:', end - start)
            time_all = time_all + end - start

            box = output_dict['detection_boxes'][0]
            score = output_dict['detection_scores'][0]
            label = output_dict['detection_classes'][0]

            if score < 2:
                # Visualization of the results of a detection.
                num += 1
                print(box, label, score)
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    label_map,
                    instance_masks=output_dict.get('detection_masks'),
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=1,
                    min_score_thresh=0.0,
                    line_thickness=8)
                image_cv2 = cv2.cvtColor(np.array(image_np), cv2.COLOR_RGB2BGR)
                cv2.imshow('pic', image_cv2)
                key = cv2.waitKey(0)
                if key == 27:
                    break

    print(num_all, num, time_all / num_all)


label_dict = {1:0, 2:90, 3:180, 4:270}


def rotate(image, angle):
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the angle to rotate clockwise),
    # then grab the sine and cosine (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image_rotated = cv2.warpAffine(image, M, (nW, nH))

    return image_rotated


def inference_path(detector, image_path, out_path, threshold_score):
    num = 0
    num_all = 0
    time_all = 0
    for file_name in os.listdir(image_path):
        if file_name.endswith('jpg'):
            num_all += 1
            print(file_name)
            image_file = os.path.join(image_path, file_name)
            image = cv2.imread(image_file)
            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            start = time.time()
            output_dict = detector.predict(image_np_expanded)
            end = time.time()
            print('time:', end - start)
            time_all = time_all + end - start

            box = output_dict['detection_boxes'][0]
            score = output_dict['detection_scores'][0]
            label = output_dict['detection_classes'][0]

            if score > threshold_score:
                num += 1
                width = image.shape[1]
                height = image.shape[0]
                y_min = int(box[0] * height)
                x_min = int(box[1] * width)
                y_max = int(box[2] * height)
                x_max = int(box[3] * width)
                image_cut = image[y_min: y_max, x_min: x_max]
                angle = label_dict[label]
                image_rotate = rotate(image_cut, 360 - angle)
                out_file = os.path.join(out_path, file_name)
                cv2.imwrite(out_file, image_rotate)
            else:
                print('no object detected:', file_name)
    print(num_all, num, time_all / num_all)


if __name__ == '__main__':
    graph_path = '/Users/gaoxin/models/ssd_mobilenet_v2_coco/exported_graph/frozen_inference_graph.pb'
    label_path = '/Users/gaoxin/models/ssd_mobilenet_v2_coco/idcard_label_map.pbtxt'
    image_path = '/Users/gaoxin/data/idcard/img_train/object_detection/data_idcard_front_eval'
    out_path = '/Users/gaoxin/data/idcard/img_eval_detected'

    detector = ObjectDetect(graph_path)
    category_index = label_map_util.create_category_index_from_labelmap(label_path, use_display_name=True)
    # inference_show(detector, category_index, image_path)
    inference_path(detector, image_path, out_path, 0.6)
