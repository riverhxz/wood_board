"""YOLO_v3 Model Defined in Keras."""

from functools import wraps

import sys
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D, Lambda, \
    Dense
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from yolo3.utils import compose
import pandas as pd
import numpy as np


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])
    return x


def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def update_prototype(feature_map: tf.Tensor, y: tf.Tensor, center: tf.Tensor, ld=5e-2):
    v_feature_map = feature_map.eval()
    v_y = np.argmax(y.eval(), axis=1)
    v_center = center.eval()

    df = pd.DataFrame({'y': v_y,
                       "feature_map": [v_feature_map[i] for i in range(v_feature_map.shape[0])]})
    df2 = df.groupby("y")["feature_map"].apply(np.mean)
    update_ops = []
    for i in df2.index:
        update_ops.append(tf.assign(center[i], (1 - ld) * v_center[i] + ld * df2.loc[i]))

    return update_ops


def test_update_prototype():
    with tf.Session().as_default() as sess:
        v_feature_map = tf.constant(np.array([[0., 1.], [1., 2.], [3., 4.]]), dtype=tf.float32)
        v_y = tf.constant([[0, 0, 1], [0, 1, 0], [0, 1, 0]])
        center = tf.Variable([[0, 0], [2, 3], [0, 0]], dtype=tf.float32)
        expectd = tf.constant([[0, 0], [2, 3], [0, 0.5]], dtype=tf.float32)
        # sess.run(tf.init)
        update_ops = update_prototype(
            v_feature_map
            , v_y
            , center
            , 0.5
        )
        sess.run(update_ops)
        assert (tf.cast(expectd, tf.float32) - center).eval().sum() == 0


def distance(x, num_anchor, num_filters, class_center, num_classes):
    """
    Distance between anchor embeding and embeding center
    :param x:
    :param num_anchor:
    :param num_filters:
    :param class_center:
    :param num_classes:
    :return: output [batch, w, h, num_archor*(num_class + 1(backgroud))]
    """
    sp = tf.shape(x)
    x = tf.reshape(x, [sp[0], sp[1], sp[2], num_anchor, 1, num_filters])
    x_diff = x - class_center
    x2 = tf.einsum('abcdef,abcdef->abcde', x_diff, x_diff) / num_filters
    x2 = tf.reshape(x2, [sp[0], sp[1], sp[2], num_anchor * (num_classes + 1)])

    return x2


def make_last_layers(x, num_filters, out_filters, layer, num_classes, num_anchor=3):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    center_dims = num_filters * 2
    class_center = tf.Variable(tf.random.normal([(num_classes + 1), center_dims]), False, name="class_center_"+str(layer))
    class_center = tf.reshape(class_center, [1, 1, 1, (num_classes + 1), center_dims])

    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    gap = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    y = DarknetConv2D(out_filters, (1, 1))(gap)

    cluster_branch = DarknetConv2D_BN_Leaky(center_dims * num_anchor, (1, 1))(x)
    _distance = lambda x: distance(x, num_anchor=num_anchor
    , num_filters=center_dims, class_center=class_center, num_classes=num_classes)
    z = Lambda(_distance, name="yolo_last_" + str(num_filters))(cluster_branch)
    y = Concatenate()([y, z])
    return x, y, gap


def test_dist():
    a = tf.constant([1, 2])
    a = tf.reshape(a, [1, 1, 1, 2])
    b = distance(a, num_anchor=1,
                 num_filters=2,
                 class_center=[[1, 3], [2, 3]],
                 num_classes=2)
    assert tf.squeeze(b) == [tf.sqrt(0.5), 1.0]


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""

    darknet = Model(inputs, darknet_body(inputs))
    x, y1, gap1 = make_last_layers(darknet.output, 512, num_anchors * (num_classes + 5), 1, num_classes)

    x = compose(
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2, gap2 = make_last_layers(x, 256, num_anchors * (num_classes + 5), 2,num_classes)

    x = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3, gap3 = make_last_layers(x, 128, num_anchors * (num_classes + 5), 3,num_classes)

    return Model(inputs, [y1, y2, y3, gap1, gap2, gap3])


def tiny_yolo_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    x1 = compose(
        DarknetConv2D_BN_Leaky(16, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(32, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(64, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(128, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(256, (3, 3)))(inputs)
    x2 = compose(
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
        DarknetConv2D_BN_Leaky(1024, (3, 3)),
        DarknetConv2D_BN_Leaky(256, (1, 1)))(x1)
    y1 = compose(
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        DarknetConv2D(num_anchors * (num_classes + 5), (1, 1)))(x2)

    x2 = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x2)
    y2 = compose(
        Concatenate(),
        DarknetConv2D_BN_Leaky(256, (3, 3)),
        DarknetConv2D(num_anchors * (num_classes + 5), (1, 1)))([x2, x1])

    return Model(inputs, [y1, y2])


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5 + (num_classes + 1)])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:7])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
                                                                anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval_2(yolo_outputs,
                anchors,
                num_classes,
                image_shape,
                max_boxes=20,
                score_threshold=.6,
                iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]  # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]  # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors) // 3  # default setting
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
    # layer分三层，  每层取对应的anchor
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1
            # print(b, j, i, k, l)
    return y_true


def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss(args, anchors, num_classes,  ignore_thresh=.5, print_loss=True, summary_loss=True):
    '''Return yolo_loss tensor

     Parameters
     ----------
     yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
     y_true: list of array, the output of preprocess_true_boxes
     anchors: array, shape=(N, 2), wh
     num_classes: integer
     ignore_thresh: float, the iou threshold whether to ignore object confidence loss

     Returns
     -------
     loss: tensor, shape=(1,)

     '''
    loss = _yolo_loss(args, anchors, num_classes, ignore_thresh, print_loss, summary_loss)

    return loss


def _yolo_loss(args, anchors, num_classes,  ignore_thresh=.5, print_loss=False, summary_loss=True):
    num_layers = len(anchors) // 3  # default setting
    yolo_outputs = args[:num_layers]
    gaps = args[num_layers:num_layers * 2]

    y_true = args[num_layers * 2:]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0]  # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
                                                     anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])
        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b + 1, ignore_mask

        from tensorflow.python.ops import control_flow_ops
        _, ignore_mask = control_flow_ops.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                       from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + \
                          (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                    from_logits=True) * ignore_mask

        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:5+num_classes], from_logits=True)

        # extend_true_class_probs = tf.concat(
        #     [true_class_probs, 1 - tf.reduce_sum(true_class_probs, axis=4, keepdims=True)], axis=4)

        num_pos = tf.reduce_sum(true_class_probs)
        pos = tf.reduce_sum(true_class_probs, axis=4, )
        all = tf.cast(tf.reduce_prod(tf.shape(true_class_probs)), K.dtype(pos))
        weight = 1 - (1 - num_pos / all) * pos
        # multi_class_loss = weight * tf.nn.softmax_cross_entropy_with_logits(
        #     labels=extend_true_class_probs
        #     , logits=raw_pred[..., 7:]
        # )

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        # multi_class_loss = K.sum(multi_class_loss) / mf
        # TODO
        loss += (xy_loss + wh_loss + confidence_loss
                 + class_loss
                 # + multi_class_loss
                 )

    # def _get_streaming_metrics(_label, _prediction, num_classes, name):
    #
    #     label = tf.reshape(_label, [-1])
    #     prediction = tf.reshape(_prediction, [-1])
    #     with tf.name_scope(name):
    #         # the streaming accuracy (lookup and update tensors)
    #         accuracy, accuracy_update = tf.metrics.accuracy(label, prediction,
    #                                                         name='accuracy')
    #         # Compute a per-batch confusion
    #         batch_confusion = tf.confusion_matrix(label, prediction,
    #                                               num_classes=num_classes,
    #                                               name='batch_confusion')
    #         # Create an accumulator variable to hold the counts
    #         confusion = tf.Variable(tf.zeros([num_classes, num_classes],
    #                                          dtype=tf.int32),
    #                                 name='confusion')
    #         # Create the update op for doing a "+=" accumulation on the batch
    #         confusion_update = confusion.assign(confusion + batch_confusion)
    #         # Cast counts to float so tf.summary.image renormalizes to [0,255]
    #         confusion_image = tf.reshape(tf.cast(confusion, tf.float32),
    #                                      [1, num_classes, num_classes, 1])
    #         # Combine streaming accuracy and confusion matrix updates in one op
    #         test_op = tf.group(accuracy_update, confusion_update)
    #
    #         # tf.summary.image('confusion', confusion_image)
    #         # tf.summary.scalar('accuracy', accuracy)
    #
    #     return {'accuracy', accuracy}
    #
    # _get_streaming_metrics(true_class_probs, raw_pred[..., 5:7], num_classes, "classify_branch")
    # _get_streaming_metrics(extend_true_class_probs, raw_pred[..., 7:], num_classes + 1, "metric_branch")

    if print_loss:
        print_op = tf.print("[xy_loss, wh_loss, confidence_loss, class_loss]",
                            *[xy_loss, wh_loss, confidence_loss, class_loss],
                            output_stream=sys.stdout)
        with tf.control_dependencies([print_op]):
            loss = loss * 1.0

    return loss
