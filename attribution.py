import numpy as np
import tensorflow as tf
import saliency.core as saliency

def model_fn(images, call_model_args, expected_keys=None):
    target_class_idx = call_model_args['class']
    model = call_model_args['model']
    images = tf.convert_to_tensor(images)

    with tf.GradientTape() as tape:
        if expected_keys==[saliency.base.INPUT_OUTPUT_GRADIENTS]:
            tape.watch(images)
            output = model(images)
            output = output[:,target_class_idx]
            gradients = np.array(tape.gradient(output, images))
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
        else:
            conv, output = model(images)
            gradients = np.array(tape.gradient(output, conv))
            return {saliency.base.CONVOLUTION_LAYER_VALUES: conv,
                    saliency.base.CONVOLUTION_OUTPUT_GRADIENTS: gradients}

# gradient - saliency map
def gradient(model, img):
    pred = model(np.array([img]))
    pred_cls = np.argmax(pred[0])
    args = {'model': model, 'class': pred_cls}

    grad = saliency.GradientSaliency()
    attr = grad.GetMask(img, model_fn, args)
    attr = saliency.VisualizeImageGrayscale(attr, percentile=100)

    return tf.reshape(attr, (*attr.shape, 1))

# smoothgrad
def smoothgrad(model, img):
    pred = model(np.array([img]))
    pred_cls = np.argmax(pred[0])
    args = {'model': model, 'class': pred_cls}

    grad = saliency.GradientSaliency()
    attr = grad.GetSmoothedMask(img, model_fn, args)
    attr = saliency.VisualizeImageGrayscale(attr, percentile=100)

    return tf.reshape(attr, (*attr.shape, 1))

# vanilla gradient
def vg(model, img):
    pred = model(np.array([img]))
    pred_cls = np.argmax(pred[0])
    args = {'model': model, 'class': pred_cls}

    grad = saliency.GradientSaliency()
    attr = grad.GetMask(img, model_fn, args)

    return attr

# integrated gradients
def ig(model, img):
    pred = model(np.array([img]))
    pred_cls = np.argmax(pred[0])
    args = {'model': model, 'class': pred_cls}

    baseline = np.zeros(img.shape)
    ig = saliency.IntegratedGradients()
    attr = ig.GetMask(img, model_fn, args, x_steps=25, x_baseline=baseline, batch_size=20)
    attr = saliency.VisualizeImageGrayscale(attr, percentile=100)

    return tf.reshape(attr, (*attr.shape, 1))

# smoothed integrated gradients
def smoothedig(model, img):
    pred = model(np.array([img]))
    pred_cls = np.argmax(pred[0])
    args = {'model': model, 'class': pred_cls}

    baseline = np.zeros(img.shape)
    ig = saliency.IntegratedGradients()
    attr = ig.GetSmoothedMask(img, model_fn, args, x_steps=25, x_baseline=baseline, batch_size=20)
    attr = saliency.VisualizeImageGrayscale(attr, percentile=100)

    return tf.reshape(attr, (*attr.shape, 1))

# guided integrated gradients
def guidedig(model, img):
    pred = model(np.array([img]))
    pred_cls = np.argmax(pred[0])
    args = {'model': model, 'class': pred_cls}

    baseline = np.zeros(img.shape)
    guidedig = saliency.GuidedIG()
    attr = guidedig.GetSmoothedMask(img, model_fn, args, x_steps=25, x_baseline=baseline, max_dist=1.0, fraction=0.5)
    attr = saliency.VisualizeImageGrayscale(attr, percentile=100)

    return tf.reshape(attr, (*attr.shape, 1))