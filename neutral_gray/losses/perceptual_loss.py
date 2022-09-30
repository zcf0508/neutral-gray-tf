# https://www.tensorflow.org/tutorials/generative/style_transfer?hl=zh-cn#%E9%A3%8E%E6%A0%BC%E8%AE%A1%E7%AE%97
import tensorflow as tf

keras = tf.keras

content_layers = ["block5_conv2"]

style_layers = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def vgg_layers(layer_names):
    """Creates a VGG model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on ImageNet data
    vgg19 = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg19.trainable = False

    outputs = [vgg19.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg19.input], outputs)
    return model


style_extractor = vgg_layers(style_layers)
content_extractor = vgg_layers(content_layers)

vgg = vgg_layers(style_layers + content_layers)

style_weight = 1e-2
content_weight = 1e4


def perceptual_loss(gen_output, target):
    outputs1 = vgg(gen_output)
    outputs2 = vgg(target)
    style_outputs1, content_outputs1 = (
        outputs1[:num_style_layers],
        outputs1[num_style_layers:],
    )
    style_outputs2, content_outputs2 = (
        outputs2[:num_style_layers],
        outputs2[num_style_layers:],
    )

    content_dict1 = {
        content_name: value
        for content_name, value in zip(content_layers, content_outputs1)
    }
    content_dict2 = {
        content_name: value
        for content_name, value in zip(content_layers, content_outputs2)
    }

    style_dict1 = {
        style_name: value for style_name, value in zip(style_layers, style_outputs1)
    }

    style_dict2 = {
        style_name: value for style_name, value in zip(style_layers, style_outputs2)
    }

    style_loss = tf.add_n(
        [
            tf.reduce_mean((style_dict1[name] - style_dict2[name]) ** 2)
            for name in style_dict1.keys()
        ]
    )
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n(
        [
            tf.reduce_mean((content_dict1[name] - content_dict2[name]) ** 2)
            for name in content_dict1.keys()
        ]
    )
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss
