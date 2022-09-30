import tensorflow as tf

keras = tf.keras


def generator_loss(disc_generated_output, gen_output, target):
    LAMBDA = 100
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = keras.losses.MeanAbsoluteError()(target, gen_output)

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss
