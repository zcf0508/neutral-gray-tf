import time
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from neutral_gray.images import ImageLoderV2
from neutral_gray.model import GRAY
from neutral_gray.config import EPOCHS, STEPS

model = GRAY().getModel()
# pre_model = tf.keras.models.load_model('', compile=False)
# model.set_weights(pre_model.get_weights())

model.compile(
      optimizer=tf.keras.optimizers.Adam(1e-3), # 用于微调
      # optimizer=tf.keras.optimizers.SGD(0.02, momentum=0.5, nesterov=True), # 用于预训练
      metrics=["accuracy"],
  )

train_images_data = ImageLoderV2("./data/0", "./data/1").load_data()
test_images_data = ImageLoderV2("./data_test/0", "./data_test/1").load_data()
model_time = str(time.time())

def lr_decay_callback():
    """ 变化学习率
    """

    # lr decay function
    def lr_decay(epoch):
      initial_lrate = 2e-2
      drop = 0.5
      epochs_drop = 10.0
      lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
      return lrate

    # lr schedule callback
    return tf.keras.callbacks.LearningRateScheduler(lr_decay, verbose=False)

history = model.fit(
  train_images_data,
  steps_per_epoch=STEPS,
  validation_data=test_images_data,
  validation_steps=2,
  epochs=EPOCHS,
  callbacks=[
    tf.keras.callbacks.TensorBoard(log_dir='./logs/' + model_time),  # tensorboard --logdir=./logs/(model_time)
    tf.keras.callbacks.EarlyStopping(
          monitor='val_loss', min_delta=0.0001, patience=15, restore_best_weights=True, verbose=1),
    lr_decay_callback(),
  ]
)

# 显示训练曲线
def show_histroy(history):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  # epochs_range = range(EPOCHS)
  epochs_range = range(len(loss))

  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show()

model.save('./models/gray-model-' + model_time)

show_histroy(history)
