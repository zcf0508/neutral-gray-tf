import time
import matplotlib.pyplot as plt
from neutral_gray.images import ImageLoderV2
from neutral_gray.model import GRAY
from neutral_gray.config import BATCH_SIZE, EPOCHS

model = GRAY().getModel()

train_images_data = ImageLoderV2("./data/0", "./data/1").load_data()
train_images, train_results = next(iter(train_images_data))

test_images_data = ImageLoderV2("./data_test/0", "./data_test/1").load_data()
test_images, test_results = next(iter(test_images_data))

history = model.fit(
  train_images,
  train_results,
  validation_data=(test_images, test_results),
  batch_size=BATCH_SIZE, 
  epochs=EPOCHS,
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

show_histroy(history)

model.save('./models/gray-model-'+str(time.time()))