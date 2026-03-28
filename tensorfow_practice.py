from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape, train_labels.shape)
print(train_labels[:10])


train_images = train_images.reshape((train_images.shape[0], 28*28)).astype('float32')/255
test_images = test_images.reshape((test_images.shape[0], 28*28)).astype('float32')/255

model = Sequential([
    Dense(128, activation='relu', input_shape= (28*28,)),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f"Test accuracy: {test_acc}")



"""
MNIST database (Modified National Institute of Standards and Technology) is a foundational dataset for machine learning, featuring 70,000 grayscale, pixel images of handwritten digits (0–9). 
Dense expects 1D input, so we flatten the 28x28 images to 784 pixels. We normalize the pixel values to be between 0 and 1.
Rectified Linear Unit (ReLU) activation (f(x)=max(0,x))
Softmax activation (f(x)=e^x / Σe^x) to output probabilities for each class. Each value is between 0 to 1 and the sum of all values equals 1.
Adam optimization - adaptive learning rates for each parameter, resulting in faster convergence and less required tuning
"""