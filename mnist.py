import tensorflow as tf
from PIL import Image
import numpy as np
import time
import os

image_path = "test.png"  # Set this to the path of your image f

# ---------------------------
# FOR TRAINING PURPOSES ONLY
# ---------------------------

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test, verbose=2)


# ---------------------------
# FOR TESTING PURPOSES ONLY
# ---------------------------

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  

    img = img.resize((28, 28), Image.ANTIALIAS)

    img_array = np.array(img)
    img_array = img_array / 255.0

    img_array = 1 - img_array

    img_array = img_array.reshape(1, 28, 28)

    return img_array

def predict_digit(model, img_array):
    logits = model.predict(img_array)
    probabilities = tf.nn.softmax(logits).numpy()
    predicted_digit = np.argmax(probabilities)
    return predicted_digit

while True:
    # Check if the image exists
    if os.path.exists(image_path):
        # Preprocess the image
        img_array = preprocess_image(image_path)

        # Predict the digit
        digit = predict_digit(model, img_array)
        print(f"Predicted Digit: {digit}")

        # Wait for 5 seconds
        time.sleep(5)