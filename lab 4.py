import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

print("Training Images Shape:", X_train.shape)
print("Testing Images Shape:", X_test.shape)

dnn_model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),

    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),  
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),

    layers.Dense(128, activation='relu'),

    layers.Dense(10, activation='softmax')
])

dnn_model.summary()

dnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- Training Deep Neural Network (DNN) ---")

history = dnn_model.fit(
    X_train,
    y_train,
    epochs=10,               
    validation_split=0.1
)

print("\n--- Evaluating DNN ---")

test_loss, test_acc = dnn_model.evaluate(X_test, y_test)

print(f"\nDNN Test Accuracy: {test_acc * 100:.2f}%")

sample = X_test[0]
prediction = dnn_model.predict(sample.reshape(1, 28, 28))

predicted_digit = np.argmax(prediction)

print("Actual Digit:", y_test[0])
print("Predicted Digit:", predicted_digit)

plt.imshow(sample, cmap='gray')
plt.title(f"Predicted: {predicted_digit}")
plt.axis('off')
plt.show()

plt.figure()

plt.plot(history.history['accuracy'], marker='o', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], marker='o', label='Validation Accuracy')

plt.title('DNN Accuracy vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.show()

plt.figure()

plt.plot(history.history['loss'], marker='o', label='Training Loss')
plt.plot(history.history['val_loss'], marker='o', label='Validation Loss')

plt.title('DNN Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.show()

y_pred = dnn_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')

plt.title("Confusion Matrix")
plt.show()