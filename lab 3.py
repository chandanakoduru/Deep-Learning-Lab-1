import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([0, 1, 1, 0])

print("Inputs:\n", X)
print("Outputs:\n", y)

mlp_model = models.Sequential([
    layers.Dense(4, activation='relu', input_shape=(2,)),  
    layers.Dense(1, activation='sigmoid')                  
])

mlp_model.summary()
mlp_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
print("\n--- Training MLP for XOR ---")

history = mlp_model.fit(
    X,
    y,
    epochs=200,
    verbose=0
)

print("Training Completed!")

loss, acc = mlp_model.evaluate(X, y, verbose=0)
print(f"\nXOR Accuracy: {acc * 100:.2f}%")
predictions = mlp_model.predict(X)

print("\nPredictions:")
for i in range(len(X)):
    print(f"Input: {X[i]} → Output: {predictions[i][0]:.4f}")


final_preds = (predictions > 0.5).astype(int)

print("\nFinal Binary Output:")
for i in range(len(X)):
    print(f"Input: {X[i]} → Predicted: {final_preds[i][0]}")

plt.figure()

plt.plot(history.history['accuracy'], label='Accuracy')
plt.title("XOR Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.show()
plt.figure()

plt.plot(history.history['loss'], label='Loss')
plt.title("XOR Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.show()