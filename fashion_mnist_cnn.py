import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix


# 1. LOAD THE FASHION-MNIST DATASET
# -------------------------------------------------
# Fashion-MNIST is built into Keras, so it will download automatically
print("Loading Fashion-MNIST dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

print("Training data shape:", x_train.shape)  # (60000, 28, 28)
print("Test data shape:", x_test.shape)      # (10000, 28, 28)


# 2. DEFINE CLASS NAMES FOR BETTER INTERPRETATION
# -------------------------------------------------
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# 3. VISUALIZE SOME SAMPLE IMAGES
# -------------------------------------------------
def show_sample_images(images, labels, class_names):
    plt.figure(figsize=(8, 8))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap="gray")
        plt.xlabel(class_names[labels[i]])
    plt.tight_layout()
    plt.show()


print("Showing sample images from training set...")
show_sample_images(x_train, y_train, class_names)


# 4. PREPROCESSING (NORMALIZATION + RESHAPE)
# -------------------------------------------------
# Scale pixel values from [0, 255] to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension for CNN: (28, 28) -> (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("New training data shape:", x_train.shape)  # (60000, 28, 28, 1)
print("New test data shape:", x_test.shape)      # (10000, 28, 28, 1)


# 5. BUILD THE CNN MODEL
# -------------------------------------------------
def create_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),  # helps to reduce overfitting
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


model = create_model()
model.summary()


# 6. TRAIN THE MODEL
# -------------------------------------------------
EPOCHS = 10
BATCH_SIZE = 64

print("Training the model...")
history = model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,  # 10% of training used as validation
    verbose=2
)


# 7. EVALUATE THE MODEL
# -------------------------------------------------
print("Evaluating on test data...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")


# 8. PLOT TRAINING & VALIDATION CURVES
# -------------------------------------------------
def plot_training_curves(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    # Accuracy
    plt.figure()
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.show()

    # Loss
    plt.figure()
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()


print("Plotting training curves...")
plot_training_curves(history)


# 9. CONFUSION MATRIX & CLASSIFICATION REPORT
# -------------------------------------------------
print("Generating predictions for evaluation metrics...")
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)


# 10. SAVE THE MODEL
# -------------------------------------------------
model.save("fashion_mnist_cnn_model.h5")
print("Model saved as 'fashion_mnist_cnn_model.h5'")


# 11. PREDICT ON A SINGLE IMAGE
# -------------------------------------------------
def predict_single_image(index=0):
    """
    Predicts and shows a single image from the test set.
    """
    img = x_test[index]
    true_label = y_test[index]

    # Model expects batch dimension
    img_batch = np.expand_dims(img, axis=0)
    prediction = model.predict(img_batch)
    predicted_label = np.argmax(prediction[0])

    plt.figure()
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(
        f"True: {class_names[true_label]} | Predicted: {class_names[predicted_label]}"
    )
    plt.axis("off")
    plt.show()


print("Showing prediction for one test image...")
predict_single_image(index=0)

print("Done!")
