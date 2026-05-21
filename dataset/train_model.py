import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

train_dir = "dataset/preprocess/train"
val_dir = "dataset/preprocess/val"
test_dir = "dataset/preprocess/test"

results_dir = "dataset/results"
os.makedirs(results_dir, exist_ok=True)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

num_classes = train_data.num_classes
print("Classes:", train_data.class_indices)

#Build model
model = models.Sequential([

    layers.Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE,IMG_SIZE,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(num_classes, activation="softmax")

])

model.summary()

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

#Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

plt.figure()

plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Training vs Validation Accuracy")
plt.legend()

plt.savefig(os.path.join(results_dir,"accuracy_graph.png"))
plt.close()

plt.figure()

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Training vs Validation Loss")
plt.legend()

plt.savefig(os.path.join(results_dir,"loss_graph.png"))
plt.close()

print("Graphs saved!")

test_loss, test_acc = model.evaluate(test_data)

print("Test Accuracy:", test_acc)

predictions = model.predict(test_data)

y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig(os.path.join(results_dir,"confusion_matrix.png"))
plt.close()

print("Confusion matrix saved!")

class_names = list(test_data.class_indices.keys())

report = classification_report(y_true, y_pred, target_names=class_names)

with open(os.path.join(results_dir,"classification_report.txt"),"w") as f:
    f.write(report)

print("Classification report saved!")

model.save("model/model.h5")

print("Model saved as model.h5")

print("Training Completed Successfully!")