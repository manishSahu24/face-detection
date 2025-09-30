import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# -------- CONFIGURATION --------
DATASET_DIR = "dataset"      # dataset folder ka naam
MODEL_OUT = "mask_detector.h5"
INIT_LR = 1e-4   # learning rate
EPOCHS = 5       # training rounds (zyada bhi kar sakte ho jaise 20)
BS = 32          # batch size
IMG_SIZE = (224, 224)

# -------- DATA PREPARATION --------
train_aug = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.20
)

train_gen = train_aug.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BS,
    class_mode="binary",
    subset="training"
)

val_gen = train_aug.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BS,
    class_mode="binary",
    subset="validation"
)

# -------- CREATE MODEL --------
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)))

for layer in baseModel.layers:
    layer.trainable = False

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten()(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(1, activation="sigmoid")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

opt = Adam(learning_rate=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

checkpoint = ModelCheckpoint(MODEL_OUT, monitor="val_loss", save_best_only=True, verbose=1)

# -------- TRAINING START --------
H = model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // BS,
    validation_data=val_gen,
    validation_steps=val_gen.samples // BS,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

# -------- SAVE FINAL MODEL --------
model.save("mask_detector_final.h5")

# -------- PLOT TRAINING --------
plt.style.use("ggplot")
plt.figure()
N = len(H.history["loss"])
plt.plot(range(0, N), H.history["loss"], label="train_loss")
plt.plot(range(0, N), H.history["val_loss"], label="val_loss")
plt.plot(range(0, N), H.history["accuracy"], label="train_acc")
plt.plot(range(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.savefig("training_plot.png")

print("✅ Training complete. Model saved as mask_detector_final.h5")
