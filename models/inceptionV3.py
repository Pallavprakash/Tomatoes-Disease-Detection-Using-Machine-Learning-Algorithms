"""
inceptionv3_model.py

Tomato Disease Detection using Transfer Learning with InceptionV3.
Trains and fine-tunes a deep learning model to classify tomato leaf diseases.
"""

import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Configuration Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5
INITIAL_EPOCHS = 50
FINE_TUNE_EPOCHS = 30
UNFREEZE_LAYERS = 100

TRAIN_DIR = 'data/sample_training_images'
VAL_DIR = 'data/sample_validation_images'
MODEL_SAVE_PATH = 'outputs/inception_v3_tomato_disease.h5'


def build_model():
    """Builds and compiles the InceptionV3 model with custom top layers."""
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze base model
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model, base_model


def prepare_data():
    """Prepares data generators for training and validation datasets."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    return train_generator, val_generator


def plot_training(history, title='Training History'):
    """Plots training and validation accuracy and loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    plt.suptitle(title)

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')

    plt.tight_layout()
    plt.show()


def train_model(model, base_model, train_gen, val_gen):
    """Trains the model in two phases: initial training and fine-tuning."""
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)

    # Phase 1: Initial training
    history = model.fit(
        train_gen,
        epochs=INITIAL_EPOCHS,
        validation_data=val_gen,
        callbacks=[early_stop, reduce_lr]
    )

    # Fine-tuning
    for layer in base_model.layers[-UNFREEZE_LAYERS:]:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history_fine = model.fit(
        train_gen,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=val_gen,
        callbacks=[early_stop, reduce_lr]
    )

    return model, history, history_fine


def plot_combined(history1, history2):
    """Plots combined training and fine-tuning history."""
    total_acc = history1.history['accuracy'] + history2.history['accuracy']
    total_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    total_loss = history1.history['loss'] + history2.history['loss']
    total_val_loss = history1.history['val_loss'] + history2.history['val_loss']

    epochs_range = range(len(total_acc))

    plt.figure(figsize=(12, 4))
    plt.suptitle("Combined Training & Fine-tuning")

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, total_acc, label='Train Accuracy')
    plt.plot(epochs_range, total_val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, total_loss, label='Train Loss')
    plt.plot(epochs_range, total_val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')

    plt.tight_layout()
    plt.show()


def main():
    print(f"TensorFlow version: {tf.__version__}")
    model, base_model = build_model()
    train_gen, val_gen = prepare_data()
    model, history, history_fine = train_model(model, base_model, train_gen, val_gen)

    model.save(MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")

    # Plot training history
    plot_combined(history, history_fine)


if __name__ == '__main__':
    main()
