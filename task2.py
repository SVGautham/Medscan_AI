



import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# Path to the dataset
dataset_path = r"C:/Users/svgau/Downloads/eye_disease/dataset"  # Raw string for Windows paths

# Parameters
img_height, img_width = 224, 224  # Input size for ResNet50
batch_size = 32
epochs = 20  # Increased epochs for better fine-tuning

from tensorflow.keras.applications.resnet50 import preprocess_input

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # ResNet50-specific preprocessing
    validation_split=0.2,  # 20% for validation
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)

# Load training data
train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
)

# Load validation data
val_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
)

# Feature Extraction: Load ResNet50 without the top layer (include_top=False)
base_model = ResNet50(input_shape=(img_height, img_width, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # Keep the base model frozen for feature extraction

# Extract features from train and validation data
def extract_features(data_generator, model, batch_size):
    """Extract features from the data generator using a pre-trained model."""
    features = []
    labels = []
    for i in range(len(data_generator)):
        batch_imgs, batch_labels = data_generator[i]
        batch_features = model.predict(batch_imgs)
        features.append(batch_features)
        labels.append(batch_labels)
        if i >= len(data_generator) - 1:  # Stop after one epoch
            break
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    return features, labels

# Extract features
print("Extracting features...")
train_features, train_labels = extract_features(train_data, base_model, batch_size)
val_features, val_labels = extract_features(val_data, base_model, batch_size)

# Reshape features for classification layers
train_features = train_features.reshape(train_features.shape[0], -1)
val_features = val_features.reshape(val_features.shape[0], -1)

# Build custom classifier on extracted features
model = Sequential([
    Dense(128, activation="relu", input_shape=(train_features.shape[1],)),  # Input shape matches feature size
    Dense(len(train_data.class_indices), activation="softmax"),  # Output layer matches class count
])

# Compile the classifier (before fine-tuning)
model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

# Train the classifier on extracted features
history = model.fit(
    train_features,
    train_labels,
    validation_data=(val_features, val_labels),
    epochs=epochs,
    batch_size=batch_size,
)

# Fine-Tuning: Unfreeze some layers of the base model
base_model.trainable = True  # Unfreeze all layers

# Choose which layers to fine-tune (you can unfreeze all or just the last few)
fine_tune_at = 100  # Fine-tune layers after the 100th layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False  # Freeze layers before the 100th layer

# Recompile the model with a smaller learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# Continue training with fine-tuning
history_fine_tune = model.fit(
    train_features,
    train_labels,
    validation_data=(val_features, val_labels),
    epochs=epochs,  # Additional epochs for fine-tuning
    batch_size=batch_size,
)

# Plot training and validation accuracy
plt.plot(history_fine_tune.history['accuracy'], label='Training Accuracy')
plt.plot(history_fine_tune.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy (Fine-Tuning)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history_fine_tune.history['loss'], label='Training Loss')
plt.plot(history_fine_tune.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss (Fine-Tuning)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the trained model
model.save(r'C:/Users/svgau/Downloads/eye_disease_classifier_finetuned.h5')
print("Fine-tuned classifier model saved successfully!")

# Test the model with a random image (as before)
def predict_random_image_with_extraction(dataset_folder, base_model, classifier_model, img_height, img_width):
    # Randomly choose a folder
    category = random.choice(os.listdir(dataset_folder))
    category_path = os.path.join(dataset_folder, category)
    
    # Randomly choose an image from the chosen folder
    image_file = random.choice(os.listdir(category_path))
    image_path = os.path.join(category_path, image_file)
    
    # Load the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Extract features using the base model
    features = base_model.predict(img_array)
    features = features.reshape(1, -1)  # Flatten features
    
    # Predict using the classifier
    predictions = classifier_model.predict(features)
    predicted_class = list(train_data.class_indices.keys())[np.argmax(predictions)]
    
    # Display the image and prediction
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class} (Actual: {category})")
    plt.axis("off")
    plt.show()

# Predict a random image using feature extraction
predict_random_image_with_extraction(dataset_path, base_model, model, img_height, img_width)
