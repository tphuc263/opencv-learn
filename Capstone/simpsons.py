import os
import numpy as np
import cv2 as cv # Still useful for image display if needed
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import LearningRateScheduler # Kept for potential future use

# Ensure GPU is available if you have one
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

IMG_SIZE = (80, 80)
channels = 1 # 1 for grayscale
# CẬP NHẬT ĐƯỜNG DẪN DỮ LIỆU CỦA BẠN TẠI ĐÂY
# Ví dụ: nếu dataset của bạn ở cùng thư mục với script:
# char_path = r'./simpsons_dataset'
# test_image_path = r'./kaggle_simpson_testset/kaggle_simpson_testset/charles_montgomery_burns_0.jpg'
char_path = r'../Resources/the-simpsons-characters-dataset/simpsons_dataset'
test_image_path = r'../Resources/the-simpsons-characters-dataset/kaggle_simpson_testset/kaggle_simpson_testset/charles_montgomery_burns_0.jpg'

# Creating a character dictionary, sorting it in descending order
# This part remains mostly the same as it's file system interaction
char_dict = {}
try:
    for char in os.listdir(char_path):
        char_dict[char] = len(os.listdir(os.path.join(char_path, char)))
except FileNotFoundError:
    print(f"Error: Dataset path not found. Please check 'char_path': {char_path}")
    print("Make sure the dataset 'simpsons_dataset' is correctly placed or update the 'char_path' variable.")
    exit() # Exit if dataset not found

# Sort in descending order (standard Python sorting for dictionary items)
char_dict = dict(sorted(char_dict.items(), key=lambda item: item[1], reverse=True))

# Getting the first 10 categories with the most number of images
characters = []
count = 0
for i in char_dict:
    characters.append(i)
    count += 1
    if count >= 10:
        break

# The `image_dataset_from_directory` sorts classes alphabetically.
# To ensure consistent mapping, we sort our `characters` list too.
class_names = sorted(characters)
num_classes = len(class_names)
print(f"Classes for training (sorted alphabetically by directory name): {class_names}")

# Map integer labels back to character names for prediction
index_to_label = {i: name for i, name in enumerate(class_names)}

# --- Data Loading and Preprocessing using tf.data.Dataset ---
BATCH_SIZE = 32
SEED = 42 # For reproducibility

# Function to preprocess images: convert to grayscale, resize, normalize, and one-hot encode labels
def preprocess_image(image, label):
    # image_dataset_from_directory loads images as RGB (3 channels).
    # Convert to grayscale if channels=1 is desired.
    if channels == 1:
        image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0 # Normalize pixel values to [0, 1]
    label = tf.one_hot(label, depth=num_classes) # One-hot encode label
    return image, label

print("\n--- Loading and Preprocessing Data ---")

# Load training data
train_ds = tf.keras.utils.image_dataset_from_directory(
    char_path,
    labels='inferred', # Labels are inferred from directory names
    label_mode='int', # Labels as integers (will be one-hot encoded in preprocess_image)
    class_names=class_names, # Use our sorted class names to ensure consistency
    image_size=IMG_SIZE,
    interpolation='bilinear',
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=0.2, # Allocate 20% for validation
    subset='training'
)

# Load validation data
val_ds = tf.keras.utils.image_dataset_from_directory(
    char_path,
    labels='inferred',
    label_mode='int',
    class_names=class_names,
    image_size=IMG_SIZE,
    interpolation='bilinear',
    batch_size=BATCH_SIZE,
    shuffle=False, # No need to shuffle validation data
    seed=SEED,
    validation_split=0.2,
    subset='validation'
)

# Apply preprocessing to both datasets and optimize for performance
train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Verify a batch shape
for image_batch, label_batch in train_ds.take(1):
    print(f"Image batch shape: {image_batch.shape}")
    print(f"Label batch shape: {label_batch.shape}")
    print(f"Number of training batches: {len(train_ds)}")
    print(f"Number of validation batches: {len(val_ds)}")
    break

# Visualizing the data (using matplotlib)
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1): # Take one batch
    for i in range(min(9, len(images))): # Display up to 9 images
        ax = plt.subplot(3, 3, i + 1)
        # Squeeze the channel dimension if it's 1 for proper display
        plt.imshow(images[i].numpy().squeeze(), cmap='gray' if channels == 1 else None)
        # Use index_to_label to get the actual character name
        predicted_label_index = np.argmax(labels[i].numpy())
        plt.title(index_to_label[predicted_label_index])
        plt.axis("off")
    plt.show()

# --- Model Definition ---
print("\n--- Building Model ---")

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE[0], IMG_SIZE[1], channels)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1024, activation='relu'))

# Output Layer - ensure output_dim matches num_classes
model.add(layers.Dense(num_classes, activation='softmax'))

model.summary()

# --- Training the model ---
print("\n--- Training Model ---")

EPOCHS = 10 # Using your defined EPOCHS
optimizer = optimizers.SGD(learning_rate=0.001, decay=1e-7, momentum=0.9, nesterov=True)

# Using 'categorical_crossentropy' since labels are one-hot encoded
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

callbacks_list = [] # Removed canaro.lr_schedule

training = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks_list
)

print("\nTraining Complete.")
print(f"The 10 characters identified for classification are: {class_names}")

# --- Testing ---
print("\n--- Testing Model ---")

# Function to prepare a single image for prediction
def prepare_for_prediction(image_path, img_size=(80,80), channels=1):
    # Load image using Keras utility, it handles resizing and color mode
    if channels == 1:
        img = tf.keras.utils.load_img(image_path, target_size=img_size, color_mode='grayscale')
    else:
        img = tf.keras.utils.load_img(image_path, target_size=img_size, color_mode='rgb')

    img_array = tf.keras.utils.img_to_array(img) # Convert to numpy array (H, W, C)
    img_array = img_array / 255.0 # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension (1, H, W, C)
    return img_array

# Display the test image using matplotlib
try:
    img_bgr = cv.imread(test_image_path) # Read with OpenCV for display purposes (BGR format)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found at {test_image_path}. Check the path.")
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB) # Convert to RGB for matplotlib
    plt.imshow(img_rgb)
    plt.title("Test Image")
    plt.axis('off')
    plt.show()

    # Make prediction
    prepared_img = prepare_for_prediction(test_image_path, IMG_SIZE, channels)
    predictions = model.predict(prepared_img)

    # Getting class with the highest probability
    predicted_index = np.argmax(predictions[0])
    predicted_character = index_to_label[predicted_index]

    print(f"Predicted character: {predicted_character}")

except FileNotFoundError as e:
    print(f"Error during testing: {e}")
    print("Please ensure 'test_image_path' points to a valid image file.")
except Exception as e:
    print(f"An unexpected error occurred during testing: {e}")