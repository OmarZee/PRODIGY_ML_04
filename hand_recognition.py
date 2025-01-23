import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

# Define paths and classes
data_dir = 'C:/Users/omars/OneDrive/Desktop/Work/Prodigy InfoTech/PRODIGY_ML_04/leapGestRecog'
subjects = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09']
classes = ['01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb', '06_index', '07_ok', '08_palm_moved', '09_c', '10_down']

class_meanings = {
    0: "Palm",
    1: "L Sign",
    2: "Fist",
    3: "Fist Moved",
    4: "Thumb Up",
    5: "Index Pointing",
    6: "OK Sign",
    7: "Palm Moved",
    8: "C Shape",
    9: "Downward Gesture"
}

# Initialize data and labels
# data = []
# labels = []

# # Load and preprocess images
# for subject in subjects:
#     subject_dir = os.path.join(data_dir, subject)
#     for class_id, class_ in enumerate(classes):  # Use enumerate for efficiency
#         class_dir = os.path.join(subject_dir, class_)
#         if os.path.exists(class_dir):  # Ensure directory exists
#             for file in os.listdir(class_dir):
#                 img_path = os.path.join(class_dir, file)
#                 img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#                 if img is not None:
#                     img = cv2.resize(img, (224, 224))  # Resize image
#                     img = img / 255.0  # Normalize pixel values
#                     data.append(img)
#                     labels.append(class_id)

# Convert to NumPy arrays
# data = np.array(data, dtype=np.float32)
# labels = np.array(labels, dtype=np.int32)

# # Save data and labels using pickle
data_file = "data.pkl"
labels_file = "labels.pkl"

# with open(data_file, 'wb') as file:
#     pickle.dump(data, file)  # Corrected variable name
# with open(labels_file, 'wb') as file:
#     pickle.dump(labels, file)  # Corrected variable name

with open(data_file, 'rb') as file:
    data = pickle.load(file)
with open(labels_file, 'rb') as file:
    labels = pickle.load(file)

X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, random_state=42)   
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_meanings), activation='softmax')  # Output layer
])

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.summary()

# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     epochs=20,
#     batch_size=32,
#     callbacks=[
#         tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
#         tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
#     ]
# )

# model.save("hand_gesture_model.h5")

# test_loss, test_acc = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {test_acc}")
# predictions = np.argmax(model.predict(X_test), axis=1)

# print(classification_report(y_test, predictions, target_names=class_meanings.values()))

# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()


# Print shapes to confirm
# print(f"Data shape: {data.shape}")
# print(f"Labels shape: {labels.shape}")

# # Visualize some images
# plt.figure(figsize=(10, 10))
# for i in range(9):
#     plt.subplot(3, 3, i + 1)
#     plt.imshow(data[i])
#     plt.title(f"Class: {classes[labels[i]]}")
#     plt.axis('off')
# plt.show()


## Real time testing

# Load the model'
model = load_model("hand_gesture_model.h5") 
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0
    input_data = np.expand_dims(normalized_frame, axis=0)

    # Predict gesture
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)
    gesture_name = class_meanings[predicted_class]

    # Display the gesture on the video feed
    cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Hand Gesture Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()