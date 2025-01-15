import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization


# Load data
gestures = [
    "A", "A2", "A3", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "S2", "T", "T2", "U", "V", "W", "X", "Y", "Z"
]

data = []
for gesture in gestures:
    data.extend(np.load(f"RSLOutput/{gesture}_data.npy", allow_pickle=True))

# Separate features and labels
X = np.array([item[0] for item in data])  # Landmarks
y = np.array([item[1] for item in data])  # Labels

# Ensure consistent sequence lengths
sequence_length = 15
X_padded = pad_sequences(X, maxlen=sequence_length, dtype='float32', padding='post', truncating='post')# Ensure landmarks are reshaped to (sequence_length, features_per_frame)
sequence_length = 15  # Number of frames in a sequence
X_fixed = [np.array(seq).reshape(-1, 63) for seq in X]  # Reshape each sequence

# Pad sequences to ensure consistent length
X_padded = pad_sequences(X_fixed, maxlen=sequence_length, dtype='float32', padding='post', truncating='post')


# One-hot encode labels
classes = sorted(set(y))  # Dynamically detect unique classes
y_encoded = to_categorical([classes.index(label) for label in y], num_classes=len(classes))

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(X_padded, y_encoded, test_size=0.2, random_state=42)

# Define model
model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(sequence_length, 63)),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(f"X_padded shape: {X_padded.shape}")
print(f"y_encoded shape: {y_encoded.shape}")
print(f"Classes: {classes}")
print("Sample input shape:", X_padded[0].shape)
print("Sample label:", y_encoded[0])

# Train model
y_train_indices = np.argmax(y_train, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_indices), y=y_train_indices)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print("Class Weights:", class_weights_dict)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=150, batch_size=32, class_weight = class_weights_dict, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Save model
model.save("gesture_recognition_modelv4.h5")
