# Quick Reference Guide - คู่มืออ้างอิงด่วน

## 🎯 คำสั่งที่ใช้บ่อย

### การสร้างโมเดล

#### 1. Simple Neural Network
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

#### 2. CNN for Images
```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

#### 3. LSTM for Time Series
```python
from tensorflow.keras.layers import LSTM

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
```

#### 4. Transfer Learning
```python
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```

### การเทรนโมเดล

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks
)
```

### Data Preprocessing

#### Normalization
```python
# Min-Max Scaling (0-1)
X_normalized = (X - X.min()) / (X.max() - X.min())

# Standardization (mean=0, std=1)
X_standardized = (X - X.mean()) / X.std()

# Using Scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

#### Image Augmentation
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)
```

#### Time Series Sequences
```python
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)
```

## 📊 Evaluation & Visualization

### Metrics
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Classification
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
```

### Confusion Matrix
```python
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
```

### Training History
```python
# Plot Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot Accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

## 🔧 Common Functions

### Load Data
```python
# MNIST
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# CIFAR-10
from tensorflow.keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# CSV
import pandas as pd
df = pd.read_csv('data.csv')

# Images from Directory
from tensorflow.keras.preprocessing.image import load_img, img_to_array
img = load_img('image.jpg', target_size=(224, 224))
img_array = img_to_array(img) / 255.0
```

### Save/Load Model
```python
# Save
model.save('model.h5')
model.save('model_folder/')  # SavedModel format

# Load
from tensorflow.keras.models import load_model
model = load_model('model.h5')

# Save weights only
model.save_weights('weights.h5')
model.load_weights('weights.h5')
```

### Predict
```python
# Single prediction
prediction = model.predict(np.expand_dims(X_test[0], 0))
class_id = np.argmax(prediction)

# Batch prediction
predictions = model.predict(X_test)
class_ids = np.argmax(predictions, axis=1)
```

## 🎨 Activation Functions

```python
# Common Activations
'relu'          # ReLU: max(0, x)
'sigmoid'       # Sigmoid: 1 / (1 + e^-x)
'tanh'          # Tanh: (e^x - e^-x) / (e^x + e^-x)
'softmax'       # Softmax: e^xi / Σe^xj
'linear'        # Linear: x (no activation)
'elu'           # ELU
'leaky_relu'    # Leaky ReLU
```

**เลือกใช้:**
- Hidden layers: `relu`
- Binary classification output: `sigmoid`
- Multi-class classification output: `softmax`
- Regression output: `linear`

## ⚙️ Optimizers

```python
# Common Optimizers
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

# Adam (แนะนำ)
optimizer = Adam(learning_rate=0.001)

# SGD
optimizer = SGD(learning_rate=0.01, momentum=0.9)

# RMSprop
optimizer = RMSprop(learning_rate=0.001)
```

## 📉 Loss Functions

```python
# Classification
'binary_crossentropy'              # Binary (2 classes)
'categorical_crossentropy'         # Multi-class (one-hot)
'sparse_categorical_crossentropy'  # Multi-class (integer labels)

# Regression
'mean_squared_error'               # MSE
'mean_absolute_error'              # MAE
'huber'                            # Huber loss
```

## 🛡️ Regularization

### Dropout
```python
from tensorflow.keras.layers import Dropout
Dropout(0.5)  # Drop 50% of neurons
```

### L1/L2 Regularization
```python
from tensorflow.keras.regularizers import l1, l2, l1_l2

Dense(64, activation='relu', kernel_regularizer=l2(0.01))
```

### Batch Normalization
```python
from tensorflow.keras.layers import BatchNormalization
BatchNormalization()
```

### Early Stopping
```python
from tensorflow.keras.callbacks import EarlyStopping
EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
```

## 🔍 Debugging Tips

### Check Shapes
```python
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")
```

### View Model Summary
```python
model.summary()
```

### Monitor Training
```python
# Verbose levels
verbose=0  # Silent
verbose=1  # Progress bar
verbose=2  # One line per epoch
```

### Check GPU
```python
import tensorflow as tf
print("GPU Available:", len(tf.config.list_physical_devices('GPU')) > 0)
```

### Memory Management
```python
# Clear session
from tensorflow.keras import backend as K
K.clear_session()

# Limit GPU memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
    )
```

## 📦 Import Template

```python
# Core
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Utilities
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)
```

## 🎯 Hyperparameter Ranges

```python
# Learning Rate
learning_rate: [1e-4, 1e-3, 1e-2]

# Batch Size
batch_size: [16, 32, 64, 128]

# Epochs
epochs: [20, 50, 100]

# Dropout Rate
dropout: [0.2, 0.3, 0.5]

# Hidden Units
units: [32, 64, 128, 256, 512]

# Number of Layers
layers: [2, 3, 4, 5]
```

## 🚀 Performance Tips

1. **Use GPU** - 10-100x faster
2. **Batch Normalization** - Faster training
3. **Mixed Precision** - Faster on modern GPUs
4. **Data Pipeline** - Use `tf.data`
5. **Model Checkpoint** - Save best model
6. **Early Stopping** - Avoid unnecessary epochs

## 📝 Code Snippets

### Plot Predictions
```python
def plot_predictions(y_true, y_pred, n=10):
    plt.figure(figsize=(15, 5))
    plt.plot(y_true[:n], 'bo-', label='True')
    plt.plot(y_pred[:n], 'r^-', label='Predicted')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
```

### Learning Rate Scheduler
```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)
```

### Custom Callback
```python
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs['val_accuracy'] > 0.95:
            print(f"\nReached 95% accuracy at epoch {epoch+1}!")
            self.model.stop_training = True
```

---

**จัดทำโดย:** Deep Learning Course
**อัปเดตล่าสุด:** 2025-01-05
