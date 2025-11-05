# บทที่ 5: โปรเจกต์จริงและการ Deploy

## เป้าหมายการเรียนรู้

- เข้าใจกระบวนการสร้างโปรเจกต์ Deep Learning แบบครบวงจร
- รู้วิธี Optimize โมเดลให้มีประสิทธิภาพ
- สามารถ Deploy โมเดลไปใช้งานจริงได้
- เรียนรู้ Best Practices

## เนื้อหา

### 5.1 End-to-End Project Workflow

**ขั้นตอนการทำโปรเจกต์:**

```
1. Problem Definition
   └─ เข้าใจปัญหาที่จะแก้
   └─ กำหนด Success Metrics

2. Data Collection & Exploration
   └─ หาข้อมูล
   └─ EDA (Exploratory Data Analysis)

3. Data Preprocessing
   └─ Cleaning
   └─ Augmentation
   └─ Split datasets

4. Model Development
   └─ Baseline model
   └─ Experiments
   └─ Hyperparameter tuning

5. Model Evaluation
   └─ Test on unseen data
   └─ Error analysis

6. Model Optimization
   └─ Quantization
   └─ Pruning
   └─ Compression

7. Deployment
   └─ Choose platform
   └─ Create API
   └─ Monitor performance

8. Maintenance
   └─ Update data
   └─ Retrain model
   └─ Handle edge cases
```

### 5.2 Model Optimization

#### 5.2.1 Quantization

แปลง weights จาก float32 → int8

```python
import tensorflow as tf

# Post-Training Quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# บันทึก
with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)
```

**ประโยชน์:**
- ขนาดเล็กลง 4 เท่า
- เร็วขึ้น 2-3 เท่า
- Accuracy ลดเล็กน้อย (1-2%)

#### 5.2.2 Pruning

ตัด weights ที่ไม่สำคัญออก

```python
import tensorflow_model_optimization as tfmot

# Define pruning schedule
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,  # ตัด 50%
        begin_step=0,
        end_step=1000
    )
}

# Apply pruning
model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
    model, **pruning_params
)

# Compile and train
model_for_pruning.compile(...)
model_for_pruning.fit(...)

# Strip pruning wrappers
final_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
```

#### 5.2.3 Knowledge Distillation

Student model เรียนจาก Teacher model

```python
# Teacher (large model)
teacher = load_large_model()

# Student (small model)
student = create_small_model()

# Distillation loss
def distillation_loss(y_true, y_pred, teacher_pred, temperature=3):
    soft_labels = tf.nn.softmax(teacher_pred / temperature)
    student_soft = tf.nn.softmax(y_pred / temperature)

    distill_loss = tf.keras.losses.categorical_crossentropy(
        soft_labels, student_soft
    )
    student_loss = tf.keras.losses.categorical_crossentropy(
        y_true, y_pred
    )

    return 0.9 * distill_loss + 0.1 * student_loss
```

### 5.3 Deployment Strategies

#### 5.3.1 TensorFlow Serving

Deploy โมเดลเป็น REST API

```bash
# บันทึกโมเดล
model.save('models/my_model/1')

# รัน TensorFlow Serving
docker run -p 8501:8501 \
  --mount type=bind,source=/path/to/models/my_model,target=/models/my_model \
  -e MODEL_NAME=my_model \
  -t tensorflow/serving
```

**Request:**
```python
import requests
import json

data = json.dumps({
    "signature_name": "serving_default",
    "instances": X_test[:5].tolist()
})

response = requests.post(
    'http://localhost:8501/v1/models/my_model:predict',
    data=data
)
predictions = response.json()['predictions']
```

#### 5.3.2 Flask API

สร้าง Web API แบบง่าย

```python
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = np.array(data['input'])

    # Preprocess
    input_data = preprocess(input_data)

    # Predict
    prediction = model.predict(input_data)

    return jsonify({
        'prediction': prediction.tolist(),
        'class': int(np.argmax(prediction))
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

#### 5.3.3 FastAPI (แนะนำ)

เร็วกว่า Flask และมี auto documentation

```python
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import tensorflow as tf

app = FastAPI()
model = tf.keras.models.load_model('model.h5')

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # Read image
    image = Image.open(file.file)

    # Preprocess
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, 0)

    # Predict
    prediction = model.predict(image_array)
    class_id = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return {
        'class': class_id,
        'confidence': confidence,
        'all_probabilities': prediction[0].tolist()
    }

# Run: uvicorn main:app --reload
```

#### 5.3.4 Streamlit App

สร้าง Web UI อย่างง่าย

```python
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title('Image Classification App')

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.h5')

model = load_model()

# Upload
uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'png'])

if uploaded_file:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image')

    # Preprocess
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, 0)

    # Predict
    if st.button('Predict'):
        prediction = model.predict(image_array)
        class_id = np.argmax(prediction)
        confidence = np.max(prediction)

        st.success(f'Prediction: Class {class_id}')
        st.info(f'Confidence: {confidence:.2%}')

        # Chart
        st.bar_chart(prediction[0])
```

#### 5.3.5 Mobile Deployment (TensorFlow Lite)

Deploy บน Android/iOS

```python
# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

**Android:**
```kotlin
// Load model
val interpreter = Interpreter(loadModelFile())

// Prepare input
val inputBuffer = FloatBuffer.allocate(224 * 224 * 3)
// ... fill buffer ...

// Run inference
val outputBuffer = FloatBuffer.allocate(num_classes)
interpreter.run(inputBuffer, outputBuffer)
```

### 5.4 Monitoring and Logging

```python
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    filename='predictions.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

@app.post('/predict')
def predict(data):
    start_time = datetime.now()

    # Predict
    prediction = model.predict(data)

    # Log
    duration = (datetime.now() - start_time).total_seconds()
    logging.info(f'Prediction: {prediction}, Time: {duration}s')

    return {'prediction': prediction}
```

### 5.5 Best Practices

#### 5.5.1 Version Control

```
project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── models/
│   ├── v1/
│   ├── v2/
│   └── experiments/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_final.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

#### 5.5.2 Experiment Tracking

```python
import mlflow

with mlflow.start_run():
    # Log parameters
    mlflow.log_param('learning_rate', 0.001)
    mlflow.log_param('batch_size', 32)

    # Train model
    history = model.fit(...)

    # Log metrics
    mlflow.log_metric('accuracy', history.history['accuracy'][-1])
    mlflow.log_metric('val_accuracy', history.history['val_accuracy'][-1])

    # Log model
    mlflow.keras.log_model(model, 'model')
```

#### 5.5.3 Testing

```python
import pytest
import numpy as np

def test_model_output_shape():
    model = load_model()
    dummy_input = np.random.rand(1, 224, 224, 3)
    output = model.predict(dummy_input)
    assert output.shape == (1, 10)

def test_preprocessing():
    image = load_image('test.jpg')
    processed = preprocess(image)
    assert processed.shape == (224, 224, 3)
    assert processed.max() <= 1.0
    assert processed.min() >= 0.0
```

#### 5.5.4 Documentation

```python
def preprocess_image(image_path: str, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Preprocess image for model input.

    Args:
        image_path (str): Path to image file
        target_size (tuple): Target size for resizing (height, width)

    Returns:
        np.ndarray: Preprocessed image array with shape (height, width, 3)
                   and values in range [0, 1]

    Example:
        >>> img = preprocess_image('cat.jpg', (224, 224))
        >>> img.shape
        (224, 224, 3)
    """
    image = Image.open(image_path)
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    return image_array
```

## ไฟล์ในบทนี้

- `01_project_structure.md` - โครงสร้างโปรเจกต์
- `02_model_optimization.ipynb` - Optimization techniques
- `03_deployment_flask.py` - Flask API example
- `04_deployment_fastapi.py` - FastAPI example
- `05_deployment_streamlit.py` - Streamlit app
- `06_monitoring.py` - Logging and monitoring
- `07_testing.py` - Unit tests

## แบบฝึกหัด

### Exercise 1: Complete ML Pipeline
- สร้าง pipeline ครบวงจร
- Train model → Optimize → Deploy
- เขียน tests และ documentation

### Exercise 2: Model Comparison Dashboard
- สร้าง dashboard เปรียบเทียบโมเดล
- แสดง metrics, training curves
- Track experiments

## Resources

- [MLOps Guide](https://ml-ops.org/)
- [TensorFlow Deployment](https://www.tensorflow.org/tfx/guide)

## Next Step

ทำโปรเจกต์จริง:
- [Project 1: Image Classification](../Projects/Project_1_Image_Classification/README.md)
- [Project 2: Time Series Prediction](../Projects/Project_2_Time_Series_Prediction/README.md)
