# Project 1: Image Classification System

## โจทย์โปรเจกต์

สร้างระบบจำแนกภาพแบบครบวงจร ตั้งแต่การเตรียมข้อมูล, สร้างโมเดล, จนถึง Deploy เป็น Web Application ที่ใช้งานได้จริง

## วัตถุประสงค์

1. ฝึกสร้าง CNN Model จากศูนย์
2. ใช้ Transfer Learning เพื่อเพิ่มประสิทธิภาพ
3. Deploy เป็น Web App ด้วย Streamlit
4. สร้างระบบที่สามารถนำไปต่อยอดได้จริง

## Dataset

เลือก 1 จาก:

### Option 1: CIFAR-10 (แนะนำสำหรับเริ่มต้น)
- 60,000 ภาพสี 32x32
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Built-in ใน Keras

### Option 2: Cats vs Dogs
- 25,000 ภาพแมวและสุนัข
- Binary Classification
- Download: [Kaggle](https://www.kaggle.com/c/dogs-vs-cats)

### Option 3: Custom Dataset (ท้าทาย)
- เลือกหัวข้อที่สนใจ เช่น:
  - ผลไม้ไทย 10 ชนิด
  - ป้ายจราจร
  - อาหารไทย
  - ดอกไม้
- รวบรวมภาพเอง 500-1000 ภาพต่อคลาส

## Requirements

```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn streamlit pillow
```

## ขั้นตอนการทำโปรเจกต์

### Phase 1: Data Preparation (ใช้เวลา 1-2 ชั่วโมง)

1. **Load และ Explore Data**
   - โหลด dataset
   - ดูตัวอย่างภาพ
   - ตรวจสอบ distribution ของคลาส

2. **Preprocessing**
   - Normalize ข้อมูล (0-255 → 0-1)
   - Resize ภาพให้เป็นขนาดเดียวกัน
   - แบ่ง Train/Validation/Test (70/15/15)

3. **Data Augmentation**
   ```python
   from tensorflow.keras.preprocessing.image import ImageDataGenerator

   train_datagen = ImageDataGenerator(
       rescale=1./255,
       rotation_range=20,
       width_shift_range=0.2,
       height_shift_range=0.2,
       horizontal_flip=True,
       zoom_range=0.2,
       shear_range=0.2,
       fill_mode='nearest'
   )
   ```

### Phase 2: Baseline Model (ใช้เวลา 1-2 ชั่วโมง)

1. **สร้าง Simple CNN**
   ```python
   model = Sequential([
       Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
       MaxPooling2D(2,2),
       Conv2D(64, (3,3), activation='relu'),
       MaxPooling2D(2,2),
       Conv2D(128, (3,3), activation='relu'),
       MaxPooling2D(2,2),
       Flatten(),
       Dense(128, activation='relu'),
       Dropout(0.5),
       Dense(num_classes, activation='softmax')
   ])
   ```

2. **Train และ Evaluate**
   - Train ประมาณ 20-30 epochs
   - Plot training curves
   - วัด accuracy, precision, recall

3. **เป้าหมาย Baseline:**
   - CIFAR-10: 60-70% accuracy
   - Cats vs Dogs: 70-80% accuracy

### Phase 3: Transfer Learning (ใช้เวลา 2-3 ชั่วโมง)

1. **เลือก Pre-trained Model**
   - MobileNetV2 (เบา, เร็ว)
   - VGG16 (standard)
   - ResNet50 (แม่น)

2. **Feature Extraction**
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
       Dense(256, activation='relu'),
       Dropout(0.5),
       Dense(num_classes, activation='softmax')
   ])
   ```

3. **Fine-tuning**
   ```python
   # Unfreeze top layers
   base_model.trainable = True
   for layer in base_model.layers[:-20]:
       layer.trainable = False

   # Compile with lower learning rate
   model.compile(
       optimizer=Adam(lr=1e-5),
       loss='categorical_crossentropy',
       metrics=['accuracy']
   )
   ```

4. **เป้าหมาย Transfer Learning:**
   - CIFAR-10: 85-90% accuracy
   - Cats vs Dogs: 95%+ accuracy

### Phase 4: Model Optimization (ใช้เวลา 1-2 ชั่วโมง)

1. **Hyperparameter Tuning**
   - Learning rate
   - Batch size
   - Dropout rate
   - Number of layers

2. **Regularization**
   - L2 regularization
   - Batch Normalization
   - Early Stopping

3. **Save Best Model**
   ```python
   callbacks = [
       ModelCheckpoint('best_model.h5', save_best_only=True),
       EarlyStopping(patience=10, restore_best_weights=True),
       ReduceLROnPlateau(factor=0.5, patience=5)
   ]
   ```

### Phase 5: Deployment (ใช้เวลา 2-3 ชั่วโมง)

1. **สร้าง Streamlit App**

```python
# app.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page config
st.set_page_config(
    page_title="Image Classifier",
    page_icon="🖼️",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('best_model.h5')

model = load_model()

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Title
st.title('🖼️ Image Classification App')
st.write('Upload an image to classify it!')

# Sidebar
with st.sidebar:
    st.header('About')
    st.write('This app classifies images into 10 categories.')
    st.write('Built with TensorFlow and Streamlit')

# Upload
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # Display image
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    image_resized = image.resize((32, 32))
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, 0)

    # Predict
    with st.spinner('Classifying...'):
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

    with col2:
        st.subheader('Results')
        st.success(f'**Prediction:** {class_names[predicted_class]}')
        st.info(f'**Confidence:** {confidence*100:.2f}%')

        # All probabilities
        st.subheader('All Probabilities')
        prob_df = pd.DataFrame({
            'Class': class_names,
            'Probability': predictions[0]
        }).sort_values('Probability', ascending=False)

        st.bar_chart(prob_df.set_index('Class'))

# Run: streamlit run app.py
```

2. **Create Dockerfile (Optional)**

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

## โครงสร้างโปรเจกต์

```
Project_1_Image_Classification/
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_transfer_learning.ipynb
│   └── 04_evaluation.ipynb
├── models/
│   ├── baseline_model.h5
│   ├── best_model.h5
│   └── model_quantized.tflite
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── app.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Evaluation Metrics

โมเดลควรผ่านเกณฑ์:

1. **Accuracy:**
   - Baseline: > 60%
   - Transfer Learning: > 85%

2. **Confusion Matrix:**
   - ไม่มีคลาสที่ทำนายผิดมากเกินไป

3. **Per-class Metrics:**
   - Precision และ Recall ทุกคลาส > 70%

4. **Inference Time:**
   - < 1 วินาทีต่อภาพ

## ส่วนขยาย (เพิ่มความท้าทาย)

1. **Gradio Interface**
   - ทำ UI ด้วย Gradio แทน Streamlit

2. **REST API**
   - Deploy ด้วย FastAPI
   - รองรับ batch prediction

3. **Real-time Webcam**
   - Classification จาก webcam แบบ real-time

4. **Model Ensemble**
   - รวมหลายโมเดลเพื่อเพิ่ม accuracy

5. **Explainability**
   - ใช้ Grad-CAM แสดงว่าโมเดลดูส่วนไหนของภาพ

## ตัวอย่าง Output

```
Model: MobileNetV2 + Custom Classifier
Test Accuracy: 89.3%
Test Loss: 0.312

Per-class Accuracy:
  airplane: 91%
  automobile: 94%
  bird: 85%
  cat: 82%
  deer: 87%
  dog: 84%
  frog: 93%
  horse: 90%
  ship: 92%
  truck: 95%

Inference time: 0.23s per image
Model size: 12.5 MB
```

## การส่งงาน

1. **Code:** GitHub repository
2. **Model:** Google Drive link
3. **Demo:** Streamlit app URL หรือ video demo
4. **Report:** สรุปผลการทดลอง, metrics, lessons learned

## Tips

- เริ่มจาก baseline ง่ายๆ ก่อน
- ใช้ small dataset ทดสอบก่อน train full dataset
- Monitor overfitting ด้วย validation set
- บันทึกทุก experiment
- Comment code ให้ชัดเจน

## Resources

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Streamlit Documentation](https://docs.streamlit.io/)

Good luck! 🚀
