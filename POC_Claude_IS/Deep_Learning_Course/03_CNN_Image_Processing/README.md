# บทที่ 3: CNN และการประมวลผลภาพ

## เป้าหมายการเรียนรู้

- เข้าใจหลักการทำงานของ Convolutional Neural Networks (CNN)
- รู้จักส่วนประกอบของ CNN (Conv, Pooling, etc.)
- เข้าใจ Transfer Learning
- สามารถสร้างโมเดล Image Classification และ Object Detection ได้

## เนื้อหา

### 3.1 Introduction to CNN

**ทำไมต้องใช้ CNN กับภาพ?**

- Neural Network ธรรมดา: ใช้ parameters เยอะเกิน
  - ภาพ 224x224x3 = 150,528 inputs
  - Connected กับ 1000 neurons = 150M parameters!

- CNN: ใช้ Convolution แทน Fully Connected
  - Parameters น้อยกว่ามาก
  - เรียนรู้ Local Patterns ได้ดี
  - Translation Invariant (เห็นวัตถุที่ตำแหน่งไหนก็ได้)

**การใช้งาน:**
- Image Classification (จำแนกภาพ)
- Object Detection (หาวัตถุในภาพ)
- Image Segmentation (แบ่งส่วนภาพ)
- Face Recognition (จำใบหน้า)

### 3.2 CNN Architecture Components

#### 3.2.1 Convolutional Layer

**หลักการ:**
- ใช้ Filter (Kernel) เลื่อนสแกนภาพ
- แต่ละ Filter เรียนรู้ Feature ต่างกัน
- ชั้นแรก: ขอบ, เส้น
- ชั้นลึก: รูปทรง, วัตถุ

**Parameters:**
```python
Conv2D(
    filters=32,        # จำนวน filters
    kernel_size=(3,3), # ขนาด filter 3x3
    strides=(1,1),     # ขั้นการเลื่อน
    padding='same',    # same = ขนาดเดิม, valid = ลดลง
    activation='relu'
)
```

**คำนวณ Output Size:**
```
output_size = (input_size - kernel_size + 2*padding) / stride + 1
```

#### 3.2.2 Pooling Layer

**หน้าที่:**
- ลดขนาดข้อมูล
- ลด parameters
- ทำให้ robust ต่อการเคลื่อนที่

**ประเภท:**

1. **Max Pooling** (นิยมที่สุด)
   ```python
   MaxPooling2D(pool_size=(2,2))
   ```
   - เลือกค่าสูงสุดใน window

2. **Average Pooling**
   ```python
   AveragePooling2D(pool_size=(2,2))
   ```
   - หาค่าเฉลี่ยใน window

3. **Global Average Pooling**
   ```python
   GlobalAveragePooling2D()
   ```
   - เฉลี่ยทั้ง feature map → 1 ค่า

#### 3.2.3 Batch Normalization

```python
BatchNormalization()
```

**ประโยชน์:**
- Normalize ระหว่าง layers
- เทรนเร็วขึ้น
- ลด Internal Covariate Shift
- ช่วย regularization

#### 3.2.4 Dropout

```python
Dropout(0.5)
```

**ประโยชน์:**
- ป้องกัน Overfitting
- ปิด neurons แบบสุ่มตอน training
- ทำให้โมเดล robust

### 3.3 CNN Architectures

#### Classic Architectures:

1. **LeNet-5 (1998)**
   - CNN แรกสุด
   - ใช้กับตัวเลขเขียนมือ

2. **AlexNet (2012)**
   - ชนะ ImageNet
   - ทำให้ Deep Learning ฮิต

3. **VGG-16/19 (2014)**
   - เน้น depth (16-19 layers)
   - ใช้ filter 3x3 ตลอด

4. **ResNet (2015)**
   - Skip Connections
   - เทรนโมเดลลึก 100+ layers ได้

5. **MobileNet (2017)**
   - เบาสำหรับ mobile
   - Depthwise Separable Convolutions

### 3.4 Transfer Learning

**แนวคิด:**
- ใช้โมเดลที่เทรนบน ImageNet (1.4M ภาพ)
- Fine-tune สำหรับงานของเราเอง
- ประหยัดเวลาและข้อมูล

**วิธีทำ:**

```python
# 1. โหลด Pre-trained Model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,  # ไม่เอา classifier เดิม
    weights='imagenet'
)

# 2. Freeze layers (ไม่ให้เทรน)
base_model.trainable = False

# 3. เพิ่ม Custom Classifier
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```

**Strategies:**

1. **Feature Extraction**
   - Freeze ทุก layer ของ base model
   - เทรนแค่ classifier

2. **Fine-tuning**
   - Freeze บางส่วน
   - เทรนต่อกับ learning rate เล็ก

### 3.5 Image Classification Workflow

```python
# 1. Data Preparation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# 2. Build Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
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

# 3. Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Train
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30
)
```

### 3.6 Object Detection (Introduction)

**ความแตกต่างจาก Classification:**
- Classification: ภาพนี้คืออะไร?
- Detection: อะไรอยู่ที่ไหนบ้าง?

**Popular Architectures:**
- YOLO (You Only Look Once) - เร็วมาก
- SSD (Single Shot Detector) - เร็ว
- Faster R-CNN - แม่นยำ

## ไฟล์ในบทนี้

- `01_cnn_basics.ipynb` - ทำความเข้าใจ CNN
- `02_build_cnn.ipynb` - สร้าง CNN จากศูนย์
- `03_transfer_learning.ipynb` - ใช้ Pre-trained Models
- `04_data_augmentation.ipynb` - เทคนิค Augmentation
- `05_object_detection_intro.ipynb` - เริ่มต้น Object Detection
- `exercises.ipynb` - แบบฝึกหัด

## แบบฝึกหัด

### Exercise 1: Custom CNN for CIFAR-10
- สร้าง CNN เพื่อจำแนกภาพ CIFAR-10 (10 classes)
- ทดลองใช้ Data Augmentation
- บรรลุ accuracy > 75%

### Exercise 2: Transfer Learning Comparison
- เปรียบเทียบ MobileNetV2, VGG16, ResNet50
- Fine-tune สำหรับ dataset ของคุณ
- วิเคราะห์ speed vs accuracy

## Resources

- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
- [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/)

## Next Step

ไปต่อที่ [บทที่ 4: RNN/LSTM และการประมวลผล Sequential Data](../04_RNN_Sequential_Data/README.md)
