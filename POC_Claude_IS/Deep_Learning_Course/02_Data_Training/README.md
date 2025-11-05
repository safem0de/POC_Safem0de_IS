# บทที่ 2: การเตรียมข้อมูลและ Training

## เป้าหมายการเรียนรู้

- เข้าใจการเตรียมข้อมูลสำหรับ Deep Learning
- รู้จัก Loss Functions และ Optimizers ต่างๆ
- เข้าใจเทคนิค Regularization
- สามารถ Train และ Evaluate โมเดลได้อย่างมีประสิทธิภาพ

## เนื้อหา

### 2.1 Data Preprocessing

**ขั้นตอนการเตรียมข้อมูล:**

1. **Data Loading**
   - โหลดข้อมูลจากแหล่งต่างๆ (CSV, Images, APIs)
   - ตรวจสอบความสมบูรณ์

2. **Data Cleaning**
   - จัดการ Missing Values
   - ลบ Outliers
   - แก้ไข Inconsistencies

3. **Data Transformation**
   - **Normalization:** ปรับให้อยู่ในช่วง 0-1
     ```python
     x_normalized = (x - x.min()) / (x.max() - x.min())
     ```

   - **Standardization:** ปรับให้มี mean=0, std=1
     ```python
     x_standardized = (x - x.mean()) / x.std()
     ```

4. **Data Augmentation** (สำหรับภาพ)
   - Rotation, Flip, Zoom
   - ช่วยป้องกัน Overfitting

5. **Train/Validation/Test Split**
   - Training: 70-80%
   - Validation: 10-15%
   - Test: 10-15%

### 2.2 Loss Functions

Loss Function วัดความผิดพลาดของโมเดล

**Loss Functions ที่นิยม:**

1. **Mean Squared Error (MSE)** - Regression
   ```
   MSE = (1/n) Σ(y_true - y_pred)²
   ```
   - ใช้กับปัญหาทำนายค่าต่อเนื่อง

2. **Binary Crossentropy** - Binary Classification
   ```
   BCE = -[y*log(p) + (1-y)*log(1-p)]
   ```
   - ใช้กับปัญหา 2 คลาส

3. **Categorical Crossentropy** - Multi-class Classification
   ```
   CCE = -Σ y_true * log(y_pred)
   ```
   - ใช้กับปัญหาหลายคลาส

4. **Sparse Categorical Crossentropy**
   - เหมือน CCE แต่ใช้กับ integer labels

### 2.3 Optimizers

Optimizer ปรับ weights เพื่อลด loss

**Optimizers ที่นิยม:**

1. **Stochastic Gradient Descent (SGD)**
   ```
   w = w - learning_rate * gradient
   ```
   - พื้นฐานที่สุด
   - ช้าแต่เสถียร

2. **Momentum**
   ```
   v = beta * v + gradient
   w = w - learning_rate * v
   ```
   - เร็วกว่า SGD
   - ลด oscillation

3. **Adam (Adaptive Moment Estimation)**
   - นิยมใช้มากที่สุด
   - ปรับ learning rate อัตโนมัติ
   - เหมาะกับปัญหาส่วนใหญ่

4. **RMSprop**
   - ดีกับ RNN
   - Adaptive learning rate

**การเลือก Learning Rate:**
- เล็กเกินไป: เรียนรู้ช้า
- ใหญ่เกินไป: ไม่ converge
- ทดลอง: 0.001, 0.0001

### 2.4 Regularization Techniques

ป้องกัน Overfitting (จำข้อมูล train มากเกินไป)

**เทคนิคที่นิยม:**

1. **L1 Regularization (Lasso)**
   ```
   Loss = Original_Loss + λ * Σ|w|
   ```
   - ทำให้บาง weights เป็น 0
   - Feature selection

2. **L2 Regularization (Ridge)**
   ```
   Loss = Original_Loss + λ * Σw²
   ```
   - ลด weights แต่ไม่เป็น 0
   - นิยมใช้มากกว่า L1

3. **Dropout**
   ```python
   layers.Dropout(0.5)  # ปิด 50% neurons แบบสุ่ม
   ```
   - ป้องกัน overfitting ได้ดี
   - ใช้ตอน training เท่านั้น

4. **Early Stopping**
   - หยุด training เมื่อ validation loss ไม่ดีขึ้น
   - ประหยัดเวลา

5. **Batch Normalization**
   ```python
   layers.BatchNormalization()
   ```
   - Normalize ระหว่าง layers
   - เร่งการเรียนรู้

### 2.5 Training Process

**ขั้นตอน:**

1. **Initialize Model**
   - กำหนดโครงสร้าง
   - Set random weights

2. **Compile Model**
   ```python
   model.compile(
       optimizer='adam',
       loss='categorical_crossentropy',
       metrics=['accuracy']
   )
   ```

3. **Callbacks**
   ```python
   callbacks = [
       EarlyStopping(patience=5),
       ModelCheckpoint('best_model.h5'),
       ReduceLROnPlateau(factor=0.5)
   ]
   ```

4. **Training**
   ```python
   history = model.fit(
       x_train, y_train,
       epochs=50,
       batch_size=32,
       validation_data=(x_val, y_val),
       callbacks=callbacks
   )
   ```

5. **Monitor Training**
   - ดู loss และ accuracy curves
   - ตรวจสอบ overfitting

### 2.6 Model Evaluation

**Metrics:**

1. **Accuracy** - ถูกกี่เปอร์เซ็นต์
2. **Precision** - ทำนายบวกแล้วถูกกี่เปอร์เซ็นต์
3. **Recall** - ของจริงบวกจับได้กี่เปอร์เซ็นต์
4. **F1-Score** - ค่ากลาง Precision และ Recall
5. **Confusion Matrix** - ดูรายละเอียดการทำนาย

**Validation Strategies:**
- Hold-out Validation
- K-Fold Cross Validation
- Stratified K-Fold

## ไฟล์ในบทนี้

- `01_data_preprocessing.ipynb` - การเตรียมข้อมูล
- `02_loss_and_optimizers.ipynb` - ทดลอง Loss Functions และ Optimizers
- `03_regularization_techniques.ipynb` - เทคนิคป้องกัน Overfitting
- `04_training_workflow.ipynb` - ขั้นตอนการ Train ที่สมบูรณ์
- `exercises.ipynb` - แบบฝึกหัด

## แบบฝึกหัด

### Exercise 1: Data Preprocessing Pipeline
- สร้าง pipeline เตรียมข้อมูลครบวงจร
- ใช้ Data Augmentation
- แบ่ง train/val/test อย่างถูกต้อง

### Exercise 2: Hyperparameter Tuning
- ทดลองหา learning rate ที่ดีที่สุด
- เปรียบเทียบ optimizers (SGD, Adam, RMSprop)
- ทดสอบเทคนิค regularization ต่างๆ

## Next Step

ไปต่อที่ [บทที่ 3: CNN และการประมวลผลภาพ](../03_CNN_Image_Processing/README.md)
