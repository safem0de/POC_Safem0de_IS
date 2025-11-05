# บทที่ 1: พื้นฐาน Deep Learning และ Neural Networks

## เป้าหมายการเรียนรู้

- เข้าใจหลักการทำงานของ Neural Networks
- รู้จัก Activation Functions ต่างๆ
- เข้าใจกระบวนการ Forward และ Backward Propagation
- สามารถสร้าง Neural Network แบบง่ายได้

## เนื้อหา

### 1.1 Introduction to Deep Learning

Deep Learning คือสาขาหนึ่งของ Machine Learning ที่ใช้ Neural Networks ที่มีหลายชั้น (layers) เพื่อเรียนรู้รูปแบบที่ซับซ้อนจากข้อมูล

**ความแตกต่างระหว่าง ML และ DL:**
- ML: ต้อง Feature Engineering ด้วยมือ
- DL: เรียนรู้ Features อัตโนมัติจากข้อมูล

**การใช้งาน:**
- Computer Vision (การมองเห็น)
- Natural Language Processing (ภาษา)
- Speech Recognition (เสียง)
- Recommendation Systems

### 1.2 Neural Network Architecture

**โครงสร้างพื้นฐาน:**
```
Input Layer → Hidden Layers → Output Layer
```

**ส่วนประกอบ:**
- **Neurons (Nodes):** หน่วยประมวลผลพื้นฐาน
- **Weights:** น้ำหนักของการเชื่อมต่อ
- **Bias:** ค่าคงที่ปรับแต่งผลลัพธ์
- **Layers:** ชั้นของ neurons

### 1.3 Activation Functions

Activation Functions ช่วยให้โมเดลเรียนรู้รูปแบบที่ซับซ้อน (non-linear)

**ฟังก์ชันที่นิยม:**

1. **Sigmoid:** σ(x) = 1 / (1 + e^(-x))
   - Output: 0 ถึง 1
   - ใช้กับ Binary Classification

2. **ReLU:** f(x) = max(0, x)
   - เร็วและใช้บ่อยที่สุด
   - แก้ปัญหา Vanishing Gradient

3. **Tanh:** tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
   - Output: -1 ถึง 1
   - ศูนย์กลางอยู่ที่ 0

4. **Softmax:** สำหรับ Multi-class Classification
   - Output: Probability Distribution

### 1.4 Forward Propagation

การส่งข้อมูลจาก Input ไปยัง Output

```
1. รับ Input
2. คูณกับ Weights และบวก Bias
3. ผ่าน Activation Function
4. ทำซ้ำทุก Layer จนถึง Output
```

### 1.5 Backward Propagation

การปรับ Weights เพื่อลด Error

```
1. คำนวณ Loss (ความผิดพลาด)
2. หา Gradient ของ Loss relative to weights
3. Update Weights ด้วย Gradient Descent
4. ทำซ้ำจนโมเดลดีขึ้น
```

**Gradient Descent:**
```
weight_new = weight_old - learning_rate * gradient
```

## ไฟล์ในบทนี้

- `01_neural_network_basics.ipynb` - Notebook อธิบายทฤษฎี
- `02_build_simple_nn.ipynb` - สร้าง Neural Network แรก
- `03_activation_functions.ipynb` - ทดลอง Activation Functions
- `exercises.ipynb` - แบบฝึกหัด
- `solutions.ipynb` - เฉลย

## แบบฝึกหัด

### Exercise 1: สร้าง Neural Network สำหรับ XOR Problem
- สร้าง network ที่มี 1 hidden layer
- แก้ปัญหา XOR logic gate
- Visualize decision boundary

### Exercise 2: เปรียบเทียบ Activation Functions
- ทดลองใช้ Sigmoid, ReLU, Tanh
- วัดความเร็วและ accuracy
- Plot และวิเคราะห์ผลลัพธ์

## Resources

- [Neural Networks Visualizer](https://playground.tensorflow.org/)
- [3Blue1Brown - Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk)

## Next Step

ไปต่อที่ [บทที่ 2: การเตรียมข้อมูลและ Training](../02_Data_Training/README.md)
