# บทที่ 4: RNN/LSTM และการประมวลผล Sequential Data

## เป้าหมายการเรียนรู้

- เข้าใจหลักการทำงานของ Recurrent Neural Networks (RNN)
- รู้จัก LSTM และ GRU สำหรับแก้ปัญหา Long-term Dependencies
- สามารถประมวลผล Time Series Data และ Text ได้
- เข้าใจพื้นฐาน NLP (Natural Language Processing)

## เนื้อหา

### 4.1 Introduction to Sequential Data

**Sequential Data คือ?**
- ข้อมูลที่มีลำดับเวลา (order matters)
- ข้อมูลก่อนหน้ามีผลกับข้อมูลถัดไป

**ตัวอย่าง:**
- Text / Language
- Time Series (ราคาหุ้น, อุณหภูมิ)
- Audio / Speech
- Video
- DNA Sequences

**ทำไม CNN/Dense NN ไม่เหมาะ?**
- ไม่สามารถจำ context ก่อนหน้าได้
- Input size ต้องคงที่
- ไม่เห็น temporal patterns

### 4.2 Recurrent Neural Networks (RNN)

**หลักการ:**
- มี "Memory" เก็บข้อมูลจากก่อนหน้า
- ใช้ output กลับมาเป็น input ในรอบถัดไป
- แชร์ weights ในทุก time step

**โครงสร้าง:**
```
Input: x₁, x₂, x₃, ..., xₜ
Hidden: h₀ → h₁ → h₂ → h₃ → ... → hₜ
Output: y₁, y₂, y₃, ..., yₜ

hₜ = tanh(Wₕₕ * hₜ₋₁ + Wₓₕ * xₜ + b)
yₜ = Wₕᵧ * hₜ
```

**ประเภท RNN:**

1. **One-to-One:** แบบ standard NN
2. **One-to-Many:** Image → Caption
3. **Many-to-One:** Sentiment Analysis
4. **Many-to-Many:** Translation, Video Classification

**ปัญหาของ RNN:**

1. **Vanishing Gradient**
   - Gradient หายไปเมื่อ backprop ไกล
   - ไม่สามารถเรียนรู้ long-term dependencies

2. **Exploding Gradient**
   - Gradient โตเร็วมาก
   - แก้ด้วย Gradient Clipping

### 4.3 Long Short-Term Memory (LSTM)

**ทำไมต้อง LSTM?**
- แก้ปัญหา Vanishing Gradient
- สามารถจำ long-term dependencies ได้
- มี "Cell State" เป็น memory

**โครงสร้าง LSTM:**

มี 3 Gates ควบคุมการไหลของข้อมูล:

1. **Forget Gate** - ลืมข้อมูลเก่า
   ```
   fₜ = σ(Wf * [hₜ₋₁, xₜ] + bf)
   ```

2. **Input Gate** - เพิ่มข้อมูลใหม่
   ```
   iₜ = σ(Wi * [hₜ₋₁, xₜ] + bi)
   c̃ₜ = tanh(Wc * [hₜ₋₁, xₜ] + bc)
   ```

3. **Output Gate** - ส่งออกข้อมูล
   ```
   oₜ = σ(Wo * [hₜ₋₁, xₜ] + bo)
   ```

**Cell State Update:**
```
cₜ = fₜ * cₜ₋₁ + iₜ * c̃ₜ
hₜ = oₜ * tanh(cₜ)
```

**การใช้งานใน Keras:**
```python
from tensorflow.keras.layers import LSTM

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(timesteps, features)),
    LSTM(64),
    Dense(1)
])
```

### 4.4 Gated Recurrent Unit (GRU)

**ความแตกต่างจาก LSTM:**
- ง่ายกว่า (2 gates แทน 3)
- Parameters น้อยกว่า
- เร็วกว่า
- Performance ใกล้เคียง LSTM

**โครงสร้าง:**

1. **Reset Gate**
   ```
   rₜ = σ(Wr * [hₜ₋₁, xₜ])
   ```

2. **Update Gate**
   ```
   zₜ = σ(Wz * [hₜ₋₁, xₜ])
   ```

**Hidden State:**
```
h̃ₜ = tanh(W * [rₜ * hₜ₋₁, xₜ])
hₜ = (1 - zₜ) * hₜ₋₁ + zₜ * h̃ₜ
```

**การใช้งาน:**
```python
from tensorflow.keras.layers import GRU

model = Sequential([
    GRU(128, return_sequences=True),
    GRU(64),
    Dense(1)
])
```

**LSTM vs GRU:**
- LSTM: มี parameters มากกว่า, ซับซ้อน
- GRU: เร็วกว่า, เหมาะกับข้อมูลน้อย
- ทดลองทั้งสองแล้วเลือกที่ดีกว่า

### 4.5 Time Series Analysis

**Workflow:**

```python
# 1. Prepare Data
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# 2. Build Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(seq_length, features)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

# 3. Compile
model.compile(optimizer='adam', loss='mse')

# 4. Train
model.fit(X_train, y_train, epochs=50, batch_size=32)

# 5. Predict
predictions = model.predict(X_test)
```

**Applications:**
- Stock Price Prediction
- Weather Forecasting
- Energy Consumption
- Traffic Prediction

### 4.6 Text Processing and NLP

#### 4.6.1 Text Preprocessing

```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Tokenization
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# 2. Padding
padded_sequences = pad_sequences(sequences, maxlen=100)

# 3. Embedding
embedding_layer = Embedding(
    input_dim=10000,    # vocabulary size
    output_dim=128,     # embedding dimension
    input_length=100    # sequence length
)
```

#### 4.6.2 Word Embeddings

**ทำไมต้องใช้?**
- แปลงคำเป็นตัวเลข (vector)
- คำที่มีความหมายใกล้กัน → vector ใกล้กัน
- จับ semantic relationships

**วิธีการ:**

1. **One-Hot Encoding** (ไม่แนะนำ)
   - Vector ขนาดใหญ่
   - ไม่มี semantic meaning

2. **Word2Vec**
   - Pre-trained embeddings
   - CBOW / Skip-gram

3. **GloVe**
   - Global Vectors
   - Pre-trained

4. **Learned Embeddings**
   ```python
   Embedding(vocab_size, embedding_dim)
   ```

#### 4.6.3 Sentiment Analysis Example

```python
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_length),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary: positive/negative
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

#### 4.6.4 Text Generation

```python
# Character-level RNN
model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(seq_length, vocab_size)),
    Dropout(0.2),
    LSTM(256),
    Dropout(0.2),
    Dense(vocab_size, activation='softmax')
])

# Generate text
def generate_text(model, start_text, length=100):
    for _ in range(length):
        encoded = tokenizer.texts_to_sequences([start_text])
        padded = pad_sequences(encoded, maxlen=seq_length)
        prediction = model.predict(padded)
        next_char = np.argmax(prediction)
        start_text += tokenizer.index_word[next_char]
    return start_text
```

### 4.7 Bidirectional RNN

**แนวคิด:**
- อ่านข้อมูลทั้ง forward และ backward
- เห็น context ทั้งก่อนและหลัง
- เหมาะกับ NLP

```python
from tensorflow.keras.layers import Bidirectional

model = Sequential([
    Embedding(vocab_size, 128),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dense(num_classes, activation='softmax')
])
```

## ไฟล์ในบทนี้

- `01_rnn_basics.ipynb` - ทำความเข้าใจ RNN
- `02_lstm_vs_gru.ipynb` - เปรียบเทียบ LSTM และ GRU
- `03_time_series_prediction.ipynb` - ทำนาย Time Series
- `04_text_preprocessing.ipynb` - เตรียมข้อมูล Text
- `05_sentiment_analysis.ipynb` - วิเคราะห์ความรู้สึก
- `06_text_generation.ipynb` - สร้าง Text
- `exercises.ipynb` - แบบฝึกหัด

## แบบฝึกหัด

### Exercise 1: Stock Price Prediction
- ใช้ LSTM ทำนายราคาหุ้น
- ใช้ข้อมูลย้อนหลัง 60 วัน
- Visualize predictions vs actual

### Exercise 2: Movie Review Sentiment Classification
- จำแนกรีวิวหนังเป็น positive/negative
- ใช้ IMDB dataset
- เปรียบเทียบ LSTM, GRU, Bidirectional

## Resources

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

## Next Step

ไปต่อที่ [บทที่ 5: โปรเจกต์จริงและการ Deploy](../05_Projects_Deployment/README.md)
