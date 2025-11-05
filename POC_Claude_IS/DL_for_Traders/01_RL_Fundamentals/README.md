# 🎮 บทที่ 1: Reinforcement Learning Fundamentals

## 🤖 เรื่องราวของหุ่นยนต์ที่เรียนรู้เอง

ลองจินตนาการว่า...

คุณสร้างหุ่นยนต์ตัวหนึ่ง วางไว้ในห้องที่มีประตูทางออก 🚪

**ปัญหา:** หุ่นยนต์ไม่รู้ว่าประตูอยู่ไหน!

### 🤔 จะสอนยังไง?

#### วิธีที่ 1: Supervised Learning (ไม่ได้ผล!)
- ต้องบอกว่า "เดินซ้าย", "เดินขวา", "เดินบน"... ทุกก้าว
- ต้องรู้คำตอบที่ถูกต้อง (Labels) ล่วงหน้า
- **ปัญหา:** สถานการณ์ใหม่ทำไม่ได้!

#### วิธีที่ 2: Unsupervised Learning (ไม่เหมาะ!)
- ปล่อยให้เดินสุ่ม แล้วหาpatterจาก
- ไม่มี goal ชัดเจน

#### วิธีที่ 3: Reinforcement Learning (ใช่เลย! ✅)
- **ไม่บอกคำตอบ** แต่บอก **ผลลัพธ์**
- เดินถูกทาง → ได้รางวัล (+1) 🎁
- เดินผิดทาง → ถูกลงโทษ (-1) ⚠️
- หุ่นยนต์เรียนรู้เอง จากการ **ลองผิดลองถูก**!

นี่คือแนวคิดของ **Reinforcement Learning**! 🚀

---

## 📚 เนื้อหาในบทนี้

### 1.1 RL Concepts พื้นฐาน
- Agent, Environment, State, Action, Reward
- Episode และ Trajectory
- Return และ Discount Factor

### 1.2 Markov Decision Process (MDP)
- Markov Property คืออะไร?
- Transition Probability
- Policy (π)
- Value Function

### 1.3 Exploration vs Exploitation
- Dilemma ระหว่างสำรวจ vs ใช้ประโยชน์
- Epsilon-Greedy Strategy

### 1.4 Grid World Example
- สร้าง Environment เอง
- Implement Random Agent
- Implement Rule-based Agent

---

## 🎯 Part 1.1: RL Concepts

### ส่วนประกอบหลักของ RL:

```
┌─────────────────────────────────────────┐
│                                         │
│    ┌───────┐        ┌──────────────┐   │
│    │ Agent │◄───────┤ Observation  │   │
│    │       │        │   (State)    │   │
│    └───┬───┘        └──────────────┘   │
│        │                    ▲           │
│        │ Action             │           │
│        ▼                    │           │
│    ┌─────────────────────────────────┐ │
│    │       Environment               │ │
│    │  (โลกที่ Agent อยู่)             │ │
│    └─────────────────────────────────┘ │
│                    │                    │
│                    ▼                    │
│             ┌────────────┐              │
│             │  Reward    │              │
│             └────────────┘              │
│                                         │
└─────────────────────────────────────────┘
```

### คำศัพท์สำคัญ:

#### 1. **Agent** (ตัวกระทำ)
- หุ่นยนต์, AI, Trading Bot
- **ทำหน้าที่:** ตัดสินใจและกระทำ

#### 2. **Environment** (สภาพแวดล้อม)
- โลกที่ Agent อยู่
- Grid World, Trading Market, Game
- **ทำหน้าที่:** ตอบสนองต่อ Action

#### 3. **State (S)**
- สถานะปัจจุบัน
- ตัวอย่าง Grid World: ตำแหน่ง (x, y)
- ตัวอย่าง Trading: ราคา, Volume, Indicators

#### 4. **Action (A)**
- การกระทำที่เลือกได้
- Grid World: ↑ ↓ ← →
- Trading: Buy, Sell, Hold

#### 5. **Reward (R)**
- **สิ่งที่สำคัญที่สุด!**
- ผลลัพธ์ที่ได้รับหลังทำ Action
- Grid World:
  - ถึงประตู = +10
  - ตกหลุม = -10
  - เดินปกติ = -1 (penalty เวลา)
- Trading:
  - ทำกำไร = +profit
  - ขาดทุน = -loss

---

## 🎮 Grid World Example

ลองดูตัวอย่าง Grid World 5x5:

```
┌───┬───┬───┬───┬───┐
│ S │   │   │   │   │  S = Start (Agent เริ่มต้น)
├───┼───┼───┼───┼───┤
│   │ X │   │ X │   │  X = Hole (ตกหลุม -10)
├───┼───┼───┼───┼───┤
│   │   │   │   │   │
├───┼───┼───┼───┼───┤
│   │ X │   │   │   │
├───┼───┼───┼───┼───┤
│   │   │   │   │ G │  G = Goal (ประตู +10)
└───┴───┴───┴───┴───┘
```

### กฎของเกม:
- Agent เริ่มที่ S (0, 0)
- เป้าหมาย: ไปถึง G (4, 4)
- Actions: ↑ ↓ ← →
- Rewards:
  - ถึง Goal: +10
  - ตก Hole: -10
  - แต่ละก้าว: -1 (encourage เดินทางสั้น)

### ตัวอย่าง Episode:

```
Step 1: S → (เดินขวา) → State=(0,1), Reward=-1
Step 2: (0,1) → (เดินลง) → State=(1,1), Reward=-10 ❌ (ตกหลุม!)

Episode จบ! Total Reward = -11
```

```
Step 1: S → (เดินลง) → State=(1,0), Reward=-1
Step 2: (1,0) → (เดินลง) → State=(2,0), Reward=-1
Step 3: (2,0) → (เดินขวา) → State=(2,1), Reward=-1
... (เดินต่อ)
Step 10: (4,3) → (เดินขวา) → State=(4,4), Reward=+10 ✅ (ถึงเป้าหมาย!)

Episode จบ! Total Reward = -9 + 10 = +1
```

---

## 📊 Part 1.2: Markov Decision Process (MDP)

### MDP คืออะไร?

**Mathematical framework** สำหรับ RL

ประกอบด้วย 5 ส่วน: (S, A, P, R, γ)

#### 1. **S** = Set of States
- ทุก state ที่เป็นไปได้
- Grid World: {(0,0), (0,1), ..., (4,4)}

#### 2. **A** = Set of Actions
- ทุก action ที่ทำได้
- Grid World: {↑, ↓, ←, →}

#### 3. **P** = Transition Probability
- P(s' | s, a) = โอกาสที่จะไปถึง state s' เมื่อทำ action a ที่ state s

Grid World (deterministic):
- P(s' | s, "↑") = 1.0 ถ้า s' คือช่องด้านบนของ s
- P(s' | s, "↑") = 0.0 ถ้าไม่ใช่

Stochastic Grid World (มี noise):
- P(s' | s, "↑") = 0.8 (ไปด้านบนจริงๆ)
- P(s' | s, "↑") = 0.1 (ไปซ้าย)
- P(s' | s, "↑") = 0.1 (ไปขวา)

#### 4. **R** = Reward Function
- R(s, a, s') = Reward ที่ได้หลังทำ action a ที่ state s และไปถึง s'

#### 5. **γ** (Gamma) = Discount Factor
- **สำคัญมาก!** ค่า 0 ถึง 1
- ลด importance ของ reward ในอนาคต

**ทำไมต้องมี Discount?**

```
รางวัลวันนี้ 100 บาท vs รางวัลปีหน้า 100 บาท
→ คนส่วนใหญ่เลือกวันนี้! (Time value of money)
```

**Discount Factor:**
- γ = 0: Agent สนใจแค่ immediate reward (สายกำไรด่วน)
- γ = 0.9: Agent คิดถึงอนาคต 10 steps
- γ = 1.0: Agent คิดถึงอนาคตทั้งหมด (ไม่ discount)

---

## 🎯 Part 1.3: Return และ Value Function

### Return (G)

**Total discounted reward** ที่จะได้รับในอนาคต

```
G_t = R_{t+1} + γ×R_{t+2} + γ²×R_{t+3} + γ³×R_{t+4} + ...
    = R_{t+1} + γ×G_{t+1}
```

**ตัวอย่าง:**
- Rewards: [1, 2, 3, 4, 5]
- γ = 0.9

```
G = 1 + 0.9×2 + 0.9²×3 + 0.9³×4 + 0.9⁴×5
  = 1 + 1.8 + 2.43 + 2.916 + 3.28
  = 11.426
```

### Value Function V(s)

**Expected Return** เมื่ออยู่ที่ state s และทำตาม policy π

```
V^π(s) = E[G_t | S_t = s]
```

**ความหมาย:** "ถ้าอยู่ที่ state s แล้วทำตาม policy π ต่อไป จะได้ reward รวมเท่าไหร่?"

### Action-Value Function Q(s, a)

**Expected Return** เมื่ออยู่ที่ state s, ทำ action a, แล้วทำตาม policy π

```
Q^π(s, a) = E[G_t | S_t = s, A_t = a]
```

**ความหมาย:** "ถ้าอยู่ที่ state s ทำ action a แล้วทำตาม policy π ต่อไป จะได้ reward รวมเท่าไหร่?"

**นี่คือ Q ใน Q-Learning!** 🎯

---

## 🤖 Part 1.4: Policy (π)

### Policy คืออะไร?

**กลยุทธ์** ในการเลือก Action

```
π(a | s) = โอกาสที่จะเลือก action a เมื่ออยู่ที่ state s
```

### ประเภท Policy:

#### 1. **Deterministic Policy**
```python
π(s) = a
```
- เลือก action เดียวเสมอ
- เช่น: ถ้าเห็นช้าง → วิ่งหนี

#### 2. **Stochastic Policy**
```python
π(a | s) = ความน่าจะเป็น
```
- เลือก action แบบสุ่มตาม probability
- เช่น: เห็นช้าง → วิ่งหนี 70%, อยู่นิ่ง 30%

### Goal ของ RL:

**หา Optimal Policy (π*) ที่ maximize expected return!**

```
π* = argmax V^π(s) สำหรับทุก state s
```

---

## ⚖️ Part 1.5: Exploration vs Exploitation

### Dilemma ที่สำคัญที่สุดใน RL!

#### 🔍 Exploration (สำรวจ)
- ลองสิ่งใหม่ๆ
- หาทางที่ดีกว่า
- **Risk:** อาจเจอทางที่แย่กว่า

#### 🎯 Exploitation (ใช้ประโยชน์)
- ทำสิ่งที่รู้ว่าดี
- ใช้ความรู้ที่มี
- **Risk:** อาจพลาดทางที่ดีกว่า

### ตัวอย่าง: ร้านอาหาร 🍕🍔🍜

คุณเคยกินร้าน A (อร่อยแน่นอน 7/10)

มี ร้าน B ใหม่ (ยังไม่รู้ อาจอร่อย 9/10 หรือ 3/10)

**คำถาม:** จะกินร้านไหน?

- **Exploitation:** กิน A (แน่ใจ 7/10)
- **Exploration:** ลอง B (อาจได้ 9/10 หรือ 3/10)

### วิธีแก้: Epsilon-Greedy (ε-greedy)

```python
ε = 0.1  # 10% explore

if random() < ε:
    action = random_action()  # Explore (10%)
else:
    action = best_action()    # Exploit (90%)
```

**ปรับ ε ตามเวลา:**
- เริ่มต้น: ε = 1.0 (explore 100%)
- กลางทาง: ε = 0.5 (explore 50%)
- ท้ายสุด: ε = 0.01 (exploit 99%)

---

## 📝 ไฟล์ในบทนี้

- `01_grid_world_basics.ipynb` - สร้าง Grid World
- `02_mdp_concepts.ipynb` - MDP และ Bellman Equation
- `03_value_functions.ipynb` - คำนวณ V(s) และ Q(s,a)
- `04_policies.ipynb` - เปรียบเทียบ Policy ต่างๆ
- `05_exploration_exploitation.ipynb` - ε-greedy experiments
- `exercises.ipynb` - แบบฝึกหัด

---

## 🎯 แบบฝึกหัด

### Exercise 1: Grid World Simulation
สร้าง Grid World 10x10 ที่มี:
- Start: (0, 0)
- Goal: (9, 9)
- Holes: 5 ตำแหน่งสุ่ม
- คำนวณ optimal path

### Exercise 2: Return Calculation
คำนวณ Return G_t จาก reward sequence:
```
Rewards = [5, -2, 3, 1, 10]
γ = 0.95
```

### Exercise 3: Policy Comparison
เปรียบเทียบ 3 policies:
1. Random Policy
2. Always go Right/Down
3. Shortest Path Policy

---

## 🚀 Next Step

ตอนนี้คุณเข้าใจพื้นฐาน RL แล้ว!

👉 [บทที่ 2: Q-Learning Basics](../02_Q_Learning_Basics/README.md)

เรียนรู้วิธีสอน Agent ให้เลือกทางที่ดีที่สุดด้วย Q-Learning!
