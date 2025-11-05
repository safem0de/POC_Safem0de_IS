# Resources - แหล่งข้อมูลและเอกสารอ้างอิง

## 📚 หนังสือแนะนำ

### ภาษาอังกฤษ
1. **Deep Learning** - Ian Goodfellow, Yoshua Bengio, Aaron Courville
   - [Free Online Version](https://www.deeplearningbook.org/)
   - Bible ของ Deep Learning

2. **Hands-On Machine Learning** - Aurélien Géron
   - Practical และมี code examples
   - ใช้ TensorFlow และ Scikit-Learn

3. **Deep Learning with Python** - François Chollet (ผู้สร้าง Keras)
   - เน้น Keras
   - เหมาะกับมือใหม่

### ภาษาไทย
1. **ปัญญาประดิษฐ์และการเรียนรู้ของเครื่อง** - ดร. ชัยพร ภิรมย์รื่น
2. **Deep Learning เบื้องต้น** - นพ.ภาคภูมิ ภู่ประเสริฐ

## 🎓 Online Courses

### ฟรี
1. **[Deep Learning Specialization - Andrew Ng](https://www.coursera.org/specializations/deep-learning)**
   - Coursera (มี Financial Aid)
   - ครอบคลุมทุกด้าน

2. **[Fast.ai - Practical Deep Learning](https://course.fast.ai/)**
   - เน้น Practical
   - Top-down approach

3. **[TensorFlow in Practice](https://www.coursera.org/specializations/tensorflow-in-practice)**
   - เน้น TensorFlow
   - มี hands-on projects

4. **[CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/)**
   - Stanford University
   - Video lectures ฟรี

5. **[CS224n: NLP with Deep Learning](http://web.stanford.edu/class/cs224n/)**
   - Stanford University
   - NLP และ Transformers

### เสียเงิน
1. **Udacity Deep Learning Nanodegree**
2. **DataCamp Deep Learning Track**

## 🎥 YouTube Channels

1. **[3Blue1Brown](https://www.youtube.com/@3blue1brown)**
   - Neural Networks series
   - Visualization สวยมาก

2. **[Sentdex](https://www.youtube.com/@sentdex)**
   - Python และ Deep Learning

3. **[StatQuest with Josh Starmer](https://www.youtube.com/@statquest)**
   - อธิบายง่ายเข้าใจ

4. **[Two Minute Papers](https://www.youtube.com/@TwoMinutePapers)**
   - Research papers สรุป

5. **[Yannic Kilcher](https://www.youtube.com/@YannicKilcher)**
   - Paper reviews

## 📝 Blogs และ Websites

1. **[Towards Data Science](https://towardsdatascience.com/)**
   - Articles และ tutorials

2. **[Machine Learning Mastery](https://machinelearningmastery.com/)**
   - Practical guides

3. **[Distill.pub](https://distill.pub/)**
   - Research papers แบบ interactive

4. **[Papers With Code](https://paperswithcode.com/)**
   - Research papers + code

5. **[colah's blog](https://colah.github.io/)**
   - Visualization ดีมาก

## 🛠️ Tools และ Libraries

### Deep Learning Frameworks
```bash
# TensorFlow/Keras
pip install tensorflow

# PyTorch
pip install torch torchvision

# JAX
pip install jax jaxlib
```

### Visualization
```bash
pip install matplotlib seaborn plotly
pip install tensorboard
```

### Data Processing
```bash
pip install numpy pandas scikit-learn
pip install opencv-python pillow
```

### Deployment
```bash
pip install fastapi uvicorn
pip install streamlit gradio
pip install flask
```

## 📊 Datasets

### Image Datasets
1. **MNIST** - Handwritten digits (Built-in Keras)
2. **CIFAR-10/100** - 10/100 classes (Built-in Keras)
3. **ImageNet** - 1000 classes, 14M images
4. **COCO** - Object detection
5. **Open Images** - Google's dataset
6. **Fashion MNIST** - Clothing items

### Text Datasets
1. **IMDB Reviews** - Sentiment analysis (Built-in Keras)
2. **20 Newsgroups** - Text classification
3. **WikiText** - Language modeling
4. **SQuAD** - Question answering

### Time Series
1. **Stock Market Data** - Yahoo Finance
2. **Weather Data** - OpenWeatherMap
3. **UCI ML Repository** - Various datasets

### Sources
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI ML Repository](https://archive.ics.uci.edu/ml/index.php)
- [Google Dataset Search](https://datasetsearch.research.google.com/)
- [Hugging Face Datasets](https://huggingface.co/datasets)

## 🧰 Interactive Tools

1. **[TensorFlow Playground](https://playground.tensorflow.org/)**
   - Visualize neural networks

2. **[CNN Explainer](https://poloclub.github.io/cnn-explainer/)**
   - How CNNs work

3. **[GAN Lab](https://poloclub.github.io/ganlab/)**
   - Interactive GAN training

4. **[Embedding Projector](https://projector.tensorflow.org/)**
   - Visualize embeddings

## 📰 Research Papers (Must Read)

### Foundational
1. **ImageNet Classification** - AlexNet (2012)
2. **Deep Residual Learning** - ResNet (2015)
3. **Attention Is All You Need** - Transformer (2017)
4. **BERT** - Bidirectional Transformers (2018)
5. **GPT-3** - Language Models (2020)

### Resources
- [arXiv.org](https://arxiv.org/) - Preprints
- [OpenReview](https://openreview.net/) - Peer reviews
- [Semantic Scholar](https://www.semanticscholar.org/) - Search

## 🎮 Practice Platforms

1. **[Kaggle](https://www.kaggle.com/)**
   - Competitions
   - Datasets
   - Notebooks

2. **[Google Colab](https://colab.research.google.com/)**
   - Free GPU/TPU
   - Jupyter notebooks

3. **[Paperspace Gradient](https://www.paperspace.com/gradient)**
   - Cloud GPUs

4. **[LeetCode](https://leetcode.com/)**
   - Coding practice

## 🌐 Communities

1. **Reddit:**
   - r/MachineLearning
   - r/deeplearning
   - r/learnmachinelearning

2. **Discord:**
   - TensorFlow Community
   - PyTorch Community
   - fast.ai

3. **Forums:**
   - Stack Overflow
   - Cross Validated (Stats Stack Exchange)

## 🔧 Development Tools

### IDEs
- **Jupyter Notebook/Lab**
- **VS Code** + Python extension
- **PyCharm**
- **Google Colab**

### Version Control
```bash
# Git basics
git init
git add .
git commit -m "message"
git push

# DVC (Data Version Control)
pip install dvc
dvc init
dvc add data/
```

### Experiment Tracking
```bash
# MLflow
pip install mlflow
mlflow ui

# Weights & Biases
pip install wandb

# TensorBoard
tensorboard --logdir=logs/
```

## 📖 Cheat Sheets

1. **[Keras Cheat Sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Keras_Cheat_Sheet_Python.pdf)**
2. **[NumPy Cheat Sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf)**
3. **[Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)**

## 🎯 Project Ideas

### Beginner
1. MNIST Digit Recognition
2. Fashion Item Classification
3. Sentiment Analysis
4. House Price Prediction

### Intermediate
1. Image Captioning
2. Object Detection
3. Named Entity Recognition
4. Music Generation

### Advanced
1. Style Transfer
2. GANs for Image Generation
3. Chatbot with Transformers
4. Reinforcement Learning Game AI

## 📱 Mobile ML

1. **TensorFlow Lite** - Android/iOS
2. **Core ML** - iOS
3. **ML Kit** - Firebase
4. **ONNX Runtime** - Cross-platform

## 🔬 Research Conferences

- **NeurIPS** - Neural Information Processing Systems
- **ICML** - International Conference on Machine Learning
- **ICLR** - International Conference on Learning Representations
- **CVPR** - Computer Vision and Pattern Recognition
- **ACL** - Association for Computational Linguistics

## 💡 Tips for Learning

1. **Code Every Day** - Practice makes perfect
2. **Read Papers** - Stay updated
3. **Join Competitions** - Kaggle, hackathons
4. **Build Projects** - Apply what you learn
5. **Share Knowledge** - Blog, GitHub
6. **Network** - Join communities

## 🆘 Getting Help

1. **Stack Overflow** - Questions
2. **GitHub Issues** - Library bugs
3. **Discord/Slack** - Real-time help
4. **Reddit** - Discussions

## 🔗 Useful Links

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [PyTorch Documentation](https://pytorch.org/)
- [Scikit-Learn Documentation](https://scikit-learn.org/)
- [NumPy Documentation](https://numpy.org/)
- [Pandas Documentation](https://pandas.pydata.org/)

---

**อัปเดตล่าสุด:** 2025-01-05

**หมายเหตุ:** Resources นี้จะมีการอัปเดตเป็นประจำ ติดตามที่ GitHub repository
