# Practice 6: MNIST Digit Classification

<br/>

## Overview
- MNIST handwritten digits dataset for supervised learning (classification), deep learning (CNN), unsupervised learning (clustering), and visualization practice.
- Use scikit-learn's fetch_openml to load the dataset and apply various algorithms including RandomForest, SVM, CNN, and KMeans.

<br/>

## Dataset Introduction
- **MNIST**: 70,000 handwritten digit images (0-9) with 28x28 pixel resolution
- Source: [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- Features: 784 pixel values (28×28 flattened)
- Target: Digit labels (0-9)

<br/>

## Environment Setup and Execution
```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install packages
pip3 install -r requirements.txt

# 3. Run practice code
python3 mnist_digit_classification.py
```

<br/>

## Main Practice Contents
- Data loading and preprocessing (fetch_openml, data reshaping)
- Supervised Learning: RandomForest, SVM for digit classification, accuracy comparison
- Deep Learning: CNN architecture, training, evaluation
- Unsupervised Learning: KMeans clustering with 10 clusters
- Visualization: Sample digit images, confusion matrix, clustering results
- Model comparison and performance analysis

<br/>

## Result Interpretation
- Supervised Learning: Compare accuracy between RandomForest, SVM, and CNN
- Deep Learning: CNN typically achieves higher accuracy due to spatial feature learning
- Unsupervised Learning: KMeans groups similar digit patterns without labels
- Visualization: Confusion matrix shows which digits are commonly misclassified

<br/>

## Key Learning Points
- **Image Data Processing**: Reshaping 1D arrays to 2D images for CNN
- **Deep Learning Basics**: Convolutional layers, pooling, dropout, dense layers
- **Model Comparison**: Traditional ML vs Deep Learning performance
- **Computer Vision**: Understanding spatial features in image data

<br/>

## Conclusion
- Practice with computer vision's most famous dataset
- Compare traditional machine learning and deep learning approaches
- Understand the power of CNNs for image classification tasks
- Learn data preprocessing techniques for image data

<br/>

## Reference
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow MNIST Tutorial](https://www.tensorflow.org/tutorials/quickstart/beginner)
- [Scikit-learn MNIST](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html)
- [Keras CNN Guide](https://keras.io/examples/vision/mnist_convnet/) 