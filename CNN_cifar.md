# CNN — CIFAR-10 (Step-by-Step Improvements)

Train and iteratively improve a **Convolutional Neural Network (CNN)** on **CIFAR-10** with TensorFlow/Keras.  
You’ll start from a simple baseline, then add **normalization**, **dropout**, **depth**, **batch normalization**, and **data augmentation** while tracking validation curves and accuracy.

---

## 🎯 Goal

- Build a working CNN for CIFAR-10 (10 classes, 32×32 RGB).
- Understand **overfitting** and fix it with standard techniques.
- Read **learning curves** and reason about generalization.

---

## 📦 Dataset & Runtime

- **Dataset:** CIFAR-10 (downloaded & extracted by the notebook).
- **Shapes:** Images are resized/kept at **(32, 32, 3)**; labels are **one-hot**.
- **Tip:** Enable **GPU**; the first cell prints available GPUs.

### How data is loaded (provided)

```python
def unpickle(filename):
    # returns (images, labels) with images as (N, 32, 32, 3) and one-hot labels

train_images, train_labels = ...
test_images,  test_labels  = ...
