# CNN on CIFAR‑10 — Assignment Notebook (Keras/TensorFlow)

This notebook incrementally builds a **convolutional neural network (CNN)** for **CIFAR‑10** image classification and teaches practical techniques to improve generalization: **normalization**, **dropout**, **deeper architectures**, **batch normalization**, and **data augmentation**. You will also **analyze learning curves** and answer short conceptual questions (CQ1–CQ11).

---

## 🧰 Setup & Dataset

**Goal:** Train CNNs on CIFAR‑10 (60k RGB images, 10 classes, 32×32px).

### What the notebook does for you
- **GPU check:** Confirms a GPU is available via `tf.config.list_physical_devices('GPU')`.
- **Download & unpack CIFAR‑10** (`cifar-10-batches-py`).
- **Load & preprocess:**
  - Reshape raw arrays to `(32, 32, 3)` per image.
  - **One‑hot encode** labels via `tf.keras.utils.to_categorical`.
  - **Merge** 5 training batches into a single training set.
- **Helper**: `train_and_evaluate(model, train_x, train_y, val_x, val_y, preprocess={}, epochs=20, augment={})`
  - Compiles with **CategoricalCrossentropy**, **Adam**, and **accuracy**.
  - Builds **ImageDataGenerator** objects:
    - `train_gen = ImageDataGenerator(**preprocess, **augment)` (fit on **train** data)
    - `val_gen   = ImageDataGenerator(**preprocess)` (fit on **train** data; **no augmentation**)
  - Trains for `epochs`, **plots loss/accuracy curves** (train vs val), and prints **final validation accuracy**.

> Why fit the validation generator on *training* statistics and avoid augmentation in validation? You’ll explain this in **CQ3** and **CQ10**.

---

## 🧱 Baseline Model (Provided)

A simple CNN is given to get you started:
- `Conv2D(32, 3×3, relu, padding="same")` → `MaxPooling2D(2×2)`
- `Conv2D(64, 3×3, relu, padding="same")` → `MaxPooling2D(2×2)`
- `Flatten` → `Dense(256, relu)` → `Dense(10, softmax)`
- Trained with `train_and_evaluate(...)` to produce **baseline curves**.

You’ll use these curves to reason about **overfitting/underfitting** (see **CQ1**).

---

## 🙋 Collaborative Questions (answer in the “*Your answer goes here.*” cells)

- **CQ1.** Why is **training loss** normally **lower** than **validation loss**?
- **CQ2.** Why does **feature normalization** (mean‑centering pixels) matter, esp. with **ReLU** networks?
- **CQ3.** Why must **the same normalization statistics** be used for both train and validation? What if not?
- **CQ4.** Which part of the **baseline curves** indicated **overfitting**?
- **CQ5.** In your own words, why does **dropout** reduce overfitting risk?
- **CQ6.** Which layer has the **largest total output volume** (total number of activations)? Why?
- **CQ7.** Which layer has the **most trainable parameters** (weights)? Why?
- **CQ8.** Why does **Batch Normalization** improve deep networks?
- **CQ9.** BN has learnable **scale** and **shift**. Are these still useful just **before ReLU**? Why/why not?
- **CQ10.** Why apply **augmentation only to training** and **not** to validation? What would go wrong otherwise?
- **CQ11.** Which augmentations did you try? Which helped the most for **CIFAR‑10**, and why?

---

## 📝 What Each “YOUR CODE HERE” Exercise Requires

Below is a **cell‑by‑cell** guide to every coding task. There are **five** `# YOUR CODE HERE` cells—one per assignment.

### 1) Assignment 1 — **Normalizing the input data**
**Where:** The `YOUR CODE HERE` at the end of *“Assignment 1: Normalizing the input data”* section.  
**Do this:**
- **Copy the baseline model** (same layers/structure).
- Create a `preprocess` **dict** enabling **mean‑centering**:
  ```python
  preprocess = {"featurewise_center": True}
  ```
- Call `train_and_evaluate(model, train_images, train_labels, test_images, test_labels, preprocess=preprocess, epochs=...)`.
- Ensure both **train** and **validation** use the **same** `preprocess` stats (the helper already fits on **train**).  
**Outcome:** Improved validation accuracy (≈ **71%**) and better‑behaved curves.  
**Discuss:** **CQ2, CQ3.**

---

### 2) Assignment 2 — **Adding Dropout**
**Where:** The `YOUR CODE HERE` right under *“Assignment 2: Adding Dropout”*.  
**Do this:**
- **Start from Assignment 1’s model**.
- Insert **Dropout** layers to fight overfitting, e.g.:
  - After each **conv‑pool block** (e.g., `Dropout(0.25)`).
  - Before the final **Dense(256)** or between Dense layers (e.g., `Dropout(0.5)`).  
- Retrain with the same preprocessing.  
**Outcome:** Higher validation accuracy (≈ **75%**), reduced train‑val gap.  
**Discuss:** **CQ4, CQ5.**

---

### 3) Assignment 3 — **Making the network deeper**
**Where:** The first `YOUR CODE HERE` in *“Assignment 3: Making the network deeper”* (followed by a `model.summary()` cell).  
**Do this:**
- **Start from Assignment 2’s model**.
- **Add two more conv‑pool blocks** to learn more complex features:
  - Mirror the last conv block’s structure (same kernel size/activation), typically increasing filters (e.g., `Conv2D(128, ...)` then `MaxPooling2D`, then `Conv2D(256, ...)` then `MaxPooling2D`) or follow the notebook’s guidance “same structure as the last conv layer”.
- Retrain; then run the provided `model.summary()` cell.  
**Outcome:** Better accuracy (≈ **79%**); **longer training** due to depth.  
**Discuss:** Use summary to answer **CQ6** (largest activation volume) and **CQ7** (most parameters).

---

### 4) Assignment 4 — **Adding Batch Normalization**
**Where:** The `YOUR CODE HERE` in *“Assignment 4: Adding Batch Normalization”*.  
**Do this:**
- **Start from the deeper model** (Assignment 3).
- Insert `layers.BatchNormalization()` around conv/dense layers. Common patterns:
  - `Conv2D → BatchNormalization → ReLU/Activation`
  - Optionally also before the Dense layer(s).
- Keep dropout as appropriate and retrain.  
**Outcome:** Another bump in validation accuracy (≈ **84%**), more stable/fast convergence.  
**Discuss:** **CQ8, CQ9** (placement relative to ReLU, role of BN’s scale/shift).

---

### 5) Assignment 5 — **Data Augmentation**
**Where:** There are **two** code cells in this section:
1. A `YOUR CODE HERE` to **configure augmentations**.
2. A `YOUR CODE HERE` to **train using those augmentations**.

**Do this:**
- Build an `augment` **dict** with `ImageDataGenerator` options (examples):
  ```python
  augment = {
      "horizontal_flip": True,
      "width_shift_range": 0.1,
      "height_shift_range": 0.1,
      "zoom_range": 0.1,
      "rotation_range": 15,
      "shear_range": 0.1,
      "fill_mode": "nearest",
  }
  ```
- Call
  ```python
  train_and_evaluate(
      model, train_images, train_labels, test_images, test_labels,
      preprocess=preprocess, augment=augment, epochs=...
  )
  ```
- **Do not** augment validation data (the helper already enforces this).  
**Outcome:** Sometimes modest accuracy gains and improved robustness; training may take longer.  
**Discuss:** **CQ10** (train‑only augmentation) and **CQ11** (what you tried; what helped on CIFAR‑10—e.g., flips, small shifts/rotations/zoom).

---

## ✅ What You’ll Learn

- How to **read/prepare CIFAR‑10**, and why **consistent normalization** matters.
- How to build and iterate on CNNs: **baseline → dropout → deeper → batch norm → augmentation**.
- How to **interpret learning curves** to detect **over/underfitting**.
- Practical Keras tooling: `ImageDataGenerator`, `model.summary()`, metrics/plots.

---

## 📎 Notes & Tips

- Keep epochs moderate at first to iterate faster; increase once your design stabilizes.
- When you change **model capacity** (depth/filters), monitor both **loss** and **accuracy** curves.
- Typical dropout rates: **0.25** after conv blocks, **0.5** before dense layers (tune as needed).
- BN generally goes **before ReLU** in modern practice; be consistent within a block.
- Set random seeds if you need **reproducible runs**; small fluctuations are normal.

---

## 🗺️ Map of “YOUR CODE HERE” cells

| Section | What you implement |
|---|---|
| **Assignment 1** | Define `preprocess = {"featurewise_center": True}`; train with `train_and_evaluate(..., preprocess=preprocess)` |
| **Assignment 2** | Insert `Dropout` layers into the model and retrain |
| **Assignment 3** | Deepen the CNN with two more conv‑pool blocks; retrain; run `model.summary()` (provided) |
| **Assignment 4** | Add `BatchNormalization` layers around conv/dense blocks; retrain |
| **Assignment 5** | (1) Build `augment` dict; (2) Train with `augment=augment` |

Good luck—and have fun pushing that validation accuracy up! 🚀
