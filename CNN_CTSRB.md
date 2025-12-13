# CNN — German Traffic Sign Recognition (GTSRB)

Build, train, and evaluate a **Convolutional Neural Network (CNN)** on the **German Traffic Sign Recognition Benchmark (GTSRB)** using **TensorFlow/Keras**. You will implement the CNN end-to-end, try multiple model variants, and report results per version.

---

## 🎯 Goal

- Train a deep CNN that classifies traffic sign images into the correct class.
- Iterate on model design (baseline → improved variants) and **record test accuracy** and changes you made.

---

## 📦 Dataset & Runtime Setup

- **Dataset:** GTSRB training set + a provided “Online-Test (sorted)” split.
- **Notebook downloads & unzips** the archives into a local `GTSRB/` folder.
- **Tip:** Run on **Kaggle** with **GPU acceleration** enabled to speed up training.

What the setup cells do:
1. **Download & unzip**  
   Fetches:
   - `GTSRB-Training_fixed.zip` → `GTSRB/Training/`
   - `GTSRB_Online-Test-Images-Sorted.zip` → `GTSRB/Online-Test-sort/`

2. **Data loader (provided)**  
   - Uses OpenCV to read `.ppm` files.
   - **Resizes** every image to **32×32** RGB.
   - Returns:
     - `train_images, train_labels`
     - `test_images, test_labels`
   - Prints array shapes so you can sanity-check the load.

> You do **not** need to change the downloader or the loader unless you want to. Your work starts in the assignment section below.

---

## 🧠 Topic Covered

- Convolutional Neural Networks for image classification:
  - Input normalization, one-hot labels
  - Conv/Pooling stacks, BatchNorm/Dropout
  - Optimizer/loss/metrics
  - Training loops, validation, testing
  - Simple experiment tracking

---

## ✅ Your Assignment (the TODO)

There is **one required coding cell** marked `# YOUR CODE HERE`. In this cell, you must implement the full classification pipeline and run your experiments.

### Implement in the TODO cell

- **Preprocessing**
  - Scale inputs to `[0, 1]` (e.g., divide by 255.0).
  - Convert integer labels to **one-hot** vectors (e.g., `tf.keras.utils.to_categorical`), making sure the number of classes matches GTSRB.

- **Model (Baseline)**
  - Build a **Keras Sequential/Functional** CNN with:
    - A few `Conv2D` + `MaxPooling2D` blocks (BatchNorm optional).
    - `Flatten` or `GlobalAveragePooling2D`.
    - Dense head ending with **softmax** over `n_classes`.
  - Keep the **input shape** consistent with the loader (32×32, 3 channels).

- **Compile**
  - Loss: `sparse_categorical_crossentropy` **or** `categorical_crossentropy` (consistent with your label format).
  - Optimizer: e.g., `Adam`.
  - Metrics: at least `accuracy`.

- **Train**
  - Choose reasonable `batch_size` and `epochs`.
  - Optionally add **validation_split** or use part of training as a validation set.
  - (Optional) **Data augmentation** with `ImageDataGenerator` / `tf.image`.

- **Evaluate**
  - Report **test accuracy** on `test_images, test_labels`.
  - Print or store key results you’ll paste into the “Version” sections.

- **(Optional, Recommended) Diagnostics**
  - Plot training/validation accuracy & loss curves.
  - Confusion matrix on test set.

---

## 🧪 Experiment Log — “Versions” You Must Fill

The notebook provides markdown sections where you **describe each experiment** you ran and its outcome. Fill these in after each training run.

### #### Version 1  *(Your description goes here.)*
What to write:
- **Architecture summary:** layers, filters, kernel sizes, regularization.
- **Training setup:** epochs, batch size, optimizer, learning rate.
- **Preprocessing:** normalization, one-hot, augmentation (if any).
- **Result:** final **test accuracy**; any notable failure modes.

### #### Version 2  *(Your description goes here.)*
- Explain what you changed from Version 1 and **why** (e.g., added BatchNorm, deeper CNN, Dropout, LR schedule).
- Report **test accuracy** and compare vs. Version 1.

> You can add **Version 3+** sections the same way if you keep iterating. Each version should clearly show **what changed** and **how it impacted** performance.

---

## 🧩 What Each Notebook Section/Cell Requires

1. **Title & Intro (Markdown)**  
   - Context for GTSRB and instructions to run on Kaggle with GPU.

2. **Download & Unzip (Code)**  
   - *No edits required.* Ensures `GTSRB/Training` and `GTSRB/Online-Test-sort` exist.

3. **Loader Explanation (Markdown)**  
   - Describes that images are resized to 32×32 and returned as NumPy arrays.

4. **Data Loader (Code)**  
   - *No edits required.* Functions:
     - `build_image_path_list(data_dir)`
     - `load_data(data_dir, size=32)`
   - Loads `train_images/train_labels` and `test_images/test_labels`.

5. **Assignment Instructions (Markdown)**  
   - Tells you to build/train a CNN using **TensorFlow/Keras** and to keep a **log of versions** and their test accuracy.

6. **Version Sections (Markdown)**
   - **Fill in text** for:
     - `#### Version 1` — describe your first model and results.
     - `#### Version 2` — describe changes and results.
   - Add more versions if you run more experiments.

7. **🚧 Coding TODO (Code): `# YOUR CODE HERE`**
   - **This is the main exercise.** Implement:
     - Preprocessing
     - Model build
     - Compile/train
     - Test evaluation
     - (Optional) Augmentation, plots, confusion matrix
   - Print final **test accuracy** so you can copy it into the Version sections.

---

## 🧗 Stretch Ideas (Optional)

- **Data augmentation:** random flips, rotations, small translations, brightness/contrast.
- **Regularization:** Dropout, weight decay (kernel_regularizer), early stopping.
- **Schedulers:** ReduceLROnPlateau, CosineDecay, OneCycle.
- **Deeper or more efficient blocks:** residual connections, depthwise separable convs.
- **Evaluation:** per-class accuracy, confusion matrix, top-k accuracy.

---

## 🔧 Requirements

- `tensorflow` / `keras`
- `opencv-python`
- `numpy`
- (Optional) `matplotlib` / `seaborn` for plots

---

## ✅ Deliverables Checklist

- [ ] Implemented the CNN in the `# YOUR CODE HERE` cell.  
- [ ] Printed final **test accuracy**.  
- [ ] Completed **Version 1** section with architecture, setup, and results.  
- [ ] Completed **Version 2** section, explaining changes and impact.  
- [ ] (Optional) Added plots/diagnostics and more versions.

