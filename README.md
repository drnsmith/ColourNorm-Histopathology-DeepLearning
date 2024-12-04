### Colour Normalisation in Deep Learning: Enhancing Histopathology Image Classification

##### Overview
This project investigates the impact of various **Colour Normalisation (CN)** techniques on the performance of deep learning models in histopathological image classification. By comparing these techniques against robust **Data Augmentation (DA)** strategies, the study aims to determine whether CN is essential for achieving high classification accuracy. The publicly available **BreakHis** dataset, comprising breast cancer histopathology images, serves as the basis for evaluation.

---

#### **Motivation**
In medical AI workflows, pre-processing steps like CN are commonly employed to standardise histopathological images, addressing staining variability and enhancing model performance. However, this study challenges the necessity of CN by exploring whether advanced DA techniques can achieve comparable or superior results, thereby simplifying the pre-processing pipeline and improving scalability in real-world healthcare applications.

---

## **Key Features**
- **Implementation of Four CN Techniques**:
  - *Channel-Based Normalisation (CBN)*
  - *Color Deconvolution (CD)*
  - *CBN combined with Contrast Limited Adaptive Histogram Equalization (CLAHE)*
  - *CD combined with CLAHE*

- **Data Augmentation Strategies**:
  - Rotation
  - Flipping
  - Zooming
  - Other transformations to enhance model robustness

- **Deep Learning Model**:
  - Utilisation of a pre-trained **DenseNet201** model for binary classification of breast cancer histopathology images

- **Evaluation Metrics**:
  - Accuracy
  - Sensitivity
  - Specificity
  - ROC-AUC
  - F1 Score

---

## **Dataset**
The project utilises the **BreakHis** dataset, a comprehensive collection of histopathological images of benign and malignant breast tumors at various magnifications. The dataset is publicly accessible and widely used for research in medical image analysis.

---

## **Getting Started**

### **Prerequisites**
- **Python Libraries**:
  - TensorFlow
  - NumPy
  - Pandas
  - scikit-learn
  - OpenCV
  - Matplotlib

Install the required libraries using pip:
```bash
pip install tensorflow numpy pandas scikit-learn opencv-python matplotlib
```

### **Installation**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/drnsmith/ColourNorm-Histopathology-DeepLearning.git
   cd ColourNorm-Histopathology-DeepLearning
   ```

2. **Download the BreakHis Dataset**:
   - Obtain the dataset from the official [BreakHis website](https://web.inf.ufpr.br/vri/breast-cancer-database) and place it in the `data/breakhis/` directory.

3. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook colornorm.ipynb
   ```

4. **Run the Notebook**:
   - Execute the cells sequentially to:
     - Load and preprocess the dataset
     - Apply CN and DA techniques
     - Train and evaluate the DenseNet201 model

---

## **Results**
The study reveals that robust DA alone achieves the highest performance across all evaluation metrics, surpassing traditional CN techniques. Key metrics include:
- **Accuracy**: 91.8%
- **Sensitivity**: 90.2%
- **Specificity**: 92.5%

These findings suggest that modern deep learning architectures, when combined with effective DA, can handle staining variability without the need for explicit CN, thereby streamlining AI workflows in histopathological image analysis.

---

## **Contributing**
Contributions are welcome! If you have ideas for improving the project or extending its scope, please:
1. Fork the repository
2. Create a new branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Acknowledgments**
Special thanks to the creators of the BreakHis dataset and the researchers whose methodologies inspired this project.

