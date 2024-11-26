### Evaluating the necessity of colour normalisation in Deep Learning (DL)-based histopathological image classification
The project evaluates multiple colour normalisation (CN) techniques and compares them against data augmentation (DA) to determine their impact on the performance of a DenseNet201 model, using the publicly available **BreakHis** dataset for breast cancer histopathology images.

#### About the Project
Medical AI workflows often rely on pre-processing steps like CN to standardise histopathological images. However, this study challenges the assumption that CN is always necessary, showing that robust DA alone can outperform traditional CN techniques. This finding has significant implications for simplifying AI workflows and making them more accessible and scalable in real-world healthcare applications.

#### Key Features
 - Implementation of four CN techniques:
**Channel-Based Normalisation (CBN)**
**Color Deconvolution (CD)**
**CBN + CLAHE**
**CD + CLAHE**

 - DA techniques including rotation, flipping, zooming, and more.
 - DenseNet201 model for binary classification of breast cancer histopathology images.

 - Evaluation metrics:
**Accuracy**, **Sensitivity**, **Specificity**, **ROC-AUC**, and **F1 Score**.

#### Dataset
The project uses the BreakHis dataset, a large collection of histopathological images of benign and malignant breast tumors at different magnifications. 

#### Getting Started
 - Prerequisites
Install the required Python libraries using pip:

```bash

pip install tensorflow numpy pandas scikit-learn opencv-python matplotlib
```
How to Run the Notebook

Clone the repository:
```bash

git clone https://github.com/drnsmith/ColourNorm-Histopathology-DeepLearning.git
```
Download the BreakHis dataset from the official site and place it in the `data/breakhis/` directory.

Open the Jupyter Notebook:
```bash

jupyter notebook colornorm.ipynb
```
Run the notebook cells step-by-step to:
 - Load and preprocess the dataset.
 - Apply CN and DA.
 - Train and evaluate the DenseNet201 model.

#### Results
DA alone achieved the highest performance across all metrics, outperforming traditional CN techniques. Key metrics:
 - Accuracy: 91.8%
 - Sensitivity: 90.2%
 - Specificity: 92.5%

The findings suggest that modern DL architectures, coupled with robust DA, can effectively handle staining variability without the need for CN.

### Contributing
Contributions are welcome! If you have ideas for improving the project or extending its scope, feel free to:

#### Fork the repository
 - Create a pull request
 - Open an issue for discussion

### License
This project is licensed under the MIT License. See the LICENSE file for details.

### Acknowledgments
Special thanks to the creators of the BreakHis dataset and the researchers whose methodologies inspired this project.
