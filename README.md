---
typora-root-url: ./figures
---

# MLF: A Lightweight and Generic Feature Fusion Architecture

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE) [![Model Status](https://img.shields.io/badge/status-Research-orange)]() [![Paper](https://img.shields.io/badge/arXiv-MLF--YOLO-green)](https://github.com/request404/MLF)



## 📌 Introduction

YOLO-MLF introduces a **Multi-Level Fusion (MLF)** convolutional architecture to improve real-time object detection. It addresses the redundancy and inefficiency of existing cross-stage convolutional designs by enabling **cross-level feature fusion**, **lightweight partial convolution (PDC)**, and **progressive spatial scaling**.

> The architecture integrates seamlessly with the YOLO framework and is validated on the DOTA-v1.5 dataset, achieving **46.5% mAP@50** and **30.2% mAP@50-95**, outperforming mainstream YOLO models.
>
> | ![](/./assets/figure1.jpg) | ![](/./assets/figure1.jpg) |
> | --------------------------- | --------------------------- |
>
> 

## 🚀 Key Features

- ✅ **Cross-Level Feature Fusion:** Enhances receptive field diversity with minimal cost.

- ✅ **Partial Convolution (PDC):** Reduces parameter overhead while boosting representation.

- ✅ **Progressive Hierarchy:** Multi-scale feature abstraction to improve small object detection.

- ✅ **Plug-and-Play YOLO Backbone:** Drop-in replacement for standard YOLO backbones.

- ✅ **Superior Generalization:** Proven effectiveness across DOTA-v1.5, NEU-DET, PCB-MARET, and more.

  ![](/./assets/figure3.jpg)

## 🧪 Benchmark Results

### 🛰 DOTA-v1.5 Dataset

| Model          | Params (M) | FLOPs (G) | mAP@50 (%) | mAP@50-95 (%) |
| -------------- | ---------- | --------- | ---------- | ------------- |
| YOLOv11-N      | 2.6        | 6.3       | 38.4       | 23.9          |
| **YOLO-MLF-N** | 2.5        | 7.2       | **38.6**   | **23.9**      |
| YOLOv11-M      | 20.0       | 67.7      | 44.7       | 28.9          |
| **YOLO-MLF-M** | 19.1       | 80.7      | **45.2**   | **29.2**      |
| YOLOv11-X      | 56.8       | 194.5     | 46.0       | 30.2          |
| **YOLO-MLF-X** | 51.5       | 188.1     | **46.5**   | **30.2**      |

### 🌍 Cross-Domain Generalization (N-scale)

| Dataset          | mAP@50 (%) | mAP@50-95 (%) | Precision (%) | Recall (%) |
| ---------------- | ---------- | ------------- | ------------- | ---------- |
| African-Wildlife | 93.8       | **78.9**      | 92.2          | 88.3       |
| NEU-DET          | **73.1**   | **40.6**      | 71.0          | 66.4       |
| LIB-DET          | 95.6       | **62.1**      | **96.1**      | 92.9       |
| PCB-MARET        | 92.5       | **49.3**      | 93.9          | 88.7       |

## 🧰 Getting Started

```bash
# Clone the repository
git clone https://github.com/request404/MLF.git
cd MLF

# Create environment
conda create -n mlf_yolo python=3.10 -y
conda activate mlf_yolo

# Install dependencies
pip install -r requirements.txt
```

## 🏋️‍♂️ Training

```python
from ultralytics import YOLO

model = YOLO('mlf.yaml')  # Custom YAML with MLF backbone

results = model.train(
    data='your_dataset.yaml',
    imgsz=640,
    epochs=300,
    batch=64,
    device=0,
)
```

## 🧪 Evaluation

```python
metrics = model.val(data='DOTAv1.5.yaml', save_json=True)
```

## 🔍 Prediction

```python
results = model.predict(source='test.jpg', conf=0.25)
results[0].show()
```

## 📄 Citation

If you use this work, please cite:

```latex
@article{chen2025mlf,
  title={MLF: A Lightweight and Generic Feature Fusion Architecture},
  author={Kequan Chen, HaoYang, Qijiang Yang, Hui Jiang, Qilin Bi, Zhansi Jiang},
  year={2025}
}
```

## 🤝 Acknowledgements

This project builds on the open-source foundation of [Ultralytics YOLO](https://github.com/ultralytics/ultralytics). We thank the YOLO community for their continuous innovation.
