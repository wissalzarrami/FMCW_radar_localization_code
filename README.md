# FMCW Radar Localization Framework 🚀📡

<p align="center">
  <img src="https://img.shields.io/badge/Code-PyTorch-blue.svg" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  <img src="https://img.shields.io/badge/Data-Google--Drive-green.svg" />
  <img src="https://img.shields.io/badge/Models-Pretrained-red.svg" />
  <img src="https://img.shields.io/badge/Benchmark-Ready-success.svg" />
</p>

---

## 🌍 Project Overview

This repository provides a comprehensive framework for **FMCW Radar-based Human Localization** leveraging deep learning architectures and signal processing techniques.

The proposed framework integrates multiple approaches for localizing humans in complex environments using raw radar data:

- Deep Learning-based Angle Estimation (DNN)
- DeepMusic-Inspired Angle Estimation
- Distance Estimation using DNN & FFT
- Stratified Dataset Design for Robust Evaluation
- Multi-person Localization Scenarios
- Cross-Lab Evaluation in Distinct Indoor Environments

This repository is designed for researchers and practitioners aiming to benchmark, improve, or deploy radar-based localization models in real-world conditions.

---

## 🗂 Repository Structure

| Folder Name                           | Description                                                 |
|--------------------------------------|------------------------------------------------------------- |
| `SOTA_angle_DNN/`                   | DNN-based angle estimation module for SOTA comparison         |
| `SOTA_angle_deepmusic/`             | DeepMusic-based angle estimation module for SOTA comparison   |
| `SOTA_distance_DNN/`                | Distance estimation using DNN for SOTA comparison             |
| `SOTA_distance_FFT/`                | Classical FFT-based distance estimation for SOTA comparison   |
| `stratifv2_GTnew_3persons/`         | Tests involving 3 persons simultaneously.                     |
| `stratifv2_GTnew_Both_Labs/`        | Data combining experiments from both labs.                    |
| `stratifv2_GTnew_test_LAB1/`        | Experiments trained with data in LAB 2 and Tested on LAB1.    |
| `stratifv2_GTnew_test_LAB2/`        | Experiments  trained with data in LAB 1 and Tested on LAB1.   |
| `stratifv2_GTnew_test_ALL/`         | All testing scenarios combined (person1 and 2 persons cases). |
| `stratifv2_GTnew_test_Person1/`     | Individual person-based testing.                              |
| `stratifv2_GTnew_test_Person13/`    | Individual person-based testing.                              |
| `stratifv2_GTnew_test_Person13/`    | Individual person-based testing.                              |
| `stratifv2_GTnew_Tests_others/`     | Other testing configurations and edge cases.                  |

---

## 📦 Dataset Access

> 📁 The complete dataset used for training, validation, and evaluation is publicly available at:

[[![Google Drive Dataset](https://img.shields.io/badge/Download-Dataset-blue)](https://drive.google.com/drive/folders/1ZoYyZUfX8yUQjeRwkmLlIXjvNGeQQazC)](https://drive.google.com/drive/folders/1ZoYyZUfX8yUQjeRwkmLlIXjvNGeQQazC)

### Dataset Details:
- Raw FMCW Radar Signals (.cf32 files)
- Structured in folders with ground truth labels of:
  - Angle
  - Distance
  - Environment (LAB1, LAB2)
  - Number of Persons (1, 2, or 3)
  
> Each folder contains 18 files: 9 per antenna (complex radar data).

---

## 🧠 Pre-trained Models

> 💾 Download all pre-trained models from:

[[![Google Drive Models](https://img.shields.io/badge/Download-Models-red)](https://drive.google.com/drive/folders/YOUR_MODEL_LINK_HERE)](https://drive.google.com/drive/folders/1T1FwFO5CJGH0goVboAhtcyvKd4ZJi9cc?usp=sharing)

These models are provided for reproducibility, benchmarking, and quick evaluation on your data. You will find a document that describe every model for each experience, also the models used for SOTA comparison.

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/wissalzarrami/FMCW_radar_localization_code.git
cd FMCW_radar_localization_code

# Install required dependencies
pip install -r requirements.txt

# Run any model
cd stratifyv2_GTnew_test_ALL
python testPolarv2.py
