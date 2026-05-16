# Macroinvertebrate Image Analysis System

## Project Goal
This project analyses the macroinvertebrate image data downloaded from Kaggle, which then generates exploratory data analysis outputs, trains a 
baseline classifier, and provides a menu-driven console interface for image predictions.

## Main Features
- dataset indexing
- class distribution analysis
- image size analysis
- baseline image classification
- deployed prediction interface

## Packages Used
| Packages      | Purpose                                                 |
|---------------|---------------------------------------------------------|
| pathlib       | clean, readable file and folder handling                |
| pandas        | store and analyse indexed image records                 |
| numpy         | numerical arrays and model feature preparation          |
| opencv-python | read, resize, and preprocess images                     |
| matplotlib    | EDA charts and evaluation visualisations                |
| seaborn       | styles distribution plots and confusion matrix heatmap  |
| scikit-learn  | train/test split, random forest classifier, and metrics |
| joblib        | save and load trained model artifacts                   |

## Installing Instructions
1. Install dependencies with `pip install -r requirements.txt`
2. Place the dataset of your choice inside `data/raw`

## How to Run Stage 1 and Stage 2 workflow, along with console menu application
1. Run `python -m src.main`
2. Run `python -m src.console_app` to use the console
3. Console menu options:
   1. Show dataset summary
   2. Generate EDA outputs
   3. Train baseline classifier
   4. Predict an image
   5. Export dataset index to CSV
   6. Show per-class image statistics
   7. Exit

## Folder Structure
```
macro_project/
|-- data/
│   |-- processed/
│   |-- raw/
|       |-- stream_macroinvertebrates/
|-- outputs/
|   |-- eda/
|   |-- models/
|   |-- reports/
|-- src/
|   |-- models/
|   |-- services/
|   |-- utils/
|   |-- config.py
|   |-- console_app.py
|   |-- main.py
|-- .gitignore
|-- error_1.pdf
|-- error_2.pdf
|-- error_3.pdf
|-- IMPLEMENTATION_SUMMARY.md
|-- MANUAL_TESTING.md
|-- README.md
|-- requirements.txt
|-- successful_run.pdf
