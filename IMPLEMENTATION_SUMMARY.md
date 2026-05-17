# Implementation Summary

## Project Goal
This project analyses the macroinvertebrate image data downloaded from Kaggle, which then generates exploratory data analysis outputs, trains a 
baseline classifier, and provides a menu-driven console interface for image predictions.

## System Design Overview
The project was designed as a modular Python application with clearly separated responsibilities across distinct service classes and modules. 
Rather than placing all logic in a single script, each stage of the workflow — indexing, analysis, preprocessing, classification, and deployment 
— was implemented as a dedicated class within its own module. A shared WorkflowService coordinates all stages and serves as the entry point for 
both the batch runner (main.py) and the console application (console_app.py), ensuring that no logic is duplicated across entry points.

The EDA findings from Stage 1 directly informed design decisions in Stage 2. After observing class imbalance across macroinvertebrate categories 
and inconsistencies in image dimensions, the preprocessing pipeline was designed to normalise all images to a fixed size of 128 × 128 pixels and 
convert them to grayscale before flattening into feature vectors. This ensured that the Random Forest classifier received consistent input 
regardless of the original image properties. The stratified train-test split used during model training further addressed class imbalance by 
ensuring proportional class representation across both sets.

All outputs from Stage 1 and Stage 2 are saved to disk under the outputs/ directory, making them available for review during the demonstration and 
for inclusion in the project report. The console application in Stage 3 loads the saved model and exposes all major workflow actions through a 
numbered menu.

## Class and Module Overview
| Class/Module      | Responsibility                                                                                                                                                                              |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DatasetIndexer    | Scans the raw data directory recursively, reads image metadata using OpenCV, and builds a Pandas DataFrame with one row per image containing its file path, label, width, height, and channel count. |
| ImageRecord       | A dataclass that stores the core metadata fields for one indexed image. Used during the indexing stage to enforce consistent structure.                                                     |
| EDAService        | Generates and saves EDA outputs including a class distribution bar chart, image width and height histograms, a sample image grid, and a summary statistics dictionary.                                                                                                                                             |
| ImagePreprocessor | Loads images in grayscale using OpenCV, resizes them to a fixed 128x128 pixel dimension, normalises pixel values to the range 0.0-1.0, and flattens the array into a 1D feature vector for model input.                                                                                                                                                     |
| ClassifierServoce | Prepares feature arrays from the indexed DataFrame, splits the data into training and test sets using stratified sampling, trains a Random Forest classifier, evaluates model performance, and saves the trained model artifact using joblib.                                                                                                                                          |
| WorkflowService   | Coordinates the shared pipeline. Instantiates and connects all service classes, caches the loaded DataFrame, and exposes show_summary(), generate_eda(), train_model(), predict_image(), and run_full_pipeline() as public actions.                                                                                                                                     |
| ConsoleApp        | Provides a menu-driven console interface with numbered options. Accepts user input in a loop, delegates each action to the WorkflowService, and exits cleanly when option 5 is selected.                                                                                                                                  |

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

## Key Features Implemented

### Stage 1 - Exploratory Data Analysis
<!-- TOC -->
* Recursive dataset indexing using DatasetIndexer, producing a structured Pandas DataFrame 
* Class distribution bar chart saved to outputs/eda/class_distribution.png 
* Image width and height histogram saved to outputs/eda/image_size_distribution.png 
* 3x3 sample image grid saved to outputs/eda/sample_grid.png 
* Summary statistics table including total images, total classes, mean width, and mean height
<!-- TOC -->

### Stage 2 - Predictive Analytics
<!-- TOC -->
* Grayscale image preprocessing with fixed resize to 128x128 pixels and pixel normalisation 
* Stratified 80/20 train-test split to preserve class proportions 
* Random Forest classifier trained with 200 estimators using parallel processing (n_jobs=-1)
* Evaluation outputs: accuracy score, full classification report, and confusion matrix heatmap 
* Trained model saved to outputs/models/macro_classifier.joblib using joblib
<!-- TOC -->

### Stage 3 - Console Deployment
<!-- TOC -->
* Menu-driven ConsoleApp with five numbered options: dataset summary, EDA generation, model training, image prediction, and exit 
* Shared WorkflowService used by both main.py and console_app.py to avoid logic duplication 
* Prediction loads the saved joblib model and outputs the predicted class to the console 
* Error handling for missing model file, unreadable image path, and invalid menu input
<!-- TOC -->

## Testing Summary
Testing was carried out manually using a structured test plan documented in MANUAL_TESTING.md.

| Scenario                  | Input                                                    | Expected Result                                               | Evidence             |
|---------------------------|----------------------------------------------------------|---------------------------------------------------------------|----------------------|
| Invalid image path        | User enters file path `wrong_file.jpg` for option 4.     | A readable error is printed rather than the program crashing. | `error_1.pdf`        |
| Invalid menu choice       | User enters an invalid menu option such as `9`.          | Print an invalid error message and redisplay's the menu.      | `error_2.pdf`        |
| Missing model file        | User attempts to run option 4 before training the model. | A friendly error message is shown.                            | `error_3.pdf`        |
| Successful end-to-end run | Run every menu option in order.                          | Provides the correct outputs for each menu option.            | `successful_run.pdf` |

## Evidence and Sample Outputs
The following outputs were generated during the project and are saved under the outputs/ directory. 
<!-- TOC -->
* outputs/eda/class_distribution.png — bar chart showing image count per macroinvertebrate class
* outputs/eda/image_size_distribution.png — histograms of image widths and heights across the dataset 
* outputs/models/classification_report.txt — per-class precision, recall, and F1 scores 
* outputs/models/confusion_matrix.png — heatmap of predicted versus actual class labels 
* outputs/models/macro_classifier.joblib - trained random forest model classifying macroinvertebrate species
* outputs/reports/dataset_index.csv - index of 294 labelled images (paths, dimensions, labels)
* Successful run of console application running with menu options visible - refer to `successful_run.pdf`
<!-- TOC -->

## Reused or Adapted Code Acknowledgement
The code provided in the `Assignment 3 Full Guidance and Coding Examples` was adapted and used to complete this project. Alongside this, the code 
used in the `console_app.py` was adapted from the use of GitHub Copilot.

## Work Division Summary
Work was divided across the three group members according to the recommended structure in the assignment guidance. All members reviewed the complete 
codebase and can explain any part of the system during the Week 13 demonstration.

| Member      | Contribution                                                               |
|-------------|----------------------------------------------------------------------------|
| Brinda      | Brinda completed stage 1.                                                  |
| Thisari     | Thisari completed stage 2.                                                 |
| Dinara      | Dinara completed the console interface.                                    |
| All members | Completed the testing, README, Implementation Summary, and Manual_Testing. |