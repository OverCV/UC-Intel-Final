# Streamlit App Layout Design
## Malware Classification - Notebook-Style Workflow

---

## App Structure Overview

**Navigation:** Sidebar with page links (max 7 pages)
**Philosophy:** Each page is self-contained, linear workflow, scroll down to complete each step
**State:** All selections persist in `st.session_state` as you navigate
**Design:** Max 7 sections per page, max 7 tabs per section

---

# Page 1: <- Home & Setup

## Section 1: Project Overview
```
[st.title] Malware Classification with Deep Learning
[st.markdown] Brief description of the project

[st.expander "What does this app do?"]
    - Upload/select malware datasets
    - Configure CNN architectures
    - Train models with different hyperparameters
    - Evaluate and visualize results
```

## Section 2: Quick Start
```
[st.container]
    [st.metric] col1: Datasets Available - 3
    [st.metric] col2: Pre-trained Models - 5
    [st.metric] col3: GPU Status - Available/Not Available

[st.button] "Start New Training Session"
    - Creates new session ID, clears old state, navigates to Dataset page
```

## Section 3: Load Previous Session
```
[st.selectbox] "Load Previous Session"
    Options: [List of saved session IDs with dates]

[st.button] "Load Session"
    - Loads session state, shows summary, allows resume
```

---

# Page 2: Dataset Configuration

## Section 1: Dataset Selection
```
[st.header] Dataset Selection

[st.multiselect] "Choose Dataset(s)"
    Options: [MalImg, Malevis, Blended]
    Default: MalImg
    - Stores in st.session_state.datasets

[st.info] Shows: Total samples, Number of classes for selected datasets
```

## Section 2: Data Split
```
[st.header] Train/Validation/Test Split

[st.columns 3]
    col1: [st.slider] "Train %" - 70% default
    col2: [st.slider] "Val %"   - 15% default
    col3: [st.slider] "Test %"  - 15% default
    - Show sum value, show warning if not sum to 100%

[st.checkbox] "Stratified Split" - True default
[st.number_input] "Random Seed" - 42 default
```

## Section 3: Sample Visualization
```
[st.header] Dataset Preview

[st.selectbox] "Filter by Family" - All families from selected datasets

[st.columns 5] Image grid showing 5 samples per row
    Each image:
        - st.image(sample)
        - st.caption(f"Family: {family_name}")

[st.number_input] "Page" - Navigate through dataset
```

## Section 4: Class Distribution
```
[st.header] Class Distribution

[st.plotly_chart] Bar chart
    X-axis: Malware families
    Y-axis: Number of samples
    Color: By dataset if multiple selected
    Hover: Show exact count + percentage
```

## Section 5: Preprocessing Configuration
```
[st.header] Image Preprocessing

[st.selectbox] "Target Size"
    Options: [224x224, 256x256, 299x299, Custom]
    - If Custom, show two number_inputs for width/height

[st.radio] "Normalization"
    Options: [[0,1], [-1,1], Z-score (dataset mean/std)]

[st.radio] "Color Mode"
    Options: [Grayscale, RGB]

[st.columns 2]
    col1: [st.image] "Before Preprocessing"
    col2: [st.image] "After Preprocessing"
    - Shows live preview of selected sample
```

## Section 6: Data Augmentation
```
[st.header] Data Augmentation

[st.radio] "Augmentation Preset"
    Options: [None, Light, Moderate, Heavy, Custom]

[st.expander "Configure Custom Augmentation"] (only if Custom selected)
    [st.checkbox] "Rotation" + [st.slider] "Degrees -" - 0-45-
    [st.checkbox] "Horizontal Flip" + [st.slider] "Probability" - 0.0-1.0
    [st.checkbox] "Vertical Flip" + [st.slider] "Probability" - 0.0-1.0
    [st.checkbox] "Brightness" + [st.slider] "Range -" - 0.0-0.5
    [st.checkbox] "Contrast" + [st.slider] "Range -" - 0.0-0.5
    [st.checkbox] "Gaussian Noise" + [st.slider] "Sigma" - 0.0-0.1

[st.button] "Preview Augmentation"
    - Shows 5 columns with augmented versions of same image
```

## Section 7: Confirm Dataset Configuration
```
[st.divider]

[st.success] "Dataset Configuration Complete"

[st.json] Show current dataset config summary
    {
        "datasets": ["MalImg"],
        "total_samples": 9339,
        "num_classes": 25,
        "split": {"train": 70, "val": 15, "test": 15},
        "preprocessing": {...},
        "augmentation": "Light"
    }

[st.button] "Next: Model Configuration -"
    - Saves config to st.session_state.dataset_config
    - Navigates to Model Builder page
```

---

# Page 3: =' Model Configuration

## Section 1: Model Type Selection
```
[st.header] Model Architecture

[st.radio] "Model Type"
    Options: [Custom CNN, Transfer Learning]
    - Changes what sections appear below
```

---

### IF Custom CNN Selected:

## Section 2: Convolutional Layers
```
[st.header] Convolutional Layers

For each block (dynamically added):
    [st.expander f"Convolutional Block {i}"] (expanded by default for first block)
        [st.columns 2]
            col1:
                [st.slider] "Number of Filters" - 32, 64, 128, 256, 512
                [st.selectbox] "Kernel Size" - 3x3, 5x5, 7x7
                [st.selectbox] "Activation" - ReLU, Mish, Swish, GELU, Leaky ReLU
                    (with tooltip showing recommendation from activation.md)
            col2:
                [st.checkbox] "Add Second Conv Layer" - False default
                [st.checkbox] "MaxPooling 2x2" - True default
                [st.slider] "Dropout" - 0.0-0.5, default 0.25

        [st.button] "Remove Block" (if more than 1 block exists)

[st.button] "+ Add Convolutional Block" (max 7 blocks)
```

## Section 3: Fully Connected Layers
```
[st.header] Dense Layers

For each dense layer (dynamically added):
    [st.expander f"Dense Layer {i}"] (expanded by default)
        [st.slider] "Units" - 64, 128, 256, 512, 1024
        [st.selectbox] "Activation" - ReLU, Mish, Swish, GELU
        [st.slider] "Dropout" - 0.0-0.7, default 0.5

        [st.button] "Remove Layer" (if more than 1 layer)

[st.button] "+ Add Dense Layer" (max 5 layers)
```

## Section 4: Output Layer (Auto)
```
[st.header] Output Layer

[st.info] Auto-configured based on dataset
    Units: {num_classes} (from dataset config)
    Activation: Softmax
```

---

### IF Transfer Learning Selected:

## Section 2: Pre-trained Model Selection
```
[st.header] Pre-trained Model

[st.radio] "Select Base Model"
    Options: [VGG16, VGG19, ResNet50, ResNet101, InceptionV3, EfficientNetB0]
    - Show info card with params, input size, description

[st.radio] "Weights"
    Options: [ImageNet (recommended), Random (train from scratch)]
```

## Section 3: Fine-tuning Strategy
```
[st.header] Fine-tuning Configuration

[st.radio] "Strategy"
    Options:
        - Feature Extraction (freeze all base layers)
        - Partial Fine-tuning (trainable last N layers)
        - Full Fine-tuning (all layers trainable, low LR)

[st.slider] "Layers to Unfreeze" - 0 to {total_layers}
    (only shown if Partial Fine-tuning selected)
```

## Section 4: Custom Top Layers
```
[st.header] Classification Head

[st.checkbox] "Global Average Pooling" - True default
[st.checkbox] "Add Dense Layer"
    - If checked, show [st.slider] "Units" - 256, 512, 1024
[st.slider] "Dropout" - 0.0-0.7, default 0.5
```

---

## Section 5: Architecture Summary (Both CNN & Transfer Learning)
```
[st.header] Model Architecture Summary

[st.text] Text representation of model architecture
    Input (224, 224, 1)
          -
    Conv2D (32 filters, 3x3, ReLU)
          -
    MaxPool2D (2x2)
          -
    ...
          -
    Dense (25, Softmax)

[st.columns 3]
    col1: [st.metric] "Total Parameters" - 2,456,789
    col2: [st.metric] "Trainable Parameters" - 2,456,789
    col3: [st.metric] "Estimated Memory" - ~38 MB
```

## Section 6: Save & Export
```
[st.expander "Advanced: Export Model Code"]
    [st.code] Python code to recreate this architecture
    [st.download_button] "Download model.py"
```

## Section 7: Continue to Training
```
[st.divider]

[st.success] "Model Configuration Complete"

[st.button] "Next: Training Configuration -"
    - Saves to st.session_state.model_config
    - Navigates to Training page
```

---

# Page 4: -Ú Training Configuration

## Section 1: Optimizer Settings
```
[st.header] Optimizer Configuration

[st.selectbox] "Optimizer"
    Options: [Adam, AdamW, SGD with Momentum, RMSprop]

[st.slider] "Learning Rate" - log scale, 0.0001 to 0.01, default 0.001
    (show in scientific notation)

[st.expander "Advanced Optimizer Parameters"]
    (Content changes based on optimizer selected)
    If Adam/AdamW:
        [st.slider] "Beta 1" - 0.9 default
        [st.slider] "Beta 2" - 0.999 default
        [st.number_input] "Epsilon" - 1e-7 default
    If SGD:
        [st.slider] "Momentum" - 0.9 default
        [st.checkbox] "Nesterov" - True default
```

## Section 2: Learning Rate Scheduler
```
[st.header] Learning Rate Scheduling

[st.radio] "Strategy"
    Options:
        - Constant (no scheduling)
        - ReduceLROnPlateau
        - Cosine Annealing
        - Step Decay
        - Exponential Decay

[st.expander "Scheduler Parameters"] (content changes per strategy)
    If ReduceLROnPlateau:
        [st.slider] "Reduction Factor" - 0.1-0.9, default 0.5
        [st.slider] "Patience" - 3-20 epochs, default 5
        [st.number_input] "Min LR" - 1e-7 default
    If Cosine Annealing:
        [st.slider] "T_max" - 10-100, default 50
        [st.number_input] "Eta_min" - 0 default
```

## Section 3: Training Parameters
```
[st.header] Training Settings

[st.slider] "Max Epochs" - 10-200, default 100
[st.selectbox] "Batch Size" - 16, 32, 64, 128 (default 32)
[st.checkbox] "Shuffle Training Data" - True default
```

## Section 4: Regularization
```
[st.header] Regularization

[st.checkbox] "L2 Weight Decay"
    - If checked: [st.slider] "Lambda" - 0.0001-0.01, default 0.0001

[st.info] "Dropout configured in Model Architecture"
    Dropout rates: Conv layers {x}%, Dense layers {y}%
    [st.button] "- Go back to modify"
```

## Section 5: Class Imbalance Handling
```
[st.header] Class Imbalance Strategy

[st.radio] "Method"
    Options:
        - Auto Class Weights (recommended)
        - Focal Loss
        - No Adjustment

[st.expander "Class Distribution"]
    [st.plotly_chart] Bar chart showing class imbalance
    [st.dataframe] Table with class counts and suggested weights
```

## Section 6: Callbacks & Early Stopping
```
[st.header] Training Callbacks

[st.checkbox] "Early Stopping" - True default
    [st.slider] "Patience" - 5-30 epochs, default 10
    [st.number_input] "Min Delta" - 0.0001 default

[st.checkbox] "Model Checkpointing" - True default
    [st.radio] "Save Best By" - [Val Loss, Val Accuracy]

[st.checkbox] "TensorBoard Logging" - False default
```

## Section 7: Experiment Metadata & Start Training
```
[st.header] Experiment Details

[st.text_input] "Experiment Name"
    - Auto-generated: {activation}_{model_type}_{dataset}_{timestamp}
    - User can edit

[st.text_area] "Description (optional)"
    Placeholder: "Testing Mish activation on baseline CNN with MalImg dataset"

[st.multiselect] "Tags"
    Options: [mish, relu, swish, baseline, transfer-learning, malimg, experiment]
    - User can add custom tags

[st.divider]

[st.form "start_training_form"]
    [st.info] "Training Configuration Summary"
        Dataset: MalImg (9,339 samples, 25 classes)
        Model: Custom CNN (2.4M params)
        Epochs: 100 | Batch: 32 | LR: 0.001
        Optimizer: Adam | Scheduler: ReduceLROnPlateau

    [st.warning] "- Training will take approximately 3-4 hours on GPU"

    [st.form_submit_button] "- START TRAINING" (large, primary button)
        - Saves all config to st.session_state.training_config
        - Initializes training state
        - Shows training monitor below
```

---

# Page 5: =- Training Monitor

## Section 1: Training Status
```
[st.header] Training in Progress

[st.status] "Training..." (expandable, updated via @st.fragment)
    Current Status: Epoch 15/100
    Estimated Time Remaining: 2h 34m
    GPU Memory: 4.2 / 8.0 GB
```

## Section 2: Progress Bar
```
[st.progress] Epoch progress - 15/100 (15%)
[st.text] "Epoch 15/100 - Batch 187/256"
```

## Section 3: Current Metrics
```
[st.columns 4]
    col1: [st.metric] "Train Loss" - 0.3421 (- 0.05 from last epoch)
    col2: [st.metric] "Train Acc" - 89.34% (- 2.1%)
    col3: [st.metric] "Val Loss" - 0.4156 (- 0.03)
    col4: [st.metric] "Val Acc" - 86.72% (- 1.8%)
```

## Section 4: Live Training Curves
```
[st.header] Training History

[@st.fragment run_every="2s"] Auto-updating charts

[st.plotly_chart] Training & Validation Loss
    X: Epoch
    Y: Loss
    Lines: Train (blue), Val (red)

[st.plotly_chart] Training & Validation Accuracy
    X: Epoch
    Y: Accuracy (%)
    Lines: Train (blue), Val (red)
```

## Section 5: Learning Rate Schedule
```
[st.plotly_chart] Learning Rate over Time
    X: Epoch
    Y: Learning Rate (log scale)
    Annotations: When LR was reduced
```

## Section 6: Training Logs
```
[st.expander "Training Logs" collapsed]
    [st.text_area] (read-only, auto-scroll to bottom)
        [15:23:45] Epoch 15/100
        [15:23:47] 187/256 - loss: 0.3421 - acc: 0.8934 - val_loss: 0.4156 - val_acc: 0.8672
        [15:23:50] EarlyStopping: val_loss did not improve
        [15:23:51] ReduceLROnPlateau: Reducing LR to 0.0005
        ...
```

## Section 7: Training Controls
```
[st.columns 3]
    col1: [st.button] "- Pause Training"
    col2: [st.button] "- Stop Training"
    col3: [st.download_button] "=- Save Checkpoint"

[st.info] "Training will continue in background. You can navigate to other pages."
```

---

# Page 6: =- Results & Evaluation

*Note: This page appears automatically when training completes, or user can select completed experiment*

## Section 1: Experiment Summary
```
[st.header] Experiment Results

[st.success] " Training Complete!"

[st.columns 2]
    col1:
        [st.metric] "Experiment ID" - exp_001_mish_cnn_malimg
        [st.metric] "Duration" - 3h 42m
        [st.metric] "Completed At" - 2025-01-20 15:45:23
    col2:
        [st.metric] "Best Epoch" - 47/100
        [st.metric] "Early Stopped At" - Epoch 57
        [st.metric] "Final LR" - 0.000125
```

## Section 2: Final Test Metrics
```
[st.header] Test Set Performance

[st.columns 4]
    col1: [st.metric] "Accuracy" - 94.25%
    col2: [st.metric] "Precision (macro)" - 93.87%
    col3: [st.metric] "Recall (macro)" - 93.92%
    col4: [st.metric] "F1-Score (macro)" - 93.89%
```

## Section 3: Training History
```
[st.header] Training Curves

[st.tabs] ["Loss", "Accuracy", "Learning Rate"]
    Tab 1: [st.plotly_chart] Loss curves (final, non-updating)
    Tab 2: [st.plotly_chart] Accuracy curves
    Tab 3: [st.plotly_chart] LR schedule
```

## Section 4: Confusion Matrix
```
[st.header] Confusion Matrix

[st.columns 2]
    col1: [st.radio] "Normalize" - [None, True (by row), Pred (by col)]
    col2: [st.selectbox] "Colormap" - [Blues, Viridis, RdYlGn]

[st.plotly_chart] Interactive confusion matrix heatmap
    - Hover shows: True={X}, Pred={Y}, Count={Z}, %={W}
    - Click to highlight row/column
    - Zoomable

[st.expander "Most Confused Class Pairs"]
    [st.dataframe] Table showing:
        True Label | Predicted As | Count | Percentage
        VB         | Alureon      | 23    | 12.3%
        Rbot       | Bifrose      | 18    | 9.8%
        ...
```

## Section 5: Per-Class Performance
```
[st.header] Classification Report

[st.selectbox] "Sort By" - [Class Name, F1-Score, Precision, Recall, Support]

[st.dataframe] Interactive classification report
    Columns: Class | Precision | Recall | F1-Score | Support
    Last rows: Macro Avg, Weighted Avg
    Color-coded by performance (green=high, red=low)

[st.plotly_chart] Bar chart of F1-Scores by class
    X: Class names
    Y: F1-Score
    Color gradient by score
```

## Section 6: ROC Curves
```
[st.header] ROC Analysis (One-vs-Rest)

[st.multiselect] "Select Classes to Display"
    - [All classes + "Select All" + "Top 5" + "Bottom 5"]

[st.plotly_chart] ROC curves
    - One line per selected class
    - Diagonal reference line (random classifier)
    - Legend with AUC scores
    - Toggle visibility by clicking legend

[st.dataframe] AUC Scores Summary
    Class | AUC Score | Interpretation
    Alureon | 0.987 | Excellent
    VB      | 0.972 | Excellent
    ...
    Macro Avg | 0.964 | Excellent
```

## Section 7: Export Results
```
[st.header] Export & Save

[st.columns 3]
    col1: [st.download_button] "=- Download PDF Report"
    col2: [st.download_button] "=- Download Metrics (CSV)"
    col3: [st.download_button] "=- Download Model (.h5)"

[st.download_button] "- Download Full Config (JSON)"
    - Includes all hyperparameters, dataset config, model architecture
```

---

# Page 7: <- Model Interpretability

## Section 1: Grad-CAM Visualization
```
[st.header] Grad-CAM: What the Model Sees

[st.file_uploader] "Upload Malware Sample (or pick from test set)"
    - OR [st.selectbox] "Select from test set"

[st.columns 3]
    col1:
        [st.image] Original Image
        [st.caption] True: {family}, Pred: {family}
    col2:
        [st.image] Grad-CAM Heatmap
        [st.caption] Focus regions highlighted
    col3:
        [st.image] Overlay
        [st.caption] Combined view

[st.slider] "Heatmap Opacity" - 0.0-1.0, default 0.4
[st.selectbox] "Target Layer" - [Last Conv, Conv Block 3, Conv Block 2]

[st.expander "Prediction Confidence"]
    [st.dataframe] Top 5 predictions with probabilities
        Rank | Class | Probability | Bar
        1    | Alureon | 98.7% | -
        2    | VB      | 1.2%  | -
        ...
```

## Section 2: t-SNE Embeddings
```
[st.header] Feature Space Visualization

[st.radio] "Dimensionality Reduction Method"
    Options: [t-SNE, UMAP, PCA]

[st.columns 2]
    col1: [st.slider] "Perplexity" - 5-50, default 30 (if t-SNE)
    col2: [st.slider] "Samples to Plot" - 100-all, default 1000

[st.radio] "Color Points By"
    Options: [True Family, Predicted Family, Correct/Incorrect]

[st.plotly_chart] Interactive 2D scatter plot
    - Each point = one sample
    - Hover: Sample ID, True class, Pred class, Confidence
    - Click: Show sample image in sidebar
    - Lasso/box select for analysis

[st.columns 2]
    col1: [st.metric] "Silhouette Score" - 0.847 (cluster quality)
    col2: [st.metric] "Davies-Bouldin Index" - 0.423
```

## Section 3: Activation Maps
```
[st.header] Convolutional Filter Activations

[st.selectbox] "Select Sample"
[st.selectbox] "Select Layer" - [Conv2D_1, Conv2D_2, Conv2D_3, ...]

[st.info] "Layer info: {filters} filters, output shape {shape}"

[st.columns 6] Feature maps grid (showing first 18 filters)
    Each cell: [st.image] of activation map + filter index

[st.button] "Show All {N} Filters" - Expands to show all
```

## Section 4: Filter Weights Visualization
```
[st.header] Learned Convolutional Filters

[st.selectbox] "Select Convolutional Layer"

[st.columns 8] Grid of 3x3 kernel weights
    Each cell: Heatmap visualization of kernel

[st.expander "Filter Statistics"]
    [st.dataframe] Per-filter stats
        Filter | Mean | Std | Min | Max | L2 Norm
```

## Section 5: LIME Explanations
```
[st.header] LIME: Local Interpretable Explanations

[st.selectbox] "Select Sample"

[st.columns 3]
    col1: [st.image] Original
    col2: [st.image] Superpixels (segmentation)
    col3: [st.image] Explanation (green=support, red=against)

[st.slider] "Number of Superpixels" - 50-500, default 200
[st.slider] "Top Features to Show" - 5-20, default 10

[st.dataframe] Top Contributing Segments
    Segment ID | Weight | Effect
    #142 | +0.347 | Strongly supports prediction
    #089 | +0.289 | Supports prediction
    #201 | -0.145 | Against prediction

[st.button] "Recompute LIME" (if parameters changed)
```

## Section 6: Misclassification Analysis
```
[st.header] Misclassified Samples Analysis

[st.slider] "Number of samples to show" - 5-50, default 10

For each misclassified sample:
    [st.expander f"Sample {id}: True={X}, Pred={Y} (conf={Z}%)"]
        [st.columns 3]
            col1: [st.image] Sample
            col2: [st.image] Grad-CAM
            col3:
                [st.dataframe] Prediction confidences
                [st.text] Possible reason for misclassification

[st.selectbox] "Filter by": [All, Specific True Class, Specific Pred Class]
```

## Section 7: Model Architecture Review
```
[st.header] Architecture Summary

[st.expander "Full Model Architecture"]
    [st.text] Layer-by-layer breakdown
    [st.dataframe] Layer table
        Layer | Type | Output Shape | Params | Trainable

[st.columns 3]
    col1: [st.metric] "Total Params" - 2,456,789
    col2: [st.metric] "Model Size" - 38 MB
    col3: [st.metric] "Inference Time" - 12ms per sample

[st.download_button] "Export Architecture Diagram"
```

---

# Sidebar (Global - All Pages)

```
[st.logo] Project logo

[st.divider]

[st.sidebar.header] Navigation
[st.navigation] (automatic page links)
    - <- Home & Setup
    - =- Dataset Configuration
    - =' Model Configuration
    - - Training Configuration
    - =- Training Monitor
    - =- Results & Evaluation
    - <- Model Interpretability

[st.divider]

[st.sidebar.header] Current Session
[st.text] Session ID: {id}
[st.text] Status: {status}

[st.divider]

[st.sidebar.header] Quick Info
[st.metric] "GPU" - Available / Not Available
[st.metric] "Memory" - 4.2/8.0 GB

[st.divider]

[st.sidebar.header] Settings
[st.toggle] "Dark Mode"
[st.toggle] "Auto-save Progress"

[st.divider]

[st.sidebar.header] Resources
[st.link_button] "=- Activation Guide" - Links to activation.md
[st.link_button] "<- Architecture Doc" - Links to arch.md
[st.link_button] "S Help"
```

---

## Notes on Design Principles

1. **Max 7 Sections per page** - Keeps pages manageable, forces focus
2. **Max 7 tabs per section** - Prevents cognitive overload
3. **Linear workflow** - Each page builds on previous, scroll down to complete
4. **State persistence** - All selections saved in `st.session_state`
5. **Self-contained pages** - Each page does ONE thing well
6. **Progressive disclosure** - Use expanders for advanced options
7. **Immediate feedback** - Show previews, live updates where possible
8. **Clear CTAs** - Big obvious buttons to continue workflow

---

## Placeholders for Discussion

### =4 NEEDS DECISION:
- [ ] Should we allow multiple experiments running in parallel?
- [ ] Should training monitor be a separate page or embedded in Training Config?
- [ ] Do we need a "Compare Experiments" feature or keep it simple?
- [ ] How to handle resume from checkpoint? Automatic or manual?

### =- OPTIONAL FEATURES:
- [ ] Hyperparameter auto-tuning (Optuna integration)
- [ ] Model ensemble builder
- [ ] Dataset upload (custom datasets)
- [ ] Export to ONNX/TFLite for deployment

### =- CONFIRMED:
-  Notebook-style linear workflow
-  7 main pages max
-  Plotly for all charts
-  Session state for persistence
-  No complex "experiment tracking" system
