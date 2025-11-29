# Kaggle Training Setup Guide

This guide explains how to train your Unified SR Model on Kaggle using the **STARE**, **SpaceNet 2**, and **Tuberculosis** datasets.

## Step 1: Prepare Your Code
1.  **Code**: Ensure your project folder contains:
    *   `src/` (with `models` and `utils`)
    *   `train.py`

## Step 2: Create a Kaggle Notebook
1.  **Dataset**: Create a new dataset (e.g., `unified-sr-code`) and upload your code.
2.  **Notebook**: Create a new notebook and add:
    *   Your `unified-sr-code` dataset.
    *   **STARE Dataset** (search for "STARE").
    *   **SpaceNet 2 Paris** (search for "SpaceNet 2 Paris").
    *   **Tuberculosis Chest X-rays** (search for "Shenzhen").

## Step 3: Run Training
1.  **Copy Code**: Copy the content of `kaggle_train.ipynb` into the notebook cells.
2.  **Verify Paths**:
    *   The notebook assumes the datasets are mounted at:
        *   `/kaggle/input/stare-dataset`
        *   `/kaggle/input/spacenet-2-paris-buildings`
        *   `/kaggle/input/tuberculosis-chest-xrays-shenzhen`
    *   **Check the "Input" section** in the right sidebar. If the names are different, update the paths in the notebook.
3.  **Run All**:
    *   Click "Run All".

## Step 4: Download Model
*   After training, the notebook will zip the `checkpoints` folder into `checkpoints.zip`.
*   The last cell will display a download link for `checkpoints.zip`.
*   Download and extract it into your local project folder (e.g., `e:\Sem VIII\Thesis_og\unified_sr_project`).

## Retraining with New Code
If you have updated the code (e.g., for auto-detection, perceptual loss, optimization), follow these steps to retrain:
1.  **Update Code Dataset**:
    *   Go to your `unified-sr-code` dataset on Kaggle.
    *   Click "New Version".
    *   Upload the updated `src` folder (now includes `loss.py` and updated `heads.py`) and `train.py`.
    *   Click "Create".
2.  **Restart Notebook**:
    *   Open your training notebook.
    *   In the right sidebar, check if the `unified-sr-code` dataset has updated.
    *   Click "Run All" again.
3.  **Inference with Self-Ensemble**:
    *   When running inference locally, you can now use the `--self_ensemble` flag for better quality:
        ```bash
        python inference.py --model_path checkpoints/model_final.pth --input_path ... --self_ensemble
        ```
