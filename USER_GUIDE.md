# Bench-Hub User Guide

Welcome to Bench-Hub, a platform for benchmarking and comparing the performance of systems (like Depth-of-Time-of-Flight sensors) using custom metrics and visualizations.

## 1. Core Concepts

*   **Project**: A container for related benchmarking efforts (e.g., "dTOF" for dToF sensors, "Lidar" for Lidar projects).
*   **Leaderboard**: A specific benchmarking comparison board within a project. It defines the metrics, visualizations, and datasets used for comparison.
*   **Dataset (Ground Truth)**: The reference data against which submissions are evaluated. Usually contains "Ground Truth" (GT) values.
*   **Submission**: A set of results loaded into the system to be compared against the dataset.
*   **Metric**: A Python function that calculates a score (scalar) or a value (per-sample) by comparing submission data to ground truth data.
*   **Visualization**: A Python function that generates an image (e.g., a plot, a heatmap) to visually compare data.

## 2. Getting Started

### 2.1 Installation
Bench-Hub is a web application running on Flask.
1.  Clone the repository.
2.  Install dependencies: `pip install -r requirements.txt`.
3.  Run the server: `python app.py`.
4.  Access the app at `http://localhost:6060`.

### 2.2 Datasets (Ground Truth)
The **Dataset** serves as the "Ground Truth" for your benchmarks.

> **Note:** Datasets are **Global**. They are not tied to a specific project. Once uploaded, a dataset can be used by any leaderboard in any project. This allows you to reuse standard benchmarks across different efforts.

### 2.3 Supported Data Types
Bench-Hub automatically detects the type of data based on file extensions and folder names inside your ZIP.

| Type | File Extensions | Description |
| :--- | :--- | :--- |
| **Image** | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff` | Loaded as image paths. Available to "Image" visualizations. |
| **Scalar** | `.txt` | Files containing a single number. Parsed as `float`. |
| **JSON** | `.json` | Parsed as a python dictionary. |
| **Histogram** | `.npz` (in `hist_*` folder) | Numpy archive with `bins` and `counts` arrays. |
| **Depth** | `.npz` (in `raw_` folder) | Numpy archive named `sample_WxH.npz`. |

### 2.4 First Steps
1.  **Create a Project**: Navigate to the "Projects" page and create a new project.
    *   **Cloning**: You can also **Clone** an existing project (settings, leaderboards, metrics) to jumpstart a new one.
2.  **Create a Leaderboard**: Inside your project, create a Leaderboard.
    *   **Import Settings**: When creating a leaderboard, you can **Import Settings** from another existing leaderboard. This copies all metric definitions and mappings, saving you from manual setup.
3.  **Upload a Dataset**: Go to the "Datasets" list.
    *   *Tip*: Your ZIP should contain a structured folder hierarchy where each sample corresponds to a file or folder.

## 3. Workflow: Running a Benchmark

### Step 1: Define Metrics
Before uploading submissions, you need metrics to evaluate them.
1.  Go to the **Leaderboard** > **Metrics** tab.
2.  Click **Add New Metric**.
3.  Select a global metric from the library (or add a new one in the "Metric Management" page).
4.  **Map Arguments**:
    *   This is the most critical step. You map the arguments of your Python function (e.g., `def my_metric(pred, gt)`) to the data fields in your dataset and submissions.
    *   Example: Map `pred` to `Submission:prediction_file.txt` and `gt` to `Dataset:ground_truth_file.txt`.

### Step 2: Upload Submissions
1.  Go to the **Leaderboard** tab.
2.  Click **Upload Submission**.
3.  Upload a ZIP file containing your results.
    *   The structure of the submission ZIP must arguably match the Dataset ZIP (same sample names) so they can be aligned.

### Step 3: View Results
*   **Leaderboard Table**: Shows aggregated scores (e.g., Mean L1 Error) for each submission.
*   **Comparison View**: Click on "Compare" or select multiple submissions to view side-by-side per-sample comparisons.
*   **Visualizations**: Custom plots (if configured) will appear in the Comparison View.

## 4. Advanced Features

### Dynamic Metrics
You can write your own metrics in Python.
*   **Per-Sample**: Calculates a value for *each* sample (e.g., "Absolute Error").
*   **Aggregated**: Calculates a single value for the *entire* dataset (e.g., "Total Accuracy", "F1 Score").

### Custom Visualizations
You can create custom plots using Python (Matplotlib/PIL).
*   These are rendered as images in the Comparison View.
*   Useful for visual error analysis (e.g., "Error Heatmap", "Point Cloud Overlay").

## 5. Troubleshooting
*   **"Metric Mapping Reset"**: If your metric mappings disappear, check that you saved them correctly in the "Edit Leaderboard" > "Metrics" tab.
*   **Recalculation**: If you change a metric definition, you may need to trigger a recalculation for existing submissions.
