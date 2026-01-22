# dTOF Benchmarking Application

This is a Flask-based web application for processing and benchmarking dTOF SPAD based pipeline histograms.

## Features

-   Dataset and submission uploads (ZIP files)
-   Dataset content parsed and stored in a local SQLite database.
-   Leaderboard creation based on datasets
-   Asynchronous processing of submissions using Celery
-   Calculation of metrics: ARE, L1, L2, and curve fit error
-   Dynamic leaderboard view with color-coded metric visualization
-   Dataset exploration view with sample details and histogram visualization.

## Prerequisites

-   Python 3.x
-   Redis

## Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd dtof_benchmarking
    ```

2.  **Install Python dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Start the Redis server:**
    Make sure your Redis server is running on its default port (6379).
    ```bash
    redis-server
    ```

## Running the Application

1.  **Start the Celery worker:**
    Open a new terminal window, navigate to the project directory, and run:
    ```bash
    celery -A app.celery worker --loglevel=info
    ```

2.  **Run the Flask application:**
    In another terminal window, navigate to the project directory, and run:
    ```bash
    python app.py
    ```

3.  **Access the application:**
    Open your web browser and go to `http://127.0.0.1:5000`.

## Data Structure

The application uses folder-based naming conventions to automatically detect and process different types of data.

### Dataset ZIP Structure:

Your dataset ZIP file should have the following folder structure.

```
dataset_name.zip
├── hist/                      # Core: Histogram .npz files (bins/counts)
├── pick/                      # Core: Ground truth peak .txt files
├── config/                    # Core: Sample configuration .json files
├── wave_shape/                # Core: Signal shape name .txt (gaussian/square)
├── tags/                      # Core: Sample tags .txt (comma separated)
├── raw_depth/                 # Dynamic: 2D depth maps .npz (name_WxH.npz)
├── hist_reference/            # Dynamic: Additional histograms .npz
└── any_folder_name/           # Generic: .png/.jpg for images, .txt for scalars
```

### Submission ZIP Structure:

Submissions follow a similar dynamic structure.

```
submission.zip
├── metric_accuracy/           # Dynamic Metric: .txt files with scalar values
├── hist_filtered/             # Dynamic Histogram: .npz files (bins/counts)
├── raw_refl_map/              # Dynamic Map: 2D .npz for interactive heatmap
└── viz_output/                # Generic Image: .png/.jpg visualizations
```

### Supported Folder Prefixes:

| Prefix | Type | Usage |
| :--- | :--- | :--- |
| `metric_` | **Metric** | Scalar values for leaderboards and comparison charts. |
| `raw_` | **Depth/Map** | 2D array data enabling interactive heatmaps and hover values. |
| `hist_` | **Histogram** | Secondary histograms viewable in comparison. |
| *None* | **Generic** | Detected as **Image** (if contains .png/.jpg) or **Scalar** (if contains .txt). |

> [!NOTE]
> `raw_histogram` is also supported as an alias for `hist_` but `hist_` is preferred for new data.

### File Naming Conventions:
- Files must be named `<sample_name>.<ext>` (e.g., `sample1.txt`, `sample1.npz`).
- For `raw_` folders, use `<sample_name>_<width>x<height>.npz` to enable spatial visualization.

## Handling DLP Blocks (Code Uploads)

If your network or computer blocks the upload of Python code, use the included **Code Obfuscator** tools located in the `scripts/` directory.

### 1. Standalone Tools
-   **[`obfuscator.html`](scripts/obfuscator.html)**: Portable tool. Open in any browser to convert code into "safe" `.txt` files.
-   **[`obfuscator_gui.py`](scripts/obfuscator_gui.py)**: Tkinter-based GUI app for local obfuscation.

### 2. Built-in "DLP Safe Mode"
When creating or editing metrics in the web UI, check the **DLP Safe Mode** box. This will encode the code to Base64 *locally* in your browser before transmission.

### 3. Packaging the GUI as an Executable
To create a standalone app (`.exe` for Windows or `.app` for macOS):
1.  Install PyInstaller: `pip install pyinstaller`
2.  Run build: `pyinstaller --onefile --windowed --name "CodeObfuscator" scripts/obfuscator_gui.py`
3.  Find your app in the `dist/` directory.
