## Project structure

Download and place the dataset folder `data-ROI-192-96` in the project root directory:

```text
Deep-Learning-Prostate-Cancer-Detection-in-MRI
│
├── data-ROI-192-96/
│
├── my_project/
│   ├── 1-uclH-data_ratio0.8/
│   │   ├── train.txt
│   │   ├── val.txt
│   │   └── test.txt
│   │
│   ├── data_loader.py
│   ├── model.py
│   ├── test.py
│   ├── train.py
│   ├── utils.py
│   └── visualise.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Quick start

### Windows (GPU)

```bash
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

# Uninstall CPU PyTorch package
pip uninstall torch -y

# Install PyTorch with CUDA support (GPU acceleration)
pip install torch --index-url https://download.pytorch.org/whl/cu128

cd my_project
python train.py

python test.py

python visualise.py
```

### macOS

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cd my_project
python train.py

python test.py

python visualise.py
```