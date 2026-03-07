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
│   ├── main.py
│   └── utils.py
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

# Install PyTorch with CUDA support (GPU acceleration)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

cd my_project
python main.py
```

### macOS

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cd my_project
python main.py
```