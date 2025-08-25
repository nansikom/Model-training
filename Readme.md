# Model Training With Faster R-CNN

This project trains an object detection model using **Faster R-CNN** with a ResNet-50 backbone on a custom dataset. The model leverages PyTorch and Torchvision for training and evaluation.

## 🚀 Features

- Uses **Faster R-CNN (ResNet-50 FPN)** from Torchvision
- Custom dataset support with bounding boxes
- Data loading with **PyTorch Dataset & DataLoader**
- Visualization of predictions with **Matplotlib**
- Image preprocessing with **OpenCV (cv2)**

## 📋 Requirements

All dependencies are listed in `requirements.txt`.

Install them using:
```bash
pip install -r requirements.txt
```

## 🛠️ Installation

### Clone The Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Create A Virtual Environment (Recommended)
```bash
python -m venv venv
```

**On Windows:**
```bash
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## 📂 Project Structure

```
project-root/
├── data/                 # Dataset folder: images, annotations
├── models/               # Saved / trained models
├── notebooks/            # Jupyter notebooks for exploration & visualization
├── src/                  # Source code
│   ├── dataset.py        # Custom Dataset class
│   ├── train.py          # Training script
│   ├── test.py           # Testing / inference script
│   └── utils.py          # Helper functions
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## 🔧 Usage

### Training The Model

Run the main training script:
```bash
python train.py
```

### Alternative Training (Google Colab)

If using `train2.py`, convert it to a Jupyter notebook and run it in Google Colab:
```bash
python train2.py
```

## 🧪 Technologies Used

- **Python 3.10+**
- **PyTorch**
- **Torchvision**
- **OpenCV**
- **Matplotlib**
- **Pandas & NumPy**

## 📊 Results

The trained model produces object detection results like this:

![Sample Wheat Image](images/readme.png)

## 📄 License

This project is licensed under the MIT License.
