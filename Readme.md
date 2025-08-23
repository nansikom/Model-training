**MODEL TRAINING WITH FASTER R-CNN**

This project trains an object detection model using **Faster R-CNN** with a ResNet-50 backbone on a custom dataset. The model leverages PyTorch and Torchvision for training and evaluation.

---

**FEATURES**
- Uses **Faster R-CNN (ResNet-50 FPN)** from Torchvision
- Custom dataset support with bounding boxes
- Data loading with **PyTorch Dataset & DataLoader**
- Visualization of predictions with **Matplotlib**
- Image preprocessing with **OpenCV (cv2)**

---

**REQUIREMENTS**
All dependencies are listed in `requirements.txt`.  

Install them using:
```bash
pip install -r requirements.txt
**INSTALLATION**

Clone the repository:

bash

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Create a virtual environment (recommended):

bash

python -m venv venv
# On Linux/Mac
source venv/bin/activate
# On Windows
venv\Scripts\activate
Install dependencies:

bash

pip install -r requirements.txt
HOW TO RUN

Training the Model

bash
python train.py
Testing the Model



python test.py
Note: Place your dataset inside a data/ folder with annotations.

**PROJECT STRUCTURE**


.
├── data/                # Dataset folder
├── train.py             # Training script
├── test.py              # Testing / inference script
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
TECHNOLOGIES USED

Python 3.8+

PyTorch

Torchvision

OpenCV

Matplotlib

Pandas & NumPy

**LICENSE**
This project is licensed under the MIT License.

