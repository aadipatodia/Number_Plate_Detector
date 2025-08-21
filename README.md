# Indian Automatic Number Plate Recognition (ANPR)

This project develops a robust and accurate Automatic Number Plate Recognition (ANPR) system for Indian vehicles. The system uses a deep learning model to first detect the number plate in an image and then applies an advanced Optical Character Recognition (OCR) engine to read the text.

---

## 🚀 **Features**

* **Number Plate Detection:** Uses a custom-trained **YOLOv8** model to accurately locate number plates in diverse images.
* **High Accuracy:** The model is trained on a consolidated dataset from **Kaggle** and **Roboflow**, ensuring high performance on real-world images.
* **Optimized OCR:** Employs **EasyOCR**, a state-of-the-art OCR library, to extract text from the detected number plates, outperforming traditional OCR methods like `pytesseract`.
* **Modular Pipeline:** The system is built in a modular fashion, allowing for easy updates and improvements to both the detection and recognition components.

---

## 🧠 **Methodology**

1.  **Data Consolidation:** We combined data from two primary sources: a sample dataset from **Kaggle** and a larger, pre-annotated dataset from **Roboflow**. This was crucial to overcome the limitations of a small dataset and prevent model overfitting.
2.  **Data Cleaning:** The datasets had different class labels and were imbalanced (more images than labels). We implemented a Python script to correct the class IDs and delete all images that lacked a corresponding annotation file. This ensured a perfectly balanced dataset for training.
3.  **Model Training:** A pre-trained YOLOv8n model was fine-tuned on the cleaned, consolidated dataset for 50 epochs.
4.  **Inference & Recognition:** The final trained model is used to predict the location of a number plate. The cropped image of the number plate is then passed to **EasyOCR** for highly accurate text extraction.

---

## 📦 **How to Run the Project**

To run the project, you need to set up your environment with the necessary libraries. After that, you can use the final inference script to detect and recognize number plates in your own images.

### 1. **Setup**

Run the following commands in your Colab notebook to install all required dependencies:

!pip install ultralytics
!pip install opencv-python
!pip install easyocr


### 2. **Final Inference Script**

Replace `/content/path_to_your_new_image.jpg` with the path to your image file. The script will use your custom-trained model to detect the number plate and then use EasyOCR to extract the text.

from ultralytics import YOLO
import cv2
import easyocr
import numpy as np

Load your custom trained model
model = YOLO('runs/detect/train/weights/best.pt')

Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

Define and load the test image
test_image_path = '/content/path_to_your_new_image.jpg'
image = cv2.imread(test_image_path)

if image is None:
print(f"Error: Unable to load image at {test_image_path}")
else:
# Run a prediction with your model
results = model(image)

if results[0].boxes:
    # Get bounding box coordinates
    box = results[0].boxes[0].xywh[0].cpu().numpy()
    x, y, box_w, box_h = box
    x1, y1 = int(x - box_w/2), int(y - box_h/2)
    x2, y2 = int(x + box_w/2), int(y + box_h/2)

    # Crop the number plate region
    number_plate_roi = image[y1:y2, x1:x2]

    # Use EasyOCR to recognize the text from the cropped image
    ocr_results = reader.readtext(number_plate_roi)

    if ocr_results:
        detected_text = ""
        for (bbox, text, prob) in ocr_results:
            detected_text += text
        
        cleaned_text = ''.join(e for e in detected_text if e.isalnum())
        
        if cleaned_text:
            print("--- Model Detected a Number Plate ---")
            print(f"Extracted Car Number: {cleaned_text}")
        else:
            print("Could not extract a readable number plate.")
    else:
        print("Could not extract a readable number plate.")

else:
    print("No number plate detected in the image.")
