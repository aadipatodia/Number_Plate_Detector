# Indian Automatic Number Plate Recognition (ANPR)

This project develops a robust and accurate Automatic Number Plate Recognition (ANPR) system for Indian vehicles. The system is designed to accurately detect a number plate in an image and then apply an advanced Optical Character Recognition (OCR) engine to read the text. The pipeline is optimized for real-world scenarios, making it suitable for applications like traffic monitoring and parking management.

***

## 🚀 **Features**

* **Number Plate Detection:** The system utilizes a custom-trained **YOLOv8n** model, a state-of-the-art real-time object detection algorithm. Training the model from scratch on a specialized dataset allows it to precisely locate number plates under a variety of conditions.
* **Consolidated Dataset:** To build a robust model that avoids overfitting, the project's training data was meticulously consolidated from two distinct sources: **Kaggle**'s diverse image collection and a pre-annotated dataset from **Roboflow**. This combination ensures the model is exposed to a wide range of number plate styles, lighting, and camera angles.
* **Advanced OCR:** The project employs **EasyOCR** for text recognition, a significant improvement over less robust methods. EasyOCR is a powerful and flexible OCR library that excels at handling the varied fonts, angles, and potential distortions found on real-world number plates.
* **Modular Design:** The ANPR pipeline is structured into independent stages (detection and recognition), which allows for easy updates and improvements to the core model.

***

## 🧠 **Methodology**

### **1. Data Acquisition and Consolidation**
The project's success is rooted in its robust training data. We sourced two distinct datasets to maximize diversity and quantity:
* **Kaggle Dataset:** This collection contained images with annotations in **XML format**, which required a custom script to convert them to the **YOLO `.txt` format**.
* **Roboflow Dataset:** This dataset was downloaded directly in the **YOLOv8 format**, providing ready-to-use images and annotations.
All files were then consolidated into a single `combined_dataset` to create a large, unified training pool.

### **2. Data Preprocessing and Cleaning**
A critical and time-consuming step was cleaning the consolidated data to prepare it for training. We addressed two main issues:
* **Class Label Correction:** The Roboflow dataset contained multiple class labels that did not match our single-class detection task (`nc: 1`). A script was implemented to correct all class IDs to `0`, resolving the **corrupt image/label** errors encountered during training.
* **Image-Label Alignment:** A major challenge was the presence of images without corresponding labels, which led to a significant data imbalance. A cleanup script was created to delete these unlabeled images, resulting in a perfectly matched dataset.

### **3. Model Training**
The model was trained using **transfer learning**, a best practice for object detection. We started with a pre-trained **YOLOv8n** model, which has already learned to recognize general features from a large dataset. The model was then fine-tuned for **50 epochs** on the consolidated, clean dataset, teaching it to specifically identify number plates while retaining its general object recognition capabilities.

### **4. Inference Pipeline**
The final ANPR system operates in a two-stage pipeline:
* **Detection:** An input image is passed to the trained **YOLOv8** model, which outputs the precise coordinates of the number plate's bounding box.
* **Recognition:** The number plate region is cropped from the original image using the bounding box coordinates. This cropped image is then fed into the **EasyOCR reader**, which processes the characters and returns the final, recognized text.
