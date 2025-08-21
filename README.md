Indian Automatic Number Plate Recognition (ANPR)
This project develops a robust and accurate Automatic Number Plate Recognition (ANPR) system for Indian vehicles. The system uses a deep learning model to first detect the number plate in an image and then applies an advanced Optical Character Recognition (OCR) engine to read the text.

🚀 Features
Number Plate Detection: Uses a custom-trained YOLOv8 model to accurately locate number plates in diverse images.

High Accuracy: The model is trained on a consolidated dataset from Kaggle and Roboflow, ensuring high performance on real-world images.

Optimized OCR: Employs EasyOCR, a state-of-the-art OCR library, to extract text from the detected number plates, outperforming traditional OCR methods like pytesseract.

Modular Pipeline: The system is built in a modular fashion, allowing for easy updates and improvements to both the detection and recognition components.

🧠 Methodology
Data Consolidation: We combined data from two primary sources: a sample dataset from Kaggle and a larger, pre-annotated dataset from Roboflow. This was crucial to overcome the limitations of a small dataset and prevent model overfitting.

Data Cleaning: The datasets had different class labels and were imbalanced (more images than labels). We implemented a Python script to correct the class IDs and delete all images that lacked a corresponding annotation file. This ensured a perfectly balanced dataset for training.

Model Training: A pre-trained YOLOv8n model was fine-tuned on the cleaned, consolidated dataset for 50 epochs.

Inference & Recognition: The final trained model is used to predict the location of a number plate. The cropped image of the number plate is then passed to EasyOCR for highly accurate text extraction.
