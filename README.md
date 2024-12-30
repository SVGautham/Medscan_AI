# Eye Disease Classification with Deep Learning

This project focuses on developing and optimizing a deep learning model for eye disease classification using ResNet50. The model was trained on over 5,000 medical images to achieve an accuracy of 92%. Various techniques such as data preprocessing, augmentation, and transfer learning were applied to enhance the model's performance and inference speed.

## Key Achievements

- **Deep Learning Model**: Developed and optimized a model using **ResNet50** architecture, achieving **92% accuracy** in classifying eye diseases.
- **Data Preprocessing**: Preprocessed and augmented **5,000+ medical images** using Python libraries like OpenCV and TensorFlow. This step helped improve model robustness and reduced image noise by **30%**.
- **Transfer Learning**: Applied transfer learning and feature extraction techniques, resulting in a **25% improvement** in model performance and inference time.

## Frontend View
![Screenshot 2024-12-23 202144](https://github.com/user-attachments/assets/01f302f4-52f1-4a61-88cc-c86cd7e05757)
![Screenshot 2024-12-23 202256](https://github.com/user-attachments/assets/0ce3bbe1-ed34-49c5-9be2-b325468c8eaf)
![Screenshot 2024-12-23 202309](https://github.com/user-attachments/assets/1d8f8a7a-2dd9-4934-bb1c-49bbcb697652)

## Requirements

Before running the project, ensure you have the following libraries installed. You can install them using the `requirements.txt` file provided.

### Libraries

- Python 3.x
- TensorFlow
- OpenCV
- Streamlit
- Other dependencies in the `requirements.txt` file.

You can install the required libraries by running:

```bash
pip install -r requirements.txt

# Example: Set your local path for the dataset in task2.py
dataset_path = 'your/local/dataset/path'

# run the python code to train the model
python task2.py

pip install streamlit
streamlit run appp.py
to get the final frontend

