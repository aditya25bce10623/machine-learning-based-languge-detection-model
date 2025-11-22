# AI-languge-detector-based-on-machine-learning
## Project Overview
This project is a Machine Learning-based application that identifies the language of a given text. It supports 7 languages and provides an accuracy nearly 98.26%. The project includes a GUI built with Gradio for ease of use

## Features
- **Multi-language Support:** Detects text input in  7 different languages.
- **High Accuracy:** it has high accuracy of 98.26% on test data.
- **Web GUI:** uses gradio for a simple intractive interface.
- **Optimized Performance:** Loads pre trained models from Google Drive to save time.

## Technologies Used
- Python
- Scikit-learn (Machine Learning)
- Pandas (Data Handling)
- Gradio (Interface)
- Google Colab (IDE)

## Model Architecture
The project uses a `CountVectorizer` for feature extraction and a `MultinomialNB` (Naive Bayes) classifier.

## How to Run
1. Open the provided Google Colab Notebook.
2. connect allow acces to your google drive
3. Upload the provided dataset (`language_data.csv`) to your Google Drive or session.
4. Run the cells sequentially.
5. The final cell will launch a Gradio link for the GUI.

## Author
[Aditya prakash singh]
reg no- 25BCE10623
