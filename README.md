# AI-languge-detector-based-on-machine-learning
## Project Overview
This project is a Machine Learning-based application that identifies the language of a given text. It supports multiple languages and provides an accuracy of nearly 97.56%. The project includes a GUI built with Gradio for ease of use
due to the size limit a pretrained model is not provided the model must be trained by the user.

## Features
- **Multi-language Support:** Detects text input in different languages.
- **High Accuracy:** it has high accuracy of 97.56% on test data.
- **Web GUI:** uses gradio for a simple intractive interface.
- **Instant translation** uses googles deeplearn library to translate text in english 
## Language used
- Python
## Libraries Used
- Scikit-learn (Machine Learning)
- Pandas (Data Handling)
- Gradio (Interface)
- joblib (file save and load)

## Model Architecture
The project uses a `TfidfCountVectorizer` for feature extraction and a `MultinomialNB` (Naive Bayes) classifier.

## Install Project Libraries**
```bash
pip install gradio
```
```bash
pip install pandas
```
```bash
pip install scikit-learn
```
```bash
pip install deep_translator
```
## How to Run
1. install all the nrequired libraries
2. due to the size limit a pretrained model is not provided the model must be trained by the user.
3. Download all the necessary files given.
4. paste the dataset (`Lang_Dataset.csv`) to the path and update the same in the trainer and reader end script
5. run the trainer.py script to train the model on given dataset.
6. on a sucessfull run trainer script will output a Gradio link

##  Methodology

This project uses a standard Natural Language Processing (NLP) pipeline to detect and translate text in real-time:

* **Data Preparation:** Text samples from multiple languages are merged using `pandas`, cleaned of duplicates, and split into an 80/20 training and testing set.
* **Feature Extraction (TF-IDF):** To handle unseen words and overlapping vocabulary (like Spanish and Italian), the text is processed using `TfidfVectorizer` with **character n-grams (3 to 5 letters)**. This trains the model to recognize unique prefixes, suffixes, and syllables rather than just whole words.
* **Model Training:** A **Multinomial Naive Bayes** classifier (`MultinomialNB`) calculates the probability of specific letter patterns belonging to a given language. The trained pipeline is then serialized with `joblib` for fast inference.
* **UI & Translation:** A **Gradio** web interface captures user text, runs it through the local ML model for detection, and then routes it through the `deep_translator` API for an instant English translation.

## Output samples
<img width="988" height="520" alt="Screenshot 2026-03-30 000106" src="https://github.com/user-attachments/assets/af54e8bb-d15e-4575-81f3-834f87c26f05" />

## Author
[Aditya prakash singh]
reg no- 25BCE10623
