# AI-languge-detector-based-on-machine-learning
## Project Overview
This project is a Machine Learning-based application that identifies the language of a given text. It supports 7 languages and provides an accuracy nearly 97.56%. The project includes a GUI built with Gradio for ease of use

## Features
- **Multi-language Support:** Detects text input in different languages.
- **High Accuracy:** it has high accuracy of 97.56% on test data.
- **Web GUI:** uses gradio for a simple intractive interface.
- 
## Language used
- Python
## Libraries Used
- Scikit-learn (Machine Learning)
- Pandas (Data Handling)
- Gradio (Interface)
- joblib (file save and load)

## Model Architecture
The project uses a `TfidfCountVectorizer` for feature extraction and a `MultinomialNB` (Naive Bayes) classifier.

**. Install Project Libraries**
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
2. Download all the necessary files given.
3. run the trainer.py script to generate the model or the pre trained model given can also be used.
4. paste the dataset (`Lang_Dataset.csv`) to the path and update the same in the trainer and reader end script
5. on a sucessfull run trainer script will output a Gradio link


## Author
[Aditya prakash singh]
reg no- 25BCE10623
