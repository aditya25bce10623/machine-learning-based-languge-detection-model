import joblib
import gradio as gr
from deep_translator import GoogleTranslator
import sys
import os # Imported os to handle relative file paths

# Get the absolute path of the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Load the Trained Model
# Join the base directory with the model filename
model_path = os.path.join(BASE_DIR, "lang_model.pkl")

try:
    mod = joblib.load(model_path)
    print("Model loaded successfully! Starting UI...")
except FileNotFoundError:
    print(f"Error: Could not find the model file at {model_path}.")
    print("Make sure 'lang_model.pkl' is in the exact same folder as this script.")
    sys.exit()

# 2. Define the Processing Logic
def process_text(input_text):
    
    """Detects Language using our custom ML Model."""

    if not input_text or len(input_text.strip()) < 2:
        return "Please enter valid text.", "No translation available."

    # Detection using our ML Model
    try:
        prediction = mod.predict([input_text])
        detected_lang = prediction[0]
        detection_result = f"Detected: {detected_lang.upper()}"
    except Exception as e:
        detection_result = f"Detection Error: {e}"

    # Translation using Deep Translator
    try:
        translated_text = GoogleTranslator(source='auto', target='en').translate(input_text)
    except Exception as e:
        translated_text = f"Translation Error: {e}"

    return detection_result, translated_text

# 3. Build and Launch the Gradio Interface
iface = gr.Interface(
    fn=process_text,
    
    # Input
    inputs=gr.Textbox(lines=4, label="Enter Text (Any Language)", placeholder="Type Hindi, French, Spanish, etc..."),
    
    # Outputs
    outputs=[
        gr.Text(label=" AI Detection Result"),
        gr.Text(label=" English Translation")
    ],
    
    # Title & Description
    title="AI Language Detector & Translator",
    description="This advanced tool uses a custom Machine Learning model to detect the language and Deep Translator to translate it into English instantly.",
    
    live=False
)

# Launch the app with a public shareable link
iface.launch(share=True)