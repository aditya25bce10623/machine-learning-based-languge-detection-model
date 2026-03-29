import joblib
import gradio as gr
from deep_translator import GoogleTranslator
import sys

# 1. Load the Trained Model
model_path = r"D:\vityarthi_ad\lang_model.pkl"
try:
    mod = joblib.load(model_path)
    print("Model loaded successfully! Starting UI...")
except FileNotFoundError:
    print(f"Error: Could not find the model file at {model_path}.")
    sys.exit()

# 2. Define the Processing Logic
def process_text(input_text):
    """
    1. Detects Language using our custom ML Model.
    2. Translates Text to English using Google Translator.
    """
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
        # 'auto' lets Google detect source, converting to 'en' (English)
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
    description="This advanced tool uses a custom Machine Learning model to detect the language and Deep Learning APIs to translate it into English instantly.",
    
    live=False
)

# Launch the app with a public shareable link
iface.launch(share=True)