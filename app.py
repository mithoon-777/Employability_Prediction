import joblib
import gradio as gr
import numpy as np
import os

MODEL_PATH = os.path.join(os.getcwd(), "employability_model_selected.joblib")
ENCODER_PATH = os.path.join(os.getcwd(), "label_encoder_fixed.joblib")

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

def predict_employability(manner_of_speaking, self_confidence, ability_to_present_ideas, communication_skills, mental_alertness):
    """Predict employability based on user input."""
    input_data = np.array([[manner_of_speaking, self_confidence, ability_to_present_ideas, communication_skills, mental_alertness]])
    prediction = model.predict(input_data)[0]
    result = label_encoder.inverse_transform([prediction])[0]
    
    return f"✅ {result}" if result == "Employable" else f"😞 {result}"

iface = gr.Interface(
    fn=predict_employability,
    inputs=[
        gr.Slider(1, 5, step=1, label="Manner of Speaking"),
        gr.Slider(1, 5, step=1, label="Self-Confidence"),
        gr.Slider(1, 5, step=1, label="Ability to Present Ideas"),
        gr.Slider(1, 5, step=1, label="Communication Skills"),
        gr.Slider(1, 5, step=1, label="Mental Alertness")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Employability Prediction",
    description="Rate yourself on the given attributes (1-5) to check your employability status.  (Mithoon😉)"
)

if __name__ == "__main__":
    iface.launch(share=True)
