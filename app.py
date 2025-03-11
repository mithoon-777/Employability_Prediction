import joblib
import gradio as gr
import numpy as np

def predict_employability(manner_of_speaking, self_confidence, ability_to_present_ideas, communication_skills, mental_alertness):
    # Load the updated model and label encoder
    model = joblib.load("employability_model_selected.joblib")
    label_encoder = joblib.load("label_encoder_fixed.joblib") 
    
    # Prepare the input data
    input_data = np.array([[manner_of_speaking, self_confidence, ability_to_present_ideas, communication_skills, mental_alertness]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Decode the prediction
    result = label_encoder.inverse_transform([prediction])[0]
    
    # Return the result with an emoji
    if result == "Employable":
        return f"✅ {result}"
    else:
        return f"😞 {result}"

# Define the Gradio interface
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
    description="Rate yourself on the given attributes (1-5) to check your employability status.  (AsthanM😉)"
)

# Run the Gradio app
if __name__ == "__main__":
    iface.launch()
