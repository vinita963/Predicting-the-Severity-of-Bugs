#This generates a token heatmap showing which words contributed most.
import shap
from transformers import pipeline
from predict import model, tokenizer

# Prediction pipeline
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

# SHAP explainer
explainer = shap.Explainer(pipe)

def explain_tokens(text):
    shap_values = explainer([text])
    shap.plots.text(shap_values[0])

explain_tokens("The system fails to load due to null pointer dereference")
