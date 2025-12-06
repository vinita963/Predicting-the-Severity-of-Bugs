
import streamlit as st
import torch
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import joblib
import os
import shap
import matplotlib.pyplot as plt
from torch.nn import functional as F

# ---------- CONFIG ----------
MODEL_DIR = "bug_severity_model"        # folder where model + tokenizer is saved
LABEL_ENCODER_PATH = "label_encoder.pkl"  # label encoder saved by my training script
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#This function loads the model ONE TIME and caches it
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    # Try to load from saved directory, otherwise fallback to roberta-base
    if os.path.isdir(MODEL_DIR):
        try:
            tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
            model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR, output_attentions=True)
            st.info(f"Loaded model/tokenizer from `{MODEL_DIR}`")
        except Exception as e:
            st.warning(f"Could not load from {MODEL_DIR}: {e}. Falling back to roberta-base.")
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            model = RobertaForSequenceClassification.from_pretrained("roberta-base", output_attentions=True)
    else:
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaForSequenceClassification.from_pretrained("roberta-base", output_attentions=True)
    model.to(DEVICE)
    model.eval()
    return model, tokenizer

#Load the Label Encoder
@st.cache_resource(show_spinner=False)
def load_label_encoder():
    if os.path.exists(LABEL_ENCODER_PATH):
        try:
            le = joblib.load(LABEL_ENCODER_PATH)
            st.info(f"Loaded label encoder from {LABEL_ENCODER_PATH}")
            return le
        except Exception as e:
            st.warning(f"Failed loading label encoder: {e}. Using numeric labels.")
    # fallback: use dummy mapping
    class DummyLE:
        def inverse_transform(self, arr):
            return [str(int(x)) for x in arr]
        @property
        def classes_(self):
            return np.arange(2)
    return DummyLE()

model, tokenizer = load_model_and_tokenizer()
le = load_label_encoder()

# Optional: if you saved a temperature value (for calibration), try to load
TEMPERATURE_PATH = "temperature.npy"
if os.path.exists(TEMPERATURE_PATH):
    try:
        temperature = float(np.load(TEMPERATURE_PATH))
        st.info(f"Loaded temperature scaling: {temperature:.3f}")
    except Exception:
        temperature = 1.0
else:
    temperature = 1.0


# Predict raw logits + attention
def predict_logits(text):
    # returns raw logits (torch tensor on cpu), and model outputs (including attentions)
    enc = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc, output_attentions=True, return_dict=True)
    logits = outputs.logits.squeeze(0).cpu()
    attentions = outputs.attentions  # tuple of (layer tensors) on device
    return logits, attentions, enc

#Convert logits → probability (with calibration)
def calibrated_probs_from_logits(logits, temperature=1.0):
    # logits: torch tensor (C,)
    if temperature <= 0:
        temperature = 1.0
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1).cpu().numpy()
    return probs

#  SENTENCE-LEVEL IMPORTANCE 
from nltk import tokenize as nltk_tokenize

@st.cache_data
def sentence_level_scores(text):
    """Return list of (sentence, probs) for each sentence."""
    sentences = nltk_tokenize.sent_tokenize(text)
    scores = []
    for s in sentences:
        logits, _, _ = predict_logits(s)
        probs = calibrated_probs_from_logits(logits, temperature)
        scores.append((s, probs))
    return sentences, np.array([p for (_, p) in scores])

#  ATTENTION-BASED TOKEN IMPORTANCE (fallback) 
def attention_token_importances(attentions, enc):
    """
    Simple heuristic: average attention from last layer over heads, from CLS token to other tokens.
    attentions: tuple(layer tensors) each shape (batch, heads, seq_len, seq_len)
    enc: tokenized encoding dict with input_ids
    Returns: tokens (list str), importances (np.array len seq_len)
    """
    # get last layer attentions (take CPU)
    last_layer = attentions[-1].detach().cpu().numpy()  # shape (1, heads, seq_len, seq_len)
    # average over heads and take CLS token (index 0) attention to tokens
    cls_to_tokens = last_layer[0].mean(axis=0)[0]  # shape (seq_len,)
    input_ids = enc['input_ids'].cpu().numpy()[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    # Align: only keep non-pad tokens (we used padding max_length, but tokens include pad token id)
    # find first pad (if any)
    try:
        pad_token_id = tokenizer.pad_token_id
    except Exception:
        pad_token_id = 1
    nonpad_mask = input_ids != pad_token_id
    tokens = [t for t, m in zip(tokens, nonpad_mask) if m]
    importances = cls_to_tokens[:len(tokens)]
    # normalize
    importances = np.abs(importances)
    if importances.sum() > 0:
        importances = importances / importances.max()
    return tokens, importances


#  SHAP EXPLAINER (caching)
@st.cache_resource(show_spinner=False)
def build_shap_explainer():
    # wrap a simple callable that accepts a list of texts and returns probability of each class
    def f(texts):
        # texts: list of strings -> return np.array (n_samples, n_classes)
        outs = []
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, truncation=True, padding='max_length', max_length=128, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                logits = model(**enc).logits
                probs = F.softmax(logits / temperature, dim=-1).cpu().numpy()
            outs.append(probs)
        return np.vstack(outs)
    # Create a KernelExplainer can be slow; use the model-based explainer if possible
    try:
        explainer = shap.Explainer(f, masker=tokenizer)  # shap will try to adapt
    except Exception:
        explainer = None
    return explainer

explainer = build_shap_explainer()

# STREAMLIT LAYOUT
st.set_page_config(page_title="Bug Severity Predictor — Explainable", layout="wide")
st.title("Bug Severity Predictor — Explainable Demo")
st.markdown(
    "Enter a bug description (from an issue tracker) and the app will show:\n\n"
    "- predicted severity\n- calibrated confidence scores\n- highlighted important sentences\n- token-level explanations (SHAP) or attention heatmap\n"
)


# Input area
text_input = st.text_area("Enter bug description", height=200, value="When I try to save changes, the application throws a NullPointerException and the UI crashes.")

col1, col2 = st.columns([2, 1])
with col2:
    st.subheader("Settings")
    use_shap = st.checkbox("Use SHAP explanations (can be slow)", value=False)
    top_k_sentences = st.slider("Top sentences to highlight", min_value=1, max_value=5, value=2)
    run_button = st.button("Explain & Predict")

# Cached predictor
@st.cache_data(show_spinner=False)
def do_predict_and_explain(text, use_shap_flag, top_k):
    result = {}
    # overall logits
    logits, attentions, enc = predict_logits(text)
    probs = calibrated_probs_from_logits(logits, temperature)
    pred_idx = int(np.argmax(probs))
    pred_label = le.inverse_transform([pred_idx])[0] if hasattr(le, 'inverse_transform') else str(pred_idx)
    # per-class probabilities
    result['probs'] = probs
    result['pred_idx'] = pred_idx
    result['pred_label'] = pred_label

    # sentence-level
    sentences, sent_probs = sentence_level_scores(text)
    # compute score per sentence as probability for predicted class
    pred_class_probs = sent_probs[:, pred_idx] if sent_probs.shape[1] > pred_idx else sent_probs.max(axis=1)
    sent_with_score = list(zip(sentences, pred_class_probs))
    sent_with_score_sorted = sorted(sent_with_score, key=lambda x: x[1], reverse=True)
    result['sentence_ranking'] = sent_with_score_sorted[:top_k]

    # token-level: try SHAP if requested and explainer available
    if use_shap_flag and explainer is not None:
        try:
            shap_values = explainer([text])
            # shap_values contains explanation for each class; we'll return the object for plotting
            result['shap_values'] = shap_values
        except Exception as e:
            result['shap_error'] = str(e)
            result['shap_values'] = None
    else:
        # fallback to attention importances
        tokens, token_imps = attention_token_importances(attentions, enc)
        result['tokens'] = tokens
        result['token_importances'] = token_imps
    return result

if run_button:
    with st.spinner("Running model and computing explanations..."):
        outcome = do_predict_and_explain(text_input, use_shap, top_k_sentences)

    # Display prediction + confidence 
    st.subheader("Prediction")
    pred_label = outcome['pred_label']
    probs = outcome['probs']
    class_names = list(le.classes_) if hasattr(le, 'classes_') else [str(i) for i in range(len(probs))]
    st.markdown(f"**Predicted severity:** `{pred_label}`")
    # show per-class probabilities table
    prob_df = pd.DataFrame({"class": class_names, "probability": probs})
    prob_df = prob_df.sort_values("probability", ascending=False)
    st.table(prob_df)

    # Confidence bar for top prediction
    top_prob = probs[outcome['pred_idx']]
    st.write("Confidence for top prediction:")
    st.progress(float(top_prob))  # visually shows the confidence
    st.write(f"{top_prob*100:.2f}%")

    # Sentence highlighting 
    st.subheader("Sentence-level highlights")
    top_sentences = outcome['sentence_ranking']
    highlight_html = ""
    # show sentences with a light yellow background proportional to score
    for s, score in top_sentences:
        intensity = int(255 - (score * 120))  # smaller number -> deeper highlight
        # clamp
        intensity = max(120, min(255, intensity))
        highlight_html += f"<div style='padding:6px;margin:4px 0;background-color: rgba(255, 245, 160, {0.3 + score*0.7});'>"
        highlight_html += f"<b>score={score:.3f}</b> — {s}</div>"
    st.markdown(highlight_html, unsafe_allow_html=True)

    #  Token-level: SHAP or attention
    st.subheader("Token-level explanation")
    if 'shap_values' in outcome and outcome['shap_values'] is not None:
        try:
            st.write("SHAP token-level explanation (text):")
            # shap.plots.text produces a matplotlib/JS interactive representation.
            # We'll draw it to a matplotlib figure and display via Streamlit
            shap_vals = outcome['shap_values'][0]
            plt.figure(figsize=(12,2))
            # shap's text plot is a special HTML/JS visualization; we attempt the matplotlib fallback:
            shap.plots.text(shap_vals)
            st.pyplot(bbox_inches='tight')
        except Exception as e:
            st.error(f"SHAP plot failed to render: {e}. Showing attention fallback.")
            if 'token_importances' in outcome:
                tokens = outcome['tokens']; imps = outcome['token_importances']
                fig, ax = plt.subplots(figsize=(12,2))
                ax.bar(range(len(tokens)), imps)
                ax.set_xticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
                st.pyplot(fig)
    else:
        # attention fallback
        tokens = outcome.get('tokens', [])
        imps = outcome.get('token_importances', np.array([]))
        if len(tokens):
            fig, ax = plt.subplots(figsize=(12,2))
            ax.bar(range(len(tokens)), imps)
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
            ax.set_title("Token importances (attention-based heuristic)")
            st.pyplot(fig)
        else:
            st.write("No token-level info available.")

    st.success("Done — use the controls to tweak SHAP / top sentences / input text.")
else:
    st.info("Enter text and click **Explain & Predict** to run the model.")

