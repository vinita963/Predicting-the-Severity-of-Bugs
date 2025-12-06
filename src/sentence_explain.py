#This script tells us that which sentence in the bug report influenced the severity prediction the most
import nltk
from predict import predict_with_confidence
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F
import torch

nltk.download('punkt_tab')

# Load model n tokenizer
tokenizer = RobertaTokenizer.from_pretrained("severity_model")
model = RobertaForSequenceClassification.from_pretrained("severity_model")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Scores just one sentence and returns probability scores for each class
def score_sentence(sentence):
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    return probs

# Human-centered explainability
def highlight_sentences(text, label_encoder):

    #Break the bug report into sentences
    sentences = nltk.sent_tokenize(text)
    sentence_scores = []
    
    #Score each sentence independently
    for sent in sentences:
        probs = score_sentence(sent)
        sentence_scores.append((sent, probs))

    # Identify most influential sentence for predicted class
    final_pred = predict_with_confidence(text)
    pred_class = final_pred["prediction"]
    pred_idx = list(label_encoder.classes_).index(pred_class)

    #The sentence with highest score for the predicted class comes first.
    ranked = sorted(sentence_scores, key=lambda x: x[1][pred_idx], reverse=True)

    return {
        "predicted_class": pred_class,
        "confidence": final_pred["confidence"],
        "ranked_sentences": [
            {"sentence": s, 
             "score_for_predicted_class": float(p[pred_idx])}
            for s, p in ranked
        ]
    }
