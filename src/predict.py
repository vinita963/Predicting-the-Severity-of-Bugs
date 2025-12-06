#This script loads your trained RoBERTa bug severity classifier and predicts: the severity label , the confidence score , confidence for all classes
import torch
import joblib
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model, tokenizer, label encoder
model = RobertaForSequenceClassification.from_pretrained("severity_model").to(device)
tokenizer = RobertaTokenizer.from_pretrained("severity_model")
label_encoder = joblib.load("label_encoder.pkl") #maps numbers back to labels (critical, major, minor, trivial)

#This disables: dropout , gradient calculation; So prediction is stable and fast.
model.eval()

#This function takes user text and outputs prediction + confidence.
def predict_with_confidence(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
 
    pred_label = probs.argmax()  #argmax → index of highest probability
    pred_class = label_encoder.inverse_transform([pred_label])[0]    #inverse_transform → convert index → label
    confidence = probs[pred_label]   #confidence → probability of predicted class

    return {
        "prediction": pred_class,
        "confidence": float(confidence),
        "all_confidences": {label_encoder.inverse_transform([i])[0]: float(p) 
                            for i, p in enumerate(probs)}
    }


print(predict_with_confidence("Application crashes when clicking on save"))
