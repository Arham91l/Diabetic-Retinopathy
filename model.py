import torch
import torch.nn as nn
from transformers import Swinv2Model

MODEL_NAME = "microsoft/swinv2-tiny-patch4-window16-256"


class SwinClassifier(nn.Module):
    """
    Swin Transformer V2 backbone + custom classification head
    for 5-class diabetic retinopathy severity grading.
    """

    def __init__(self, model_name: str = MODEL_NAME, num_classes: int = 5):
        super().__init__()
        self.swin = Swinv2Model.from_pretrained(model_name)
        hidden_size = self.swin.config.hidden_size

        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        outputs = self.swin(pixel_values=x)
        pooled = outputs.pooler_output
        x = self.dropout1(pooled)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)
