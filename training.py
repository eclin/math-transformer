import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import random
import numpy as np

from typing import Tuple
import time

from Data import DataGenerator
from Modules import EDTransformer

SEED = 69
torch.manual_seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    print("GPU is available!")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available. Using CPU.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_generator = DataGenerator()
math_data = data_generator.generate_data()

# Create DataLoaders
BATCH_SIZE = 32
train_dataset = TensorDataset(math_data.train_equations, math_data.train_answers)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = TensorDataset(math_data.test_equations, math_data.test_answers)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

EPOCHS = 30
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EMB_SIZE = 32

model = EDTransformer(EMB_SIZE, math_data.num_tokens, math_data.max_equation_length, math_data.max_answer_length, encoder_layers=4, decoder_layers=4, heads=4)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss() # (ignore_index=tokens.index('<PAD>'))

# torch.save(model.state_dict(), "./model.pt")
model.load_state_dict(torch.load("./model_chkpt_190.pt", map_location=torch.device(device)))
for epoch in range(30, 30 + EPOCHS):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for equations, answers in train_dataloader:
        equations, answers = equations.to(device), answers.to(device)
        optimizer.zero_grad()

        output = model(equations, answers[:, :-1])
        output = output.view(-1, math_data.num_tokens)
        target = answers[:, 1:].reshape(-1)

        loss = loss_fn(output, target)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        predictions = output.argmax(dim=1)
        correct_predictions += (predictions == target).sum().item()
        total_predictions += target.numel()

    avg_loss = total_loss / len(train_dataloader)
    accuracy = correct_predictions / total_predictions
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"./model_chkpt_{epoch + 1}.pt")