import torch
import numpy as np

from Modules import EDTransformer
from Data import TOKENS, answer_to_token_ids, equation_to_token_ids

model = EDTransformer(emb_dim=32, num_tokens=len(TOKENS), max_context_length=9, max_primary_length=7, encoder_layers=4, decoder_layers=4, heads=4)
model.load_state_dict(torch.load("./models/model_chkpt_190.pt"))

# Test inference
equation = ["69 + 69"]
equation_tokens = equation_to_token_ids(equation, max_length=9).unsqueeze(0)

answer_tokens = torch.LongTensor([[TOKENS.index('<BOS>')]])

model.eval()
MAX_OUTPUT = 15
with torch.no_grad():
    pred_token = ""
    output_str = ""
    count = 0
    while (pred_token != '<END>' and count < MAX_OUTPUT):
        count += 1
        probs = model(equation_tokens, answer_tokens)

        pred_token_id = torch.argmax(probs[0, -1, :])
        pred_token = TOKENS[pred_token_id]
        answer_tokens = torch.cat([answer_tokens, torch.LongTensor([[pred_token_id]])], dim=1)
        output_str += pred_token
        print(f"Predicted Token: {pred_token}")
        print(f"Current Output: {output_str}")

dummy_input1 = torch.randint(low=0, high=len(TOKENS) - 1, size=(1, 9))
dummy_input2 = torch.randint(low=0, high=len(TOKENS) - 1, size=(1, 7))

# dummy_input1 = torch.randint(1, 9)
# dummy_input2 = torch.randint(1, 7)

torch.onnx.export(
    model,
    (dummy_input1, dummy_input2),
    "model.onnx",
    export_params=True,
    input_names=["context", "primary"],
    output_names=["output"],
    dynamic_axes={
        "input_1": {0: "batch_size", 1: "feature_dim_1"},  # Allow dynamic batch & feature dims
        "input_2": {0: "batch_size", 1: "feature_dim_2"},  # Same for second input
        "output": {0: "batch_size"}  # Make output batch size dynamic
    }
)

# # Verify the ONNX model
import onnx
import onnxruntime

onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession("model.onnx")


equation_tokens = equation_to_token_ids(equation, max_length=9).unsqueeze(0)

answer_tokens = torch.LongTensor([[TOKENS.index('<BOS>'), TOKENS.index('1'), TOKENS.index('<BOS>'), TOKENS.index('<BOS>'), TOKENS.index('<BOS>'), TOKENS.index('<BOS>'), TOKENS.index('<BOS>')]])

onnx_input = {
    "context": equation_tokens.numpy(),
    "primary": answer_tokens.numpy(),
}
onnx_output = torch.tensor(ort_session.run(None, onnx_input))

print(f"type(onnx_output): {type(onnx_output)}")
print(f"onnx_output.shape: {onnx_output.shape}")
print("ONNX Output: ", onnx_output)


