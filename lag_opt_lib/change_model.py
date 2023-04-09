import torch
import mace

# Load the model
model = torch.load('MACE_model_cpu.model')

# Convert all float tensors to double tensors
model = model.double()

# Save the converted model
torch.save(model, 'MACE_model_cpu_double.model')


