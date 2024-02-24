from transformers import T5ForConditionalGeneration, T5Config

pretrained_path_A = "/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-splade/t5-splade-0/checkpoint/"
# Load model A
model_A = T5ForConditionalGeneration.from_pretrained(pretrained_path_A)

# Initialize model B with 't5-base' configuration
model_B_config = T5Config.from_pretrained("t5-base")
model_B = T5ForConditionalGeneration(model_B_config)

# Copy encoder weights from model A to model B
model_B.encoder.load_state_dict(model_A.encoder.state_dict())

# Load the weights from 't5-base' for the first 11 decoder layers
t5_base = T5ForConditionalGeneration.from_pretrained("t5-base")

# Copy the weights from 't5-base' for the first 11 decoder layers
for i in range(11):  # model_B has 12 decoder layers, and we're filling in the first 11
    model_B.decoder.block[i].load_state_dict(t5_base.decoder.block[i].state_dict())

# Map the single decoder layer from model A to the last decoder layer of model B
state_dict_A = model_A.decoder.block[0].state_dict()
state_dict_A.pop("layer.0.SelfAttention.relative_attention_bias.weight", None)
print(state_dict_A.keys())
#print(model_B.decoder.block[-1])

model_B.decoder.block[-1].load_state_dict(state_dict_A)
# Save the new model B if needed
pretrained_path_B = "/home/ec2-user/quic-efs/user/hansizeng/work/term_generative_retriever/experiments-splade/t5-splade-0/checkpoint_12l/"
model_B.save_pretrained(pretrained_path_B)