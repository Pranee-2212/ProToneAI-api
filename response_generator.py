import torch
from unsloth import FastLanguageModel
import json
import time

# --- Configuration ---
# Set this to the directory where you saved your merged model in 4-bit
MODEL_PATH = "./model2"
MAX_SEQ_LENGTH = 2048 # Must match the length used in fine-tuning
LOAD_IN_4BIT = True # Must match the saving method

# Define the chat prompt template used during training
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Global variables for the model and tokenizer to ensure singleton loading
loaded_model = None
loaded_tokenizer = None



def load_model_and_tokenizer(model_path: str, max_seq_length: int, load_in_4bit: bool):
    """Loads the fine-tuned model and tokenizer only once."""
    global loaded_model, loaded_tokenizer
    if loaded_model is None or loaded_tokenizer is None:
        print(f"Loading model from {model_path}...")
        loaded_model, loaded_tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
            local_files_only=True # Ensure loading from local files only
        )
        FastLanguageModel.for_inference(loaded_model)
        print("Model loaded successfully and ready for inference.")
    return loaded_model, loaded_tokenizer




def generate_response(model, tokenizer, instruction: str, input_context: str = "", max_new_tokens: int = 192, temperature: float = 0.5):
    """Generates a response from the loaded model with adjustable parameters."""
    EOS_TOKEN = tokenizer.eos_token

    prompt = ALPACA_PROMPT.format(instruction, input_context, "")

    inputs = tokenizer(
        [prompt],
        return_tensors="pt"
    ).to("cuda")

    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )
    end_time = time.time()
    response_time = end_time - start_time

    response = tokenizer.batch_decode(outputs)[0]

    # Simple cleanup: remove the prompt and the EOS token
    response_start_tag = "### Response:\n"
    response_end_tag = EOS_TOKEN

    start_index = response.find(response_start_tag)
    if start_index != -1:
        response = response[start_index + len(response_start_tag):].strip()

    if response.endswith(response_end_tag):
        response = response[:-len(response_end_tag)].strip()

    return response, response_time




# --- Test Cases ---
test_cases = [
    {
        "instruction": "Explain the concept of prompt engineering in the context of large language models.",
        "input": "Provide a simple example to illustrate your explanation."
    },
    {
        "instruction": "Write a short, creative story about a robot who discovers a love for painting.",
        "input": ""
    },
    {
        "instruction": "Summarize the main points of the provided text.",
        "input": "The quick brown fox jumps over the lazy dog. This is a classic pangram used to test typewriters and computer keyboards. Pangrams are sentences that contain every letter of the alphabet at least once."
    },
    {
        "instruction": "What is the capital of France?",
        "input": ""
    },
    {
        "instruction": "Generate a list of 5 tips for effective time management.",
        "input": "Focus on techniques for students."
    },
]



# --- Load Model and Run Tests ---
model, tokenizer = load_model_and_tokenizer(MODEL_PATH, MAX_SEQ_LENGTH, LOAD_IN_4BIT)

print("\n--- Running Tests with Optimized Functions ---")
for i, case in enumerate(test_cases):
    instruction = case["instruction"]
    input_context = case.get("input", "")

    print(f"\n--- Test Case {i+1} ---")
    print(f"Instruction: {instruction}")
    if input_context:
        print(f"Input: {input_context}")

    generated_text, response_time = generate_response(model, tokenizer, instruction, input_context)

    print(f"Generated Response (Time: {response_time:.2f} seconds):")
    print(generated_text)
    print("-" * 30)

# --- Notes for Deployment ---
print("\n--- Notes for Deployment ---")
print("This code block demonstrates loading and inference. For a web app, you would typically wrap the inference logic in a web framework (like Flask or FastAPI) and handle incoming requests.")
print("Ensure your deployment environment has the necessary libraries (torch, unsloth, transformers) and a compatible GPU.")
print("Consider exploring inference backends like ONNX Runtime or TensorRT for further performance optimization in a production environment.")