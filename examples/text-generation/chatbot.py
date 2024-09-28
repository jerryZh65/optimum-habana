import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.habana import GaudiConfig

def load_model_and_tokenizer(model_name):
    """
    Load the model and tokenizer for text generation.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Switch to evaluation mode
    model.eval()

    return model, tokenizer

def generate_response(prompt, model, tokenizer, max_length=50):
    """
    Generate a response from the model based on the input prompt.
    """
    # Encode the input prompt into tokens
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Create attention mask
    attention_mask = torch.ones_like(input_ids)
    
    # Generate output using the model
    # top_k and top_p: These control how many words the model considers at each generation step. 
    # A lower top_k (like 30) or a more conservative top_p (like 0.9) will restrict the word choice and make the model more consistent.
    output = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2, # Ensure no repetition of n-grams
        do_sample=True, # Use sampling to allow diverse outputs
        top_k=30, # Consider top 30 tokens at each generation step
        top_p=0.9, # Use nucleus sampling for more coherent output
        temperature=0.6, # Reduce temperature for more coherent responses
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,  # Set pad_token_id to eos_token_id
        eos_token_id=tokenizer.eos_token_id,  # Ensure it terminates at EOS token
        repetition_penalty=1.3,  # Increase repetition penalty
        num_beams=3, # Beam search explores multiple possibilities for generating text and picks the most coherent output.
        early_stopping = True # Stop generating once the model finds an appropriate stopping point
    )
    
    # Decode the output back into text
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Return only the generated part of the response
    return response[len(prompt):]

def chatbot():
    """
    Main function to start the chatbot.
    """
    # Define the model name (you can change this to any supported model)
    # model_name = "gpt2"  # Replace with any supported Gaudi-optimized model if needed
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Start the chat loop
    print("Chatbot is ready! Type 'exit' to end the conversation.")
    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Generate and print the bot's response
        bot_response = generate_response(user_input, model, tokenizer)
        print(f"Bot: {bot_response}")

if __name__ == "__main__":
    chatbot()
