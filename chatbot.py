from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Chatbot:
    def __init__(self, model_name='microsoft/DialoGPT-large'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.chat_history_id = None

    def get_response(self, user_input):
        # Encode the user input and add it to the chat history
        new_input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([self.chat_history_id, new_input_ids], dim=-1) if self.chat_history_id is not None else new_input_ids

        # Generate a response from the model
        self.chat_history_id = self.model.generate(bot_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id,do_sample=True, top_k=50, top_p=0.95, temperature=0.8)
        
        # Decode the response and return it
        response = self.tokenizer.decode(self.chat_history_id[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response
    

def main():
    chatbot = Chatbot()
    print("Chatbot is ready! Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        response = chatbot.get_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()
