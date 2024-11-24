import os
import time
from dotenv import load_dotenv
from typing import List
from dataclasses import dataclass
import goodfire
import numpy as np
import torch
from sklearn.metrics import f1_score
import gc

from TuringLLM.inference import TuringLLMForInference
from SAE.SAE_TopK import SAE





load_dotenv()
GOODFIRE_API_KEY = os.environ.get('GOODFIRE_API_KEY')
goodfire_client = goodfire.Client(GOODFIRE_API_KEY)

turing_sae_subset_layer_latent_count = 320
turing_sae_sum_latent_values = torch.zeros((12, turing_sae_subset_layer_latent_count))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class TuringLLMConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 16
    n_embd: int = 1024
    hidden_size: int = 4096
    norm_eps: float = 1e-5
    
sae_dim = 40960






def generate_sentence(prompt: str = "", variant = False, max_completion_tokens: int = 24) -> str:
    variant = goodfire.Variant("meta-llama/Meta-Llama-3-8B-Instruct")
    # prompt = f"Please write just a single sentence that has a word containing '{token}' in it"
    response = goodfire_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=variant,
        stream=False,
        max_completion_tokens=max_completion_tokens,
    )
    sentence = response.choices[0].message['content'].strip()
    return sentence





def evaluate_specific_token_comprehensive(
    target_token: str,
    client,
    variant,
    turing_model,
    sae,
    num_positive: int = 10,
    activation_threshold: float = 0.5
):
    positive_sentences = [generate_sentence(target_token, client, variant) for _ in range(num_positive)]
    
    print(f"Generated {num_positive} positive sentences.")
    
    all_sentences = positive_sentences
    labels = [1] * num_positive
    
    activations = []
    
    # '''The code of SAE and Turing would go here. The code will calculate activation values of every sentence generated for a particular token and append in activations list'''
    
    # '''Now if the activation value is too low, we wont take it into consideration . Only if the token gets activated for more than a certain value we say it is desirable'''
    predictions = [1 if act >= activation_threshold else 0 for act in activations]
    
  
    f1 = f1_score(labels, predictions)
    accuracy = (np.array(labels) == np.array(predictions)).mean()
        
    evaluation_results = {
        "target_token": target_token,
        "activation_threshold": activation_threshold,
        "f1_score": f1,
        "accuracy": accuracy
    }
    
    return evaluation_results





def evaluate_connecting_tokens(
    token_pair: List,
    latent_identifier: int,
    client,
    variant,
    turing_model,
    num_positive: int = 10,
    activation_threshold: float = 0.5
):
    token1 = token_pair[0]
    token2 = token_pair[1]
    
    positive_sentences = []
    for _ in range(num_positive//2):
        prompt = f"Generate a sentence that includes both {token1} and {token2}:"
        sentence = generate_sentence(prompt, client, variant)
        positive_sentences.append(sentence)
    
    print(f"Generated {num_positive} positive sentences for token pair {token_pair}.")
    
    all_sentences = positive_sentences
    labels = [1] * num_positive
    # '''Actually this means that both tokens activated and passed a threshold'''
    
    activations = []
    
    # '''The code of Turing and SAE would come here.  PLs chekck that both tokens get actiovated only then label of prediction will become 1'''
    
    # '''I dont think this code needs to change as the list will check for both values automatically whether they are greater than threshold or not'''
    predictions = [1 if act >= activation_threshold else 0 for act in activations]
    
    
    f1 = f1_score(labels, predictions)
    accuracy = (np.array(labels) == np.array(predictions)).mean()
    

    
    evaluation_results = {
        "token_pair": token_pair,
        "activation_threshold": activation_threshold,
        "f1_score": f1,
        "accuracy": accuracy
    }
    
    return evaluation_results





def run():
    
    print("")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    
    
    print("Loading Turing-LLM...")
    max_length = 64 + 1
    turing = TuringLLMForInference(collect_latents=True, max_length=max_length)
    
    sae_model = None
        
    
    for layer_index in range(TuringLLMConfig.n_layer):
        print("")
        print(f"Layer {layer_index+1}")
        print("  Loading SAE...")
        sae_model = SAE(TuringLLMConfig.n_embd, sae_dim, 128, only_encoder=True).to(device)
        sae_model = torch.compile(sae_model)
        sae_model.load(f"../../sparse_autoencoders/sae/sae_layer_{layer_index}.pth")
        if layer_index == 0:
            sae_model.k = 128 + (4 * 16)
        else:
            sae_model.k = 128 + (layer_index * 16)
        
        for latent_index in range(turing_sae_subset_layer_latent_count):
            start_time = time.time()
            print(f"  Processing Latent {str(latent_index+1).zfill(len(str(turing_sae_subset_layer_latent_count)))}", end="\r")
            print(f"  Processing Latent {str(latent_index+1).zfill(len(str(turing_sae_subset_layer_latent_count)))}  |  Duration: {time.time()-start_time:.2f}s                                                    ")
    
        del sae_model
        torch.cuda.empty_cache()
        gc.collect()
        print("")





run()
# prompt = "Please no introduction text, write just a short sentence which contains a word that has the exact text 'ure' in it."
# prompt = "Please no introduction text, write just a short sentence that has both 'ure' and 'agricult' in it."
# prompt = "Please no introduction text, write just a short sentence that has both 'event' and 'events' in it."
# res = generate_sentence(prompt)
# print(res.replace("\n", "\\n"))
