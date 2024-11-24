import time
import json
from dataclasses import dataclass
import torch
import gc
from transformers import AutoTokenizer

from TuringLLM.inference import TuringLLMForInference
from SAE.SAE_TopK import SAE





subset_layer_latent_count = 82
num_tokens_per_sequence = 64
sequences_per_batch = 256

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

turing_sae_latent_values = torch.zeros((12, subset_layer_latent_count))





def run():
    print("")
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print("Loading Turing-LLM...")
    max_length = 64 + 1
    turing = TuringLLMForInference(collect_latents=True, max_length=max_length)
            
    # Tokenizer       
    tokenizer_model_id = "microsoft/Phi-3-mini-4k-instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id, use_fast=True, local_files_only=True, _fast_init=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id, use_fast=True, _fast_init=True)
    print("")
        
    for layer_index in range(TuringLLMConfig.n_layer):
        print("")
        print(f"Layer {layer_index+1}")
        
        # Load SAE
        print("  Loading SAE...")
        sae_model = SAE(TuringLLMConfig.n_embd, sae_dim, 128, only_encoder=True).to(device)
        sae_model = torch.compile(sae_model)
        sae_model.load(f"./SAE/sae/sae_layer_{layer_index}.pth")
        if layer_index == 0:
            sae_model.k = 128 + (4 * 16)
        else:
            sae_model.k = 128 + (layer_index * 16)
        
        # Load Data
        layer_eval_data = [None for _ in range(subset_layer_latent_count)]
        with open(f'./latent_data/{layer_index}/eval_inputs.jsonl', 'r') as f:
            for i, line in enumerate(f):
                if i < subset_layer_latent_count:
                    layer_eval_data[i] = json.loads(line)
               
        # Tokenize Text
        for i, layer_eval_input in enumerate(layer_eval_data):
            layer_eval_input["text_tokens"] = tokenizer.encode(layer_eval_input["text"])[:64]
            if len(layer_eval_input["text_tokens"]) == 0:
                print("Error. Not Tokens for input: ", layer_eval_input)
                layer_eval_data[i] = None
                continue
            if len(layer_eval_input["text_tokens"] < 64):
                while len(layer_eval_input["text_tokens"]) < 64:
                    layer_eval_input["text_tokens"] = layer_eval_input["text_tokens"] + layer_eval_input["text_tokens"]
                layer_eval_input["text_tokens"] = layer_eval_input["text_tokens"][:64]
        layer_eval_data = [layer_eval_input for layer_eval_input in layer_eval_data if layer_eval_input is not None]
    
        layer_num_sequences_run = 0
        for batch_index in range(0, len(layer_eval_data), sequences_per_batch):
            batch_start_time = time.time()
            batch_layer_eval_data = layer_eval_data[batch_index:batch_index+sequences_per_batch]
            print(f"Processing Latents {batch_index+1}-{batch_index+sequences_per_batch+1} / {len(layer_eval_data)} ({((batch_index+1)/len(layer_eval_data))*100:.2f}%)  |  Turing Inference", end="\r")
            with torch.no_grad():
                max_length = num_tokens_per_sequence + 1
                logits, latents = turing.generate_batch([input["text_tokens"] for input in batch_layer_eval_data], max_length=max_length, tokenize=False, decode=False, ignore_end=True)
    
            print(f"Processing Latents {batch_index+1}-{batch_index+sequences_per_batch+1} / {len(layer_eval_data)} ({((batch_index+1)/len(layer_eval_data))*100:.2f}%)  |  SAE Inference", end="\r")
            with torch.no_grad():
                activations = [torch.split(layer_latents.view(-1, 1024), num_tokens_per_sequence * sequences_per_batch)[0] for layer_latents in latents[0]]
                x = activations[layer_index].to(device)
                sae_latents, _, _ = sae_model.encode(x)
                sae_latents = sae_latents.view(sequences_per_batch, -1, sae_dim).to(device)
                print(sae_latents.shape)
                for i in range(sequences_per_batch):
                    latent_index = layer_eval_data[i+batch_index]["latent_index"]
                    layer_eval_data[i+batch_index]["activation_value"] = sae_latents[latent_index].item()
                    turing_sae_latent_values[layer_index] += torch.tensor(sae_latents[0:subset_layer_latent_count], device="cpu")
                    layer_num_sequences_run += 1
            
            print(f"Processing Latents {batch_index+1}-{batch_index+sequences_per_batch+1} / {len(layer_eval_data)} ({((batch_index+1)/len(layer_eval_data))*100:.2f}%)  |  Duration: {time.time()-batch_start_time:.2f}s                      ")
            print(layer_eval_data[batch_index:batch_index+2])
            print("")
            
        # Get Average Latent Values
        turing_sae_latent_values[layer_index] = turing_sae_latent_values[layer_index] / layer_num_sequences_run
        
        # Process Eval Data
        for i, layer_eval_dict in enumerate(layer_eval_data):
            latent_index = layer_eval_dict["latent_index"]
            latent_activation_value = layer_eval_dict["activation_value"]
            latent_avg_activation_value = turing_sae_latent_values[layer_index][latent_index]
            latent_activation_value_distance_from_avg = latent_activation_value - latent_avg_activation_value
            layer_eval_data[i]["activation_distance"] = latent_activation_value_distance_from_avg
            if latent_activation_value_distance_from_avg > 0:
                layer_eval_data[i]["success"] = True
            else:
                layer_eval_data[i]["success"] = False
        
        # Save Eval Data
        with open(f"./latent_data/{layer_index}/eval_results.jsonl", 'w') as f:
            for layer_eval_dict in layer_eval_data:
                f.write(json.dumps(layer_eval_dict) + '\n')
    
        # Clear for Next Layer
        del sae_model
        torch.cuda.empty_cache()
        gc.collect()
        print("")





run()
