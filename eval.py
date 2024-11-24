import time
import json
from dataclasses import dataclass
import torch
import gc
from transformers import AutoTokenizer
import warnings

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

warnings.filterwarnings("ignore")





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
    
    space_token = tokenizer.encode(" ")[0]
        
    for layer_index in range(TuringLLMConfig.n_layer):
        print(f"Layer {layer_index+1}")
        
        # Load SAE
        sae_model = SAE(TuringLLMConfig.n_embd, sae_dim, 128, only_encoder=True).to(device)
        sae_model = torch.compile(sae_model)
        sae_model.load(f"./SAE/sae/sae_layer_{layer_index}.pth")
        if layer_index == 0:
            sae_model.k = 128 + (4 * 16)
        else:
            sae_model.k = 128 + (layer_index * 16)
        
        # Load Data
        layer_eval_data = []
        with open(f'./latent_data/{layer_index}/eval_inputs.jsonl', 'r') as f:
            for i, line in enumerate(f):
                layer_eval_data.append(json.loads(line))
               
        # Tokenize Text
        for i, layer_eval_input in enumerate(layer_eval_data):
            layer_eval_input["text_tokens"] = tokenizer.encode(layer_eval_input["text"])[:64]
            if len(layer_eval_input["text_tokens"]) == 0:
                layer_eval_data[i] = None
                continue
            if len(layer_eval_input["text_tokens"]) < 64:
                while len(layer_eval_input["text_tokens"]) < 64:
                    layer_eval_input["text_tokens"] = layer_eval_input["text_tokens"] + layer_eval_input["text_tokens"]
                layer_eval_input["text_tokens"] = layer_eval_input["text_tokens"][:64]
            if layer_eval_input["type"] == "top-token":
                layer_eval_input["text_tokens"] = layer_eval_input["text_tokens"][:62] + [space_token] + [layer_eval_input["token"]]
            elif layer_eval_input["type"] == "connecting-tokens":
                layer_eval_input["text_tokens"] = layer_eval_input["text_tokens"][:60] + [space_token] + [layer_eval_input["tokens"][0]] + [space_token] + [layer_eval_input["tokens"][1]]
        layer_eval_data = [layer_eval_input for layer_eval_input in layer_eval_data if layer_eval_input is not None]
    
        num_activation_values_added = 0
        for batch_index in range(0, len(layer_eval_data), sequences_per_batch):
            batch_start_time = time.time()
            batch_layer_eval_data = layer_eval_data[batch_index:batch_index+sequences_per_batch]
            batch_length = len(batch_layer_eval_data)
            print(f"  Processing Latents {batch_index+1}-{batch_index+batch_length} / {len(layer_eval_data)}  |  Turing Inference        ", end="\r")
            with torch.no_grad():
                max_length = num_tokens_per_sequence + 1
                logits, latents = turing.generate_batch([input["text_tokens"] for input in batch_layer_eval_data], max_length=max_length, tokenize=False, decode=False, ignore_end=True)
    
            print(f"  Processing Latents {batch_index+1}-{batch_index+batch_length} / {len(layer_eval_data)}  |  SAE Inference           ", end="\r")
            with torch.no_grad():
                activations = [torch.split(layer_latents.view(-1, 1024), num_tokens_per_sequence * batch_length)[0] for layer_latents in latents[0]]
                x = activations[layer_index].to(device)
                sae_latents, _, _ = sae_model.encode(x)
                sae_latents = sae_latents.view(batch_length, -1, sae_dim).to(device)
                for i in range(batch_length):
                    if (i+batch_index) >= len(layer_eval_data):
                        continue
                    latent_index = layer_eval_data[i+batch_index]["latent_index"]
                    text_tokens = layer_eval_data[i+batch_index]["text_tokens"]
                    eval_data_type = layer_eval_data[i+batch_index]["type"]
                    
                    if eval_data_type == "top-token":
                        # Find token
                        token = layer_eval_data[i+batch_index]["token"]
                        if token not in text_tokens:
                            layer_eval_data[i+batch_index]["tokens_not_found"] = True
                            continue
                        token_index = text_tokens.index(token)
                        layer_eval_data[i+batch_index]["activation_value"] = sae_latents[i][token_index][latent_index].item()
                        
                    elif eval_data_type == "connecting-tokens":
                        # Find tokens and get average act
                        tokens = layer_eval_data[i+batch_index]["tokens"]
                        if tokens[0] not in text_tokens or tokens[1] not in text_tokens:
                            layer_eval_data[i+batch_index]["tokens_not_found"] = True
                            continue
                        token_0_index = text_tokens.index(tokens[0])
                        token_1_index = text_tokens.index(tokens[1])
                        activation_token_0 = sae_latents[i][token_0_index][latent_index].item()
                        activation_token_1 = sae_latents[i][token_1_index][latent_index].item()
                        layer_eval_data[i+batch_index]["activation_value"] = (activation_token_0 + activation_token_1) / 2
                        
                    elif eval_data_type == "detecting-dataset-topic":
                        # Get top value
                        layer_eval_data[i+batch_index]["activation_value"] = max(sae_latents[i][token_index][latent_index].item() for token_index in range(64))
                        
                    for token_sae_latents in sae_latents[i]:
                        turing_sae_latent_values[layer_index] += torch.tensor(token_sae_latents[0:subset_layer_latent_count], device="cpu")
                        num_activation_values_added += 1
            
            print(f"  Processing Latents {batch_index+1}-{batch_index+batch_length} / {len(layer_eval_data)}  |  Duration: {time.time()-batch_start_time:.2f}s                      ")
            
        # Get Average Latent Values
        turing_sae_latent_values[layer_index] = turing_sae_latent_values[layer_index] / num_activation_values_added
        
        # Process Eval Data
        for i, layer_eval_dict in enumerate(layer_eval_data):
            if "tokens_not_found" in layer_eval_dict and layer_eval_dict["tokens_not_found"] is True:
                continue
            latent_index = layer_eval_dict["latent_index"]
            latent_activation_value = layer_eval_dict["activation_value"]
            latent_avg_activation_value = turing_sae_latent_values[layer_index][latent_index].item()
            latent_activation_value_distance_from_avg = abs(latent_activation_value - latent_avg_activation_value)
            layer_eval_data[i]["activation_distance"] = latent_activation_value_distance_from_avg
            if latent_activation_value_distance_from_avg > 0.01:
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
