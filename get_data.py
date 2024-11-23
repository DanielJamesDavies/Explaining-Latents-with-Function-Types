import time
import os
from dataclasses import dataclass
import numpy as np
import torch
import gc
import h5py
from transformers import AutoTokenizer





sae_dim = 40960

latents_path = "../../automated_interpretability/latent_data"





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
            
            
            
            
            
def collectDataForSpecificToken(layer_top_tokens_sorted, layer_top_values_sorted, folder_to_save, tokenizer):
    start_time = time.time()
    layer_top_tokens = None
    top_k_token = 6
    for latent_index in range(layer_top_tokens_sorted.shape[0]):
        layer_tokens_flat = layer_top_tokens_sorted[latent_index].clone().flatten()
        layer_values_flat = layer_top_values_sorted[latent_index].clone().flatten()
        layer_tokens_flat[layer_values_flat == 0] = -1
        
        unique_latent_tokens = torch.unique(layer_tokens_flat)
        unique_latent_tokens = unique_latent_tokens[unique_latent_tokens != -1]
        unique_latent_tokens_count = unique_latent_tokens.shape[0]
        if unique_latent_tokens_count < top_k_token:
            if layer_top_tokens is None:
                layer_top_tokens = torch.full((1, top_k_token), -1, device=device)
            else:
                layer_top_tokens = torch.cat((layer_top_tokens, torch.full((1, top_k_token), -1, device=device)), dim=0)
            continue
        
        mask = layer_tokens_flat != -1
        layer_tokens_flat_clean = layer_tokens_flat[mask]
        layer_values_flat_clean = layer_values_flat[mask]
        summed_values = torch.bincount(layer_tokens_flat_clean, weights=layer_values_flat_clean)
        unique_latent_values = summed_values[unique_latent_tokens]
        
        _, topk_indices = torch.topk(unique_latent_values, top_k_token)
        top_tokens = unique_latent_tokens[topk_indices].unsqueeze(0)
        if layer_top_tokens is None:
            layer_top_tokens = top_tokens
        else:
            layer_top_tokens = torch.cat((layer_top_tokens, top_tokens), dim=0)
        
        if (latent_index+1) % 1024 == 0:
            print(f"    Processed Latent {latent_index+1}  Duration: {time.time() - start_time:.2f}s", end="\r")
    print("")
    
    # Display Preview of Results
    print("    Preview of Results:")
    for i in range(8):
        print("      ", [tokenizer.decode([token]) for token in layer_top_tokens[i] if token != -1])
    
    # Save
    print("    Saving layer_top_tokens.h5")
    with h5py.File(f"{folder_to_save}/layer_top_tokens.h5", "w") as h5_file:
        h5_file.create_dataset("tensor", data=layer_top_tokens.cpu())
            
            
            
            
            
def collectDataForConnectingTokens(layer_top_tokens_sorted, layer_top_values_sorted, folder_to_save, tokenizer):
    start_time = time.time()
    layer_top_token_relationships = None
    top_k_token_relationships = 3
    for latent_index in range(layer_top_tokens_sorted.shape[0]):
        layer_tokens = layer_top_tokens_sorted[latent_index].clone()
        layer_values = layer_top_values_sorted[latent_index].clone()
        layer_tokens[layer_values == 0] = -1
        unique_latent_tokens = torch.unique(layer_tokens.flatten())
        unique_latent_tokens = unique_latent_tokens[unique_latent_tokens != -1]
        unique_latent_tokens_count = unique_latent_tokens.shape[0]
        if unique_latent_tokens_count < top_k_token_relationships:
            if layer_top_token_relationships is None:
                layer_top_token_relationships = torch.full((1, top_k_token_relationships, 2), -1, device=device)
            else:
                layer_top_token_relationships = torch.cat((layer_top_token_relationships, torch.full((1, top_k_token_relationships, 2), -1, device=device)), dim=0)
            continue
        
        frequency_matrix = torch.zeros((unique_latent_tokens_count, unique_latent_tokens_count), dtype=torch.int8, device=device)
        
        for i, sequence in enumerate(layer_tokens):
            if i < 16:
                token_mask = torch.isin(unique_latent_tokens, sequence)
                frequency_matrix += token_mask.unsqueeze(0) & token_mask.unsqueeze(1)
            
        frequency_matrix.fill_diagonal_(0)
        frequency_matrix = torch.triu(frequency_matrix, diagonal=1)
        
        frequency_matrix_flat = frequency_matrix.flatten()
        _, indices = torch.topk(frequency_matrix_flat, k=top_k_token_relationships)
        rows = indices // frequency_matrix.size(1)
        cols = indices % frequency_matrix.size(1)
        top_k_indices = torch.stack((rows, cols), dim=1)
        top_token_relationships = unique_latent_tokens[top_k_indices].unsqueeze(0)
        if layer_top_token_relationships is None:
            layer_top_token_relationships = top_token_relationships
        else:
            layer_top_token_relationships = torch.cat((layer_top_token_relationships, top_token_relationships), dim=0)
        
        if (latent_index+1) % 1024 == 0:
            print(f"    Processed Latent {latent_index+1}  Duration: {time.time() - start_time:.2f}s", end="\r")
    print("")
    
    # Display Preview of Results
    print(layer_top_token_relationships.shape)
    print("    Preview of Results:")
    for i in range(8):
        print("      ", [[tokenizer.decode([token]) for token in tokens if token != -1] for tokens in layer_top_token_relationships[i]])
    
    # Save
    print("    Saving top_token_relationships.h5")
    with h5py.File(f"{folder_to_save}/top_token_relationships.h5", "w") as h5_file:
        h5_file.create_dataset("tensor", data=layer_top_token_relationships.cpu())
        
       
        
        
        
def collectDataForDetectingDatasetTopic(layer_top_tokens_sorted, layer_top_values_sorted, folder_to_save, tokenizer, dataset_paths_data):
    start_time = time.time()
    for latent_index in range(layer_top_tokens_sorted.shape[0]):
        print("")
            
            
            
            

def run():
    
    # Setting Up
    print("")
    
    current_path = os.path.dirname(__file__)
    os.makedirs("./latent_data", exist_ok=True)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    
    
    # Check Files Exist
    latent_data_dir_list = os.listdir(latents_path)
    
    for name in ['latents_sae_frequencies.pth', 'latents_sae_tokens_from_sequence.h5', 'latents_sae_values_from_sequence.h5']:
        if name not in latent_data_dir_list:
            print(f"File Not Found: {name}")
            
            
            
    # Tokenizer       
    tokenizer_model_id = "microsoft/Phi-3-mini-4k-instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id, use_fast=True, local_files_only=True, _fast_init=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id, use_fast=True, _fast_init=True)
    
    
    
    # Run
    for layer_index in range(TuringLLMConfig.n_layer):
        start_time = time.time()
        folder_to_save = f"./latent_data/{layer_index}"
        os.makedirs(folder_to_save, exist_ok=True)
        
        print("")
        print(f"Processing Layer {str(layer_index+1).zfill(len(str(TuringLLMConfig.n_layer)))} / {TuringLLMConfig.n_layer}")
        
        
        
        # Get Data
        print("")
        print("  Getting Top Sequences Data...")
        with h5py.File(f"{latents_path}/latents_sae_tokens_from_sequence.h5", 'r') as h5f:
            layer_top_tokens = np.asarray(h5f['tensor'][layer_index, :, :, :])
            layer_top_dataset_paths = torch.tensor(layer_top_tokens[..., :1])
            layer_top_tokens = torch.tensor(layer_top_tokens[..., 1:]).to(device)
        with h5py.File(f"{latents_path}/latents_sae_values_from_sequence.h5", 'r') as h5f:
            layer_top_values = np.asarray(h5f['tensor'][layer_index, :, :, :])
            layer_top_values = torch.tensor(layer_top_values[..., 1:]).to(device)
        
        layer_top_tokens_dedup = layer_top_tokens.clone().to(device)
        layer_top_values_dedup = layer_top_values.clone().to(device)
        for i in range(1, layer_top_tokens_dedup.shape[-1]):
            mask = layer_top_tokens_dedup == layer_top_tokens_dedup[..., i].unsqueeze(-1)
            mask = mask & ~(torch.arange(layer_top_tokens_dedup.shape[-1], device=device).unsqueeze(0) >= i)
            
            # Add Duplicate Values
            layer_top_values_dedup += mask * layer_top_values_dedup[..., i].unsqueeze(-1).expand(-1, -1, layer_top_values_dedup.shape[-1])
            
            # Remove Duplicates
            row_mask = torch.any(mask, dim=-1, keepdim=True).expand(-1, -1, layer_top_tokens_dedup.size(-1)).to(device)
            layer_top_tokens_dedup[row_mask & (torch.arange(layer_top_tokens_dedup.size(-1), device=device) == i)] = -1
            layer_top_values_dedup[row_mask & (torch.arange(layer_top_values_dedup.size(-1), device=device) == i)] = 0
        
        layer_top_values_sorted, indices = torch.sort(layer_top_values_dedup, dim=-1, descending=True)
        layer_top_tokens_sorted = torch.gather(layer_top_tokens_dedup, dim=-1, index=indices)
        
        # Get Dataset Paths Data
        print("  Getting Dataset Paths...")
        with open(f'{latents_path}/dataset_paths.txt', 'r') as f:
            dataset_paths = "<|SPLIT|>".join(f.read().splitlines())
            
        def replaceDatasetPathsWithReadablePaths(curr_path, dataset_paths):
            print("    ", curr_path, "                         ", end="\r")
            path_contents = os.listdir(curr_path)
            path_contents = sorted(path_contents, key=lambda name: (".txt" in name, name))
            for name in path_contents:
                if name.split(".")[-1] == "txt":
                    if "items.txt" in path_contents:
                        with open(f'{curr_path}/items.txt', 'r', encoding='utf-8') as f:
                            items = [item.strip() for item in f.read().splitlines() if len(item.strip()) != 0]
                        for i in range(len(items)):
                            dataset_paths = dataset_paths.replace(curr_path[len("../../datasets/"):] + "/" + str(i), curr_path[len("../../datasets/"):] + "/" + str(items[i]))
                    elif "public_figures.txt" in path_contents:
                        with open(f'{curr_path}/public_figures.txt', 'r', encoding='utf-8') as f:
                            public_figures = [item.strip() for item in f.read().splitlines() if len(item.strip()) != 0]
                        for i in range(len(public_figures)):
                            dataset_paths = dataset_paths.replace(curr_path[len("../../datasets/"):] + "/" + str(i), curr_path[len("../../datasets/"):] + "/" + str(public_figures[i]))
                    elif "short_stories.txt" in path_contents:
                        with open(f'{curr_path}/short_stories.txt', 'r', encoding='utf-8') as f:
                            short_stories = [item.strip() for item in f.read().splitlines() if len(item.strip()) != 0]
                        for i in range(len(short_stories)):
                            dataset_paths = dataset_paths.replace(curr_path[len("../../datasets/"):] + "/" + str(i), curr_path[len("../../datasets/"):] + "/" + str(short_stories[i]))
                elif "." not in name:
                    dataset_paths = replaceDatasetPathsWithReadablePaths(curr_path + "/" + name, dataset_paths)
            return dataset_paths
            
        dataset_paths = replaceDatasetPathsWithReadablePaths("../../datasets/text", dataset_paths)
            
        print("  Getting Dataset Paths Data...                                                                                    ")
        dataset_paths_data = []
        for dataset_path in dataset_paths.split("<|SPLIT|>"):
            dataset_path = dataset_path[5:].split("/")[2:]
            dataset_path_list = [dataset_path[0]]
            for folder in dataset_path[1:]:
                if folder == "items" or folder.endswith(".txt"):
                    continue
                if not folder.isnumeric():
                    if folder not in dataset_path_list:
                        dataset_path_list.append(folder)
            dataset_paths_data.append(dataset_path_list)
        
        print(dataset_paths_data)
        
        # # Display Sorted Tokens
        # for i in range(12):
        #     print(tokenizer.batch_decode([token for token in layer_top_tokens_sorted[1][i] if token != -1]))
        #     print(layer_top_values_sorted[1][i])
        #     print("")
        #     print("")
        
        # For SpecificToken()
        print("")
        print("  For SpecificToken()")
        collectDataForSpecificToken(layer_top_tokens_sorted, layer_top_values_sorted, folder_to_save, tokenizer)
        
        # For ConnectingTokens()
        print("")
        print("  For ConnectingTokens()")
        collectDataForConnectingTokens(layer_top_tokens_sorted, layer_top_values_sorted, folder_to_save, tokenizer)
            
        # For SpecificWord()
        print("")
        print("  For SpecificWord()")
        
        # For DetectingDatasetTopic()
        print("")
        print("  For DetectingDatasetTopic()")
        collectDataForDetectingDatasetTopic(layer_top_tokens_sorted, layer_top_values_sorted, folder_to_save, tokenizer, dataset_paths_data)
        
        # For DetectingSpecificConcept()
        print("")
        print("  For DetectingSpecificConcept()")
        
        
        
        # Post Run
        duration = time.time() - start_time
        print("")
        print(f"Duration: {duration:.2f}s")
        print("")
        print("")
        


run()
