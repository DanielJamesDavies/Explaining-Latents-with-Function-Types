import time
import os
from dataclasses import dataclass
import numpy as np
import torch
import gc
import h5py
import json
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt




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
    
    
    
    

def getDatasetPathsData():
    with open(f'{latents_path}/dataset_paths.txt', 'r') as f:
        dataset_paths = "<|SPLIT|>".join(f.read().splitlines())
    
    dataset_path_names = {}
    def getDatasetPathNames(curr_path, dataset_path_names):
        path_contents = os.listdir(curr_path)
        path_contents = sorted(path_contents, key=lambda name: (".txt" in name, name))
        for name in path_contents:
            if name.split(".")[-1] == "txt":
                if "items.txt" in path_contents:
                    with open(f'{curr_path}/items.txt', 'r', encoding='utf-8') as f:
                        items = [item.strip() for item in f.read().splitlines() if len(item.strip()) != 0]
                    curr_path_excluding_front = "/".join(curr_path.split("/")[6:])
                    for i in range(len(items)):
                        dataset_path_names[str(curr_path_excluding_front + "/items/" + str(i))] = str(items[i])
                elif "public_figures.txt" in path_contents:
                    with open(f'{curr_path}/public_figures.txt', 'r', encoding='utf-8') as f:
                        public_figures = [item.strip() for item in f.read().splitlines() if len(item.strip()) != 0]
                    curr_path_excluding_front = "/".join(curr_path.split("/")[6:])
                    for i in range(len(public_figures)):
                        dataset_path_names[str(curr_path_excluding_front + "/public_figures/" + str(i))] = str(public_figures[i])
                elif "short_stories.txt" in path_contents:
                    with open(f'{curr_path}/short_stories.txt', 'r', encoding='utf-8') as f:
                        short_stories = [item.strip() for item in f.read().splitlines() if len(item.strip()) != 0]
                    curr_path_excluding_front = "/".join(curr_path.split("/")[6:])
                    for i in range(len(short_stories)):
                        dataset_path_names[str(curr_path_excluding_front + "/items/" + str(i))] = str(short_stories[i])
                        dataset_path_names[str(curr_path_excluding_front + "/short_stories/" + str(i))] = str(short_stories[i])
            elif "." not in name:
                dataset_path_names = getDatasetPathNames(curr_path + "/" + name, dataset_path_names)
        return dataset_path_names
    dataset_path_names = getDatasetPathNames("../../datasets/text", dataset_path_names)
        
    dataset_paths_data = [False for _ in range(len(dataset_paths.split("<|SPLIT|>")))]
    for i, dataset_path in enumerate(dataset_paths.split("<|SPLIT|>")):
        dataset_path = dataset_path[5:].split("/")[2:]
        new_dataset_path = dataset_path[0]
        for j, folder in enumerate(dataset_path[1:]):
            if str("/".join(dataset_path[:j+1]) + "/" + folder) in dataset_path_names:
                new_dataset_path += "/" + dataset_path_names[str("/".join(dataset_path[:j+1]) + "/" + folder)]
            else:
                new_dataset_path += "/" + folder
        dataset_path_list = [dataset_path[0]]
        for folder in new_dataset_path.split("/")[1:]:
            if folder == "items" or folder.endswith(".txt") or folder.isnumeric():
                continue
            if folder[-1].isdigit():
                folder = folder[:-1]
            if folder not in dataset_path_list:
                dataset_path_list.append(folder)
        dataset_paths_data[i] = dataset_path_list
        
    return dataset_paths_data
            
            
            
            
            
            
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
        
        if (latent_index+1) % 4096 == 0:
            print(f"    Processed Latent {latent_index+1}  Duration: {time.time() - start_time:.2f}s", end="\r")
    print("")
    
    # Display Preview of Results
    print("    Preview of Results:")
    for i in range(8):
        print("      ", [tokenizer.decode([token]) for token in layer_top_tokens[i] if token != -1])
    
    # Save
    print("    Saving layer_top_tokens.h5")
    with h5py.File(f"{folder_to_save}/layer_top_tokens.h5", "w") as h5_file:
        h5_file.create_dataset("data", data=layer_top_tokens.cpu())
            
            
            
            
            
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
            if i < 24:
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
        
        if (latent_index+1) % 4096 == 0:
            print(f"    Processed Latent {latent_index+1}  Duration: {time.time() - start_time:.2f}s", end="\r")
    print("")
    
    # Display Preview of Results
    print("    Preview of Results:")
    for i in range(8):
        print("      ", [[tokenizer.decode([token]) for token in tokens if token != -1] for tokens in layer_top_token_relationships[i]])
    
    # Save
    print("    Saving top_token_relationships.h5")
    with h5py.File(f"{folder_to_save}/top_token_relationships.h5", "w") as h5_file:
        h5_file.create_dataset("data", data=layer_top_token_relationships.cpu())
        
       
        
        
        
def collectDataForDetectingDatasetTopic(layer_top_dataset_paths, folder_to_save, tokenizer, dataset_paths_data):
    start_time = time.time()
    
    text_encoder_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    text_encoder_model = text_encoder_model.to(device)
    
    top_dataset_topics = [["None"] for _ in range(layer_top_dataset_paths.shape[0])]    
    for latent_index in range(layer_top_dataset_paths.shape[0]):
        topics_frequencies = {}
        for key in [item for index in layer_top_dataset_paths[latent_index] for item in dataset_paths_data[index]]:
            if key not in topics_frequencies:
                topics_frequencies[key] = 1
            else:
                topics_frequencies[key] += 1
        topics_sorted = sorted(topics_frequencies, key=lambda x: topics_frequencies[x], reverse=True)
        # print("topics_sorted", topics_sorted)
        # print("")
        sequence_embeddings = text_encoder_model.encode(topics_sorted, device=device)
        embeddings_array = np.array(sequence_embeddings)
        linkage_matrix = sch.linkage(embeddings_array, method='ward')
        # plt.figure(figsize=(10, 7))
        # dendrogram = sch.dendrogram(linkage_matrix, labels=topics_sorted)
        
        cluster_labels = sch.fcluster(linkage_matrix, t=2, criterion='maxclust')
        cohesion_scores = {}
        for cluster_id in np.unique(cluster_labels):
            cluster_points = embeddings_array[np.where(cluster_labels == cluster_id)[0]]
            if len(cluster_points) > 1:
                pairwise_distances = pdist(cluster_points)
                cohesion_scores[cluster_id] = np.mean(pairwise_distances)
            else:
                cohesion_scores[cluster_id] = 0
        most_cohesive_cluster_id = min(cohesion_scores, key=cohesion_scores.get)
        most_cohesive_cluster_indices = np.where(cluster_labels == most_cohesive_cluster_id)[0]
        most_cohesive_cluster = [topics_sorted[i] for i in most_cohesive_cluster_indices][:16]
        top_dataset_topics[latent_index] = [str(topic) for topic in most_cohesive_cluster]

        # print("Top Dataset Topics:", most_cohesive_cluster)
                    
        # plt.title('Hierarchical Clustering Dendrogram')
        # plt.xlabel('Embeddings')
        # plt.xticks(rotation=45, ha='right')
        # plt.ylabel('Distance')
        # plt.show()
        
        if (latent_index+1) % 4096 == 0:
            print(f"    Processed Latent {latent_index+1}  Duration: {time.time() - start_time:.2f}s", end="\r")
    
    print("    Preview of Results:")
    for i in range(8):
        print("      ", top_dataset_topics[i][:7])
    
    with open(f"{folder_to_save}/top_dataset_topics.h5.jsonl", 'w') as f:
        for topics in top_dataset_topics:
            f.write(json.dumps(topics) + '\n')
            
            
            

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
    print("")
        
        
        
    # Get Dataset Paths Data
    print("Getting Dataset Paths Data...")
    dataset_paths_data = getDatasetPathsData()
    
    
    
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
            layer_top_dataset_paths = torch.tensor(layer_top_tokens[..., 0])
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
        
        # For DetectingDatasetTopic()
        print("")
        print("  For DetectingDatasetTopic()")
        collectDataForDetectingDatasetTopic(layer_top_dataset_paths, folder_to_save, tokenizer, dataset_paths_data)
            
        # For SpecificWord()
        print("")
        print("  For SpecificWord()")
        
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
