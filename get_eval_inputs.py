import os
import time
import json
import h5py
from dotenv import load_dotenv
from dataclasses import dataclass
import goodfire
import torch
import gc
from transformers import AutoTokenizer





load_dotenv()
GOODFIRE_API_KEY = os.environ.get('GOODFIRE_API_KEY')
goodfire_client = goodfire.Client(GOODFIRE_API_KEY)
goodfire_base_model = "meta-llama/Meta-Llama-3-8B-Instruct"

subset_layer_latent_count = 82

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






def goodfire_generate(prompt: str = "", model: str = goodfire_base_model) -> str:
    response = goodfire_client.chat.completions.create(
        messages=[{ "role": "user", "content": prompt }],
        model=model,
        stream=False,
        max_completion_tokens=24,
    )
    return response.choices[0].message['content'].strip()





def getTopTokensSentences(latent_index, layer_top_tokens, top_dataset_topics, tokenizer):
    sentences = []
    latent_dataset_topics = [topic for topic in top_dataset_topics[latent_index] if topic != "subject"][2:6]
        
    variant = goodfire.Variant(goodfire_base_model)
    for topic in latent_dataset_topics:
        features, _ = goodfire_client.features.search(topic, model=variant, top_k=2)
        if features:
            for feature in features:
                variant.set(feature, 0.04)
    
    for i, token in enumerate(layer_top_tokens[latent_index][:2]):
        token = token.item()
        if token == -1:
            continue
        decoded_token = tokenizer.decode([token])
        prompt = f"Early on, use the subword appropriately: '{decoded_token}'. Please write \"Sentence:\" and then write just a short sentence. The sentence should have a word with '{decoded_token}' in it."
        
        res = goodfire_generate(prompt)
        if res.startswith("Sentence:"):
            res = res[len("Sentence:"):].strip()
        sentences.append({ "latent_index": latent_index, "type": "top-token", "token": token, "top_token_index": i, "is_variant": False, "text": res })
            
        res = goodfire_generate(prompt, variant)
        if res.startswith("Sentence:"):
            res = res[len("Sentence:"):].strip()
        sentences.append({ "latent_index": latent_index, "type": "top-token", "token": token, "top_token_index": i, "is_variant": True, "text": res })
        
    return sentences





def getConnectingTokensSentences(latent_index, top_token_relationships, tokenizer):
    sentences = []
    for i, tokens in enumerate(top_token_relationships[latent_index][:2]):
        tokens = tokens.tolist()
        if tokens[0] == -1 or tokens[1] == -1:
            continue
        decoded_tokens = tokenizer.batch_decode([[token] for token in tokens])
        prompt = f"Please write \"Sentence:\" and then write just a short sentence. Using common english, the sentence should have words with '{decoded_tokens[0]}' and '{decoded_tokens[1]}' in them."
        
        res = goodfire_generate(prompt)
        if res.startswith("Sentence:"):
            res = res[len("Sentence:"):].strip()
        sentences.append({ "latent_index": latent_index, "type": "connecting-tokens", "tokens": tokens, "connecting_tokens_index": i, "text": res })
    
    return sentences





def getDetectingDatasetTopicSentences(latent_index, top_dataset_topics, tokenizer):
    latent_dataset_topics = [topic for topic in top_dataset_topics[latent_index] if topic != "subject"][:6]
    prompt = f"One sentence. Please just write \"Sentence:\" and then write just a short single sentence surrounding these topics: {', '.join(latent_dataset_topics)}"
    
    res = goodfire_generate(prompt)
    if res.startswith("Sentence:"):
        res = res[len("Sentence:"):].strip()
        
    return [{ "latent_index": latent_index, "type": "detecting-dataset-topic", "topics": latent_dataset_topics, "text": res }]





def run():
    print("")
    print("Subset Layer Latent Count: ", subset_layer_latent_count)
    
    torch.cuda.empty_cache()
    gc.collect()
            
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
           
        # Load Layer Data 
        print("  Loading Layer Data...")
        with h5py.File(f'./latent_data/{layer_index}/layer_top_tokens.h5', 'r') as h5f:
            layer_top_tokens = torch.tensor(h5f['data']).to(device)
        
        with h5py.File(f'./latent_data/{layer_index}/top_token_relationships.h5', 'r') as h5f:
            top_token_relationships = torch.tensor(h5f['data']).to(device)
        
        top_dataset_topics = [None for _ in range(subset_layer_latent_count)]
        with open(f'./latent_data/{layer_index}/top_dataset_topics.jsonl', 'r') as f:
            for i, line in enumerate(f):
                if i < subset_layer_latent_count:
                    top_dataset_topics[i] = json.loads(line)
        
        # Get Sentences
        sentences = []
        for latent_index in range(subset_layer_latent_count):
            start_time = time.time()
            print(f"  Getting Sentences for Latent {str(latent_index+1).zfill(len(str(subset_layer_latent_count)))}", end="\r")
            sentences = sentences + getTopTokensSentences(latent_index, layer_top_tokens, top_dataset_topics, tokenizer)
            sentences = sentences + getConnectingTokensSentences(latent_index, top_token_relationships, tokenizer)
            sentences = sentences + getDetectingDatasetTopicSentences(latent_index, top_dataset_topics, tokenizer)
            print(f"  Acquired Sentences for Latent {str(latent_index+1).zfill(len(str(subset_layer_latent_count)))}  |  Duration: {time.time()-start_time:.2f}s")
    
        with open(f"./latent_data/{layer_index}/eval_inputs.jsonl", 'w') as f:
            for sentence_dict in sentences:
                f.write(json.dumps(sentence_dict) + '\n')
            
        print("")





run()
