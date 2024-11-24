
from typing import List
import goodfire
import numpy as np
from sklearn.metrics import f1_score

import os
GOODFIRE_API_KEY = '''Put the API key from env file here'''
client = goodfire.Client(
    GOODFIRE_API_KEY
  )

def generate_sentence(token: str, client, variant, max_tokens: int = 100) -> str:
    variant = goodfire.Variant("meta-llama/Meta-Llama-3-8B-Instruct")
    prompt = f"Generate a coherent and diverse sentence that includes the word '{token}':"
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=variant,
        stream=False,
        max_tokens=max_tokens,
    )
    sentence = response.choices[0].message['content'].strip()
    print(f"Generated Sentence: {sentence}")
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
    
    '''The code of SAE and Turing would go here. The code will calculate activation values of every sentence generated for a particular token and append in activations list'''
    
    '''Now if the activation value is too low, we wont take it into consideration . Only if the token gets activated for more than a certain value we say it is desirable'''
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
    

    
    print(f"Generated {num_positive} positive and {num_positive} negative sentences for token pair {token_pair}.")
    
    
    all_sentences = positive_sentences
    labels = [1] * num_positive
    '''Actually this means that both tokens activated and passed a threshold'''
    
  
    activations = []
    
    '''The code of Turing and SAE would come here.  PLs chekck that both tokens get actiovated only then label of prediction will become 1'''
    
    '''I dont think this code needs to change as the list will check for both values automatically whether they are greater than threshold or not'''
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