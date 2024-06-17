import os
os.environ["CUDA_VISIBLE_DEVICES"] =  "2,3"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import transformers
import array
import zlib
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, MixtralForCausalLM, BitsAndBytesConfig, FalconForCausalLM
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch, PartialState
from huggingface_hub import snapshot_download
import sys
import traceback
import bitsandbytes 
import numpy as np
import constriction
from Arithmetic_Coder import AC_compress_file, AC_decompress_file

bnb_config = BitsAndBytesConfig( #optional configuration to run the llm on 4bit precision.
load_in_4bit=True,
bnb_4bit_use_double_quant=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.bfloat16
)
bnb_4bit_compute_dtype=torch.float16


class LLMZip:
    def __init__(self, model, method):
        self.model_name = model
        if self.model_name == "Mixtral":
            self.model = MixtralForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1", quantization_config = bnb_config, load_in_4bit=True , attn_implementation="flash_attention_2",  device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1") 
        elif self.model_name == "Yi":
            self.model = AutoModelForCausalLM.from_pretrained("01-ai/Yi-34B",quantization_config=bnb_config, attn_implementation="flash_attention_2", device_map="auto" ) #,,   quantization_config=bnb_config load_in_4bit=True ,trust_remote_code = True,
            self.tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-34B")
        # self.model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b", quantization_config=bnb_config,  device_map="auto")
        # self.tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")
        for param in self.model.parameters():
            param.requires_grad = False
        if torch.cuda.is_available():  
            dev = "cuda:0" 
        else:  
            dev = "cpu" 
        self.device = torch.device(dev) 
        self.context_size = 1024
        self.bsz = 128
        self.compression_method = method
        
    def save_to_csv(self, language, total_chars, token_length, char_token_ratio, entropy, compression_ratio, redundancy):
        csv_path = f'{self.model_name}_data_{self.compression_method}.csv'
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            if not file_exists:
                writer.writerow(['Language', 'Total Characters', 'Total Tokens', 'Characters/Tokens', 'Entropy', 'Compression Ratio', 'Redundancy'])

            writer.writerow([language, total_chars, token_length, char_token_ratio, entropy, compression_ratio, redundancy])
    
    def compress_data(self,  input_path, output_path):
        if self.compression_method == 'Ranks':
            return self.zip(input_path,output_path)
        elif self.compression_method == 'AC':
            return AC_compress_file(self.model,self.tokenizer, input_path,output_path, 5)
        else:
            raise ValueError("Unsupported compression method. Valid methods are 'Ranks' and 'AC'")
    
    def decompress_data(self,  input_path, output_path):
        if self.compression_method == 'Ranks':
            return self.unzip(input_path,output_path)
        elif self.compression_method == 'AC':
            return AC_decompress_file(self.model,self.tokenizer,input_path,output_path,5)
        else:
            raise ValueError("Unsupported compression method. Valid methods are 'Ranks' and 'AC'")
        
    def tokenize(self, text, language, path):  #takes as input the text in string form, returns a PyTorch Tensor containing the Token Ids. 
        print("Tokenizing...")
        tokens_filename = f"Tokens{language}Text{self.model_name}.txt"
        tokens_path = os.path.join(path, tokens_filename)
        if os.path.exists(tokens_path):
            print("Loading Tokens from file...")
            with open(tokens_path, 'r', encoding='utf-8') as file:
                token_str = file.read().strip()
                token_ids = list(map(int, token_str.split(',')))
                tokens = torch.tensor(token_ids)
        else:
            tokens = self.tokenizer(text, return_tensors="pt")
            tokens = tokens["input_ids"].squeeze()
        
        with open(tokens_path, 'w', encoding='utf-8') as file:
            file.write(','.join(map(str, tokens.tolist())))
            
        return tokens
        
    
    
    def pad(self, tokens, value): #pads the input so that it is a multiple of the context size.
        remainder = len(tokens) % self.context_size #calculate how much of from a multiple of ContextSize our input tokens are.
        if remainder>0:
            pad_amount = self.context_size - remainder
            print(pad_amount)
            padding = torch.tensor([value]*pad_amount)
            tokens = torch.cat((tokens, padding))
        else:
            pad_amount = 0

        return tokens, pad_amount
    
    def forward(self, tokens, index, past=None):
        with torch.no_grad():
            inputs = {}
            inputs['input_ids'] = tokens[:, index].reshape(-1, 1)

            output = self.model(**inputs, past_key_values=past)
            logits = output.logits 
            if len(logits.shape) > 2:
                logits = logits.reshape((logits.shape[0], -1))
            return logits, output.past_key_values
        
    def calculate_entropy(self, no_characters, probs):
            entropy = (torch.sum(-1*torch.log2(probs)).item())/no_characters
            relative_entropy = (len(probs)*(-1*np.log2(1 / len(self.tokenizer.get_vocab()))))/no_characters
            redundacy = 1 - (entropy/relative_entropy)
            return entropy, redundacy
        
    def zip(self,text_path, zip_path):
        with open(text_path, encoding="utf-8") as f:
            text = f.read()
        num_characters = len(text)
        language = os.path.splitext(os.path.basename(text_path))[0]
        print(language)
        directory_path = os.path.join("Text", language)
        
        tokens = self.tokenize(text, language, directory_path)
        num_tokens = len(tokens)
        char_per_token = f"{num_characters / num_tokens:.2f}"
            
        tokens, pad_amount = self.pad(tokens, self.tokenizer.eos_token_id)
        tokens = tokens.reshape(-1, self.context_size)
        
        
        ranks = torch.zeros(tokens.shape)
        probs = torch.zeros(tokens.shape)
        
        eos = torch.tensor([self.tokenizer.eos_token_id]*tokens.shape[0]).unsqueeze(1)
        tokens = torch.cat((eos, tokens), 1)
        tokens = tokens.to(self.device)
        batches = tokens.shape[0]//self.bsz
        if tokens.shape[0] % self.bsz != 0:
            batches += 1
        print("Getting Ranks...")
        for i in range(batches):
            batch = tokens[i*self.bsz:(i + 1)*self.bsz]
            curr_ranks = torch.zeros((batch.shape[0], batch.shape[1]-1))
            curr_probs = torch.zeros((batch.shape[0], batch.shape[1]-1))
            past = None
            print(i, "out of", batches)
            for j in range(batch.shape[1]-1):
                if j % 100 == 0:
                    print(j, "out of", batch.shape[1]-1)
                logits, past = self.forward(batch, j, past)
                logits, sorted_tokens = torch.sort(logits, descending=True)
                probabilities = F.softmax(logits, dim=-1)
                next_tokens = batch[:, j + 1]
                next_tokens_expanded = next_tokens.view(-1, 1).expand_as(sorted_tokens)
                rank_indices = (sorted_tokens==next_tokens_expanded).nonzero(as_tuple=True)
                rank_indices = rank_indices[1]
                curr_ranks[:, j] = rank_indices
                curr_probs[:, j] = probabilities.gather(1, rank_indices.view(-1, 1)).squeeze()
                
            ranks[i*self.bsz:(i + 1)*self.bsz] = curr_ranks
            probs[i*self.bsz:(i + 1)*self.bsz] = curr_probs
            
        ranks = ranks.flatten().int()
        
        probs = probs.flatten()
        probs = torch.where(probs == 0, probs + 0.001, probs)
        print("Probs are: ", probs)
        if pad_amount > 0: #remove padding ammount if there is any
            ranks = ranks[:-pad_amount]
            probs = probs[:-pad_amount]
        print("Ranks Shape: ", len(ranks) , " Probs shape: ")
        entropy, redundancy = self.calculate_entropy(num_characters, probs)
        
        with open(f"measurements{self.model_name}.txt", 'a') as file:
            file.write(f"The Entropy for {language} given by {self.model_name} is: {entropy}" + '\n')
            
        probs_filename = f"Probabilities{self.model_name}{language}.txt"
        probs_path = os.path.join(directory_path, probs_filename)
        with open(probs_path, 'w', encoding='utf-8') as file:
            file.write(','.join(map(str, probs.tolist())))
        
            
        ranks_list = array.array("H", ranks)
        zipped_ranks = zlib.compress(ranks_list, level=9)
        
        with open(zip_path, "wb") as file:
            file.write(zipped_ranks)
            print(f"Compression Complete! Saved file as {zip_path}")
        compression_ratio = os.path.getsize(zip_path)* 8/ num_characters   
        self.save_to_csv(language, num_characters, num_tokens, char_per_token, entropy, compression_ratio, redundancy )
        
    
    def unzip(self,zip_path,unzip_path):
        with open(zip_path, "rb") as file:
            zipped = file.read()
        ranks = zlib.decompress(zipped)
        ranks = array.array("H", ranks)
        ranks = torch.tensor(ranks)
        with torch.no_grad():
            ranks, pad_amount = self.pad(ranks, -999)
            ranks = ranks.reshape(-1, self.context_size)
            tokens = torch.zeros(ranks.shape, dtype=int)
            tokens = tokens.to(self.device)
            eos = torch.full((tokens.shape[0], 1), self.tokenizer.eos_token_id, dtype=tokens.dtype, device=self.device)
            # eos = torch.tensor([self.tokenizer.eos_token_id]*tokens.shape[0]).unsqueeze(1)
            tokens = torch.cat((eos, tokens), 1)
            batches = tokens.shape[0]//self.bsz
            if tokens.shape[0] % self.bsz != 0:
                batches += 1
            print("Getting Tokens...")
            for i in range(batches):
                print(i, "out of", batches)
                curr_ranks = ranks[i*self.bsz:(i + 1)*self.bsz] 
                batch = tokens[i*self.bsz:(i + 1)*self.bsz].to(self.device) 
                past = None
                for j in range(self.context_size):
                    if j % 100 == 0:
                        print(j, "out of", batch.shape[1]-1)
                    logits, past = self.forward(batch, j, past)
                    logits, sorted_tokens = torch.sort(logits, descending=True)
                    indices = curr_ranks[:, j].clone()
                    mask = indices == -999
                    valid_indices = torch.where(mask, torch.tensor(0, device=indices.device), indices)
                    decoded_tokens = sorted_tokens[torch.arange(indices.shape[0]), valid_indices]
                    decoded_tokens[mask] = self.tokenizer.eos_token_id
                    batch[:,j+1] = decoded_tokens.int()
                    # decoded_text = self.tokenizer.batch_decode(decoded_tokens.int(), skip_special_tokens=False, clean_up_tokenization_spaces=True)
                tokens[i*self.bsz:(i + 1)*self.bsz] = batch
                
            tokens = tokens[:, 1:].int()
            tokens = tokens.flatten()
            if pad_amount != 0:
                tokens = tokens[:-pad_amount]
            tokens = tokens.reshape((1, -1))
            text = self.tokenizer.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            text = "".join(text)
            
            with open(unzip_path, "w", encoding="utf-8") as f:
                f.write(text)
                print(f"Decompression Complete! Saved as {unzip_path}")
                
    def check(self, original_text, unzipped_text): #takes as input the original file and the unzipped version, and checks wether they are the same.
        try:
            with open(original_text, 'r', encoding='utf-8') as original, open(unzipped_text, 'r', encoding='utf-8') as unzipped:
                before = original.read()
                after = unzipped.read()
                if before == after:
                    print("Zipping and Unzipping was done Succesfully!")
                    return True
                else:
                    print("The two text files are not the same! There was an error in the process!")
                    return False
        except FileNotFoundError:
            print("One of the files was not found.")
            return False
        except IOError as e:
            print(f"An error occurred while reading the files: {e}")
            return False
        
def measure_time(start, end, language, model):
    time = end - start
    time = str(time)
    with open(f"Time_Measurements{model}.txt", 'a') as file:
        file.write(f"Time taken by {model} for {language} to zip is {time}.\n")
if __name__ == "__main__":
    
    # Mixtral = LLMZip("Mixtral")
    model = "Mixtral"
    LLMZip = LLMZip(model, "AC")
    try:
        
        ItalianPath = "Bible/Italian/Italian.txt"
        ItalianZipped = f"Bible/Italian/ItalianZipped{model}.gpz"
        start = time.time()
        LLMZip.compress_data(ItalianPath,ItalianZipped)
        end = time.time()
        measure_time(start,end, "Italian", model)
        
        # RussianPath = "Bible/Russian/Russian.txt"
        # RussianZipped = f"Bible/Russian/RussianZipped{model}.gpz"
        # start = time.time()
        # LLMZip.zip(RussianPath,RussianZipped)
        # end = time.time()
        # measure_time(start,end, "Russian", model)
        
        # JapanesePath = "Bible/Japanese/Japanese.txt"
        # JapaneseZipped = f"Bible/Japanese/JapaneseZipped{model}.gpz"
        # start = time.time()
        # LLMZip.zip(JapanesePath,JapaneseZipped)
        # end = time.time()
        # measure_time(start,end, "Japanese", model)

        
      
        
    except Exception as e:
        error_message = f"An error occurred: {e}\n"
        error_trace = traceback.format_exc()
        with open("errors.txt", "a") as error_file:
            error_file.write(error_message)
            error_file.write(error_trace)
        print(error_message, file=sys.stderr)
        print(error_trace, file=sys.stderr)
  
        # smalltxtPath = "smalltxt.txt"
        # smalltxtZippedPath = "smalltxtZipped.gpz"
        # smalltxtUnzippedPath = "smalltxtUnzipped.txt"
        # LLMZip.zip(smalltxtPath,smalltxtZippedPath)
        # LLMZip.unzip(smalltxtZippedPath, smalltxtUnzippedPath)
    
        # LLMZip.check(smalltxtPath, smalltxtUnzippedPath):
