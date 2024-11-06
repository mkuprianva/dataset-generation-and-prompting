# LOADING THE LIBRARIES
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          TrainingArguments, 
                          Trainer, 
                          set_seed,
                          AutoConfig,
                          BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig, ModelConfig
from peft import LoraConfig, get_peft_model
import random
import datasets
import torch
import flash_attn_2_cuda
from torch import nn
import deepspeed
import time
from datetime import datetime
import wandb
from huggingface_hub import login as hf_login
from wandb import login as wb_login
import re
import argparse
import os
from typing import TypedDict
AI_MODEL_PATH='/workspaces/VA_work/AI_models/COT_Router'
DATASET_PATH = '/workspaces/VA_work/AI_models/external_repos/datasets'

#PARSE ARGS
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default=f'{AI_MODEL_PATH}/model/llama-3.2-3B/', help="Path to the local model or HF repo that you wish to use for fine tuning")
parser.add_argument('-l', '--local_model', default=True, action="store_true", help="Load only a local model")
parser.add_argument('-sl', '--max_seq_length', default=2048, type=int, help="Max sequence length of the model")
parser.add_argument('-d', "--dataset", type=str, default=f"{DATASET_PATH}/CoT-Collection/data/CoT_collection_en.json", help="Path of the json or csv file that you want to use for fine tuning")
parser.add_argument('-sf', '--source_field', default='source', type=str, help="Source field of the dataset")
parser.add_argument('-tf', '--target_field', default='target', type=str, help="Target field of the dataset")
parser.add_argument('-rf', '--rationale_field', default='rationale', type=str, help="Rationale field of the dataset")
parser.add_argument('-n', "--name", type=str, default='fine_tune_llama3-3B_COT', help="Name of the fine tuning run")
parser.add_argument('-hf',"--hf_token", type=str, default=f"{AI_MODEL_PATH}/HF_token", help="Path to Hugging Face token")
parser.add_argument('-wb',"--wandb_token", type=str, default=f"{AI_MODEL_PATH}/wandb_token", help="Path to W&B token")
parser.add_argument('-t', "--test_run", action='store_true', help="Run in testing mode with 100 items from dataset")
parser.add_argument('-ts', "--test_size", default=100, type=int, help="Number of items to use for testing")
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
# Include DeepSpeed configuration arguments
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()
if args.test_run:
    print("runing a test run with 100 samples of train dataset")
class COT_Finetuning():
    def __init__(self, args):
        self.args = args
        self.hf_dataset_test = None
        self.hf_dataset_train = None
        self.model = None
        self.tokenizer = None
#LOGIN TO HF AND WB
    def api_login(self,hf_token, wandb_token):
        if hf_token is None:
            hf_token = self.args.hf_token
        if wandb_token is None:
            wandb_token = self.args.wandb_token
        with open(hf_token, 'r') as f:
            HF_token = f.read()
            f.close()
        hf_login(HF_token)

        with open(args.wandb_token, 'r') as w:
            wandb_token = w.read()
            w.close()
        wb_login(key=wandb_token)

    #LOAD THE DATASET


    def json_file_parser(self,json_file, source, target, rationale, train_test_split=0.9):
        """
        Parses a JSON file to extract 'source', 'target', and 'rationale' fields, and constructs 
        textual prompts for AI model training. The function splits data into training and testing 
        sets based on the specified ratio.

        Args:
            json_file (str): Path to the JSON file containing data.
            source (list): List to store extracted 'source' data.
            target (list): List to store extracted 'target' data.
            rationale (list): List to store extracted 'rationale' data.
            train_test_split (float, optional): Ratio for splitting data into training and testing 
                sets. Defaults to 0.9.

        Returns:
            Dict: Dictionary containing training and testing data.
        """
        class ds_prepare(TypedDict):
            source: list[str]
            target: list[str]
            rationale: list[str]
            text: list[str]
        ds_dict: ds_prepare = {'source': [], 'target': [], 'rationale': [], 'text': []}
        ds_dict_train: ds_prepare = {'source': [], 'target': [], 'rationale': [], 'text': []}
        ds_dict_test: ds_prepare = {'source': [], 'target': [], 'rationale': [], 'text': []}
        counter = 0
        with open(json_file, 'r') as j:
            ds_dict = ds_dict_train
            for line in j:

                if f'"{source}": "' in line:
                    ds_dict['source'].append(re.sub(f'"{source}": "', '', line).replace('",', '').strip())
                if f'"{target}": "' in line:
                    ds_dict['target'].append(re.sub(f'"{target}": "', '', line).replace('",', '').strip())
                if f'"{rationale}": "' in line:
                    ds_dict['rationale'].append(re.sub(f'"{rationale}": "', '', line).replace('",', '').strip())
                    # print(re.sub('"source": "', '', line).replace('",', '').strip())

                if len(ds_dict['source']) == len(ds_dict['target']) == len(ds_dict['rationale']) and len(ds_dict['rationale'])>0: 
                    #We have build out the full dict and can now generate the text prompt:
                    # Set variables by grabbing the last array item of the prior dicts
                    # print(len(ds_dict['rationale']))
                    l_source = ds_dict['source'][-1]
                    l_target = ds_dict['target'][-1]
                    l_rationale = ds_dict['rationale'][-1]
                    prompt = f""" 
                    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                    You are a helpful AI assistant that utilizes reasoning to answer questions. For the following instructions please use only the content in the question. Please provide your reasoning and then the answer to the question
                    <|eot_id|><|start_header_id|>user<|end_header_id|>
                    [Question]: {l_source}
                    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                    [Reasoning]: {l_rationale}
                    [Answer]: {l_target}
                    <|end_of_text|>
                    """
                    if len(ds_dict_train['text'])> self.args.test_size and self.args.test_run:
                        print('testing set created')
                        break
                    if len(ds_dict['text']) == 0 or prompt != ds_dict['text'][-1]:
                        ds_dict['text'].append(prompt)
                        if random.random() < train_test_split:
                            ds_dict = ds_dict_train
                        else:
                            ds_dict = ds_dict_test    
            j.close()
        return {'train': ds_dict_train, 'test': ds_dict_test}

    def dataset_loader(self,dataset_path:str ='', source:str='', target:str='', rationale:str='', train_test_split=0.9):
        seed=2
        set_seed(seed)
        if dataset_path =='':
            dataset_path = self.args.dataset
        if source =='':
            source = self.args.source_field
        if target =='':
            target = self.args.target_field
        if rationale =='':
            rationale = self.args.rationale_field
        if dataset_path.endswith('.json'):
            parsed_data = self.json_file_parser(dataset_path, source=source, target=target, rationale=rationale, train_test_split=train_test_split)
        elif dataset_path.endswith('.csv'):
            pass
        elif dataset_path.endswith('.jsonl'):
            pass
        else:
            raise ValueError('The dataset must be a json or csv file')
        
        train_dataset = datasets.Dataset.from_dict(parsed_data['train'])
        test_dataset = datasets.Dataset.from_dict(parsed_data['test'])
        train_dataset = train_dataset.map(remove_columns=['source', 'target', 'rationale'], batched=True)
        test_dataset = test_dataset.map(remove_columns=['source', 'target', 'rationale'], batched=True)
        train_dataset = train_dataset.shuffle(seed=seed)
        self.hf_dataset_train = train_dataset
        self.hf_dataset_test = test_dataset
        


    #LOAD THE MODEL
    def load_model(self):
        class CastOutputToFloat(nn.Sequential):
            def forward(self, x):
                return super().forward(x).to(torch.float32)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        model_config = ModelConfig(model_name_or_path=self.args.model,
                                    attn_implementation="flash_attention_2",
                                    )
        if self.args.local_model:
            self.model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, 
                                                     local_files_only=True,
                                                     config=model_config,
                                                     quantization_config=bnb_config,
                                                     token=True
                                                     )
            self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path,
                                                      local_files_only=True,
                                                      additional_special_tokens=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
                                                      token=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, 
                                                     trust_remote_code=True,
                                                     config=model_config,
                                                     quantization_config=bnb_config,
                                                     token=True
                                                     )
            self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path,
                                                      trust_remote_code=True,
                                                      additional_special_tokens=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
                                                      token=True)
   
        # for param in self.model.parameters():
        #     param.requires_grad = False  # freeze the model - train adapters later
        #     if param.ndim == 1:
        #         # cast the small parameters (e.g. layernorm) to fp32 for stability
        #         param.data = param.data.to(torch.float32)

        self.model.gradient_checkpointing_enable()  # reduce number of stored activations
        self.model.enable_input_require_grads()
        self.model.use_cache = False
        self.model.lm_head = CastOutputToFloat(self.model.lm_head)
            
        config = LoraConfig(
            r=32,  #attention heads
            lora_alpha=64,  #alpha scaling
            target_modules=['gate_proj', 'k_proj', 'v_proj', 'down_proj', 'up_proj', 'o_proj', 'q_proj'],
            lora_dropout=0.1,  # dropout probability for layers
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False)
        self.model = get_peft_model(self.model, config)

        # Print Trainable parameters
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token

    def set_up_sft_train(self):
        """
        Sets up the training configuration for the SFTTrainer.

        Args:
        - model: The model to be fine-tuned.
        - args: The SFTConfig object containing the training configuration.
        - train_dataset: The dataset used for training.
        - eval_dataset: The dataset used for evaluation.
        - tokenizer: The tokenizer used to preprocess the dataset.
        - config: The SFTConfig object containing the training configuration.

        Returns:
        - None
        """
        self.sft_config = SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            bf16=True,
            packing=True,
            use_liger=True,
            overwrite_output_dir=True,
            max_seq_length=self.args.max_seq_length,
            neftune_noise_alpha=5,
            output_dir='checkpoints',
            dataset_text_field='text',
            # deepspeed='ds_llama3_COT.json',
            report_to="wandb",
            logging_steps=1,
            run_name=f"COT_llama3-3B_tuning_original_model-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            
        )
        self.trainer = SFTTrainer(
            model=self.model,
            args=self.sft_config,
            train_dataset=self.hf_dataset_train,
            eval_dataset=self.hf_dataset_test,
            tokenizer=self.tokenizer)
    def load_only(self):
        """
        Loads necessary components for model fine-tuning, including API login, dataset loading,
        model initialization, and training setup. Measures and prints the time taken for each step.

        Steps:
        - Logs into required APIs using provided tokens.
        - Loads the dataset and prepares it for training.
        - Initializes the model for fine-tuning.
        - Sets up the training configuration.

        Prints the time taken for each of the above steps.
        """
        api_start_time = time.perf_counter()
        self.api_login(self.args.hf_token, self.args.wandb_token)
        api_end_time = time.perf_counter()
        print(f"Time to login: {api_end_time - api_start_time:0.4f} seconds")
        dataset_load_start_time = time.perf_counter()
        self.dataset_loader()
        dataset_load_end_time = time.perf_counter()
        print(f"Time to load dataset: {dataset_load_end_time - dataset_load_start_time:0.4f} seconds")
        model_load_start_time = time.perf_counter()
        self.load_model()
        model_load_end_time = time.perf_counter()
        print(f"Time to load model: {model_load_end_time - model_load_start_time:0.4f} seconds")
        training_setup_start_time = time.perf_counter()
        self.set_up_sft_train()
        training_setup_end_time = time. perf_counter()
        print(f"Time to set up training: {training_setup_end_time - training_setup_start_time:0.4f} seconds")

    def load_and_run(self):
        self.load_only()
        with wandb.init(project="Llama-COT", job_type="train", # the project I am working on
            tags=['llama-3.2', 'COT'],
            notes =f"Fine tuning llama 3.2 with COT-Collection. CoT Prompt Instruction and QLora"):
                self.trainer.train()
        self.trainer.save_model('model/llama/COT')
#Testing and debugging
if __name__ == "__main__":
    # model = AutoModelForCausalLM.from_pretrained("model/llama-3.2-3B", local_files_only=True)
    j = '/workspaces/VA_work/AI_models/external_repos/datasets/CoT-Collection/data/CoT_collection_en.json'
    tic = time.perf_counter()
    m = COT_Finetuning(args=args)
    
    m.load_and_run()