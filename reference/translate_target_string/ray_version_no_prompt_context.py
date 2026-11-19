import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import requests
from bs4 import BeautifulSoup, Tag, NavigableString
import dotenv
dotenv.load_dotenv()

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_community.llms import HuggingFaceHub

from preprocess_data import clean, read_html_file
from common.process_html import inject_translate_tag, extract_translate_tag

from bs4 import BeautifulSoup

import ray
from typing import Optional
import argparse

import configs.gen_params as gen_params

DATA_PATH = "./data/survey/"
TRANSLATE_TAG= gen_params.TRANSLATE_TAG

llm = None
def groq_llama_3_1_70b_versatile(user_msg, html_str):
    global llm
    if llm is None:
        llm = ChatGroq(
            # api_key=os.getenv("GROQ_API_KEY"),
            # model="llama3-70b-8192",
            model="llama-3.1-70b-versatile",
            temperature=0.1
        )
    template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert translator."),
        ("human", user_msg),
    ])

    chain = template|llm
    
    response = chain.invoke({"html":html_str, "id":0})
  
    return response.content


huggingface_llm = None
def huggingface_llm_func(user_msg, html_str, model_name="meta-llama/Llama-3.1-8B-Instruct"):
    global huggingface_llm
    if huggingface_llm is None:
        huggingface_llm = {}  # Initialize empty dict if None
    print(model_name)
    if model_name not in huggingface_llm:
        # 모델과 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        huggingface_llm[model_name] = {
            "model": model,
            "tokenizer": tokenizer
        }
    
    # Get current model and tokenizer
    current_model = huggingface_llm[model_name]["model"]
    current_model.eval()
    with torch.no_grad():
        current_tokenizer = huggingface_llm[model_name]["tokenizer"]
        
        messages = [
            {"role": "system", "content": "You are an expert translator."},
            {"role": "user", "content": user_msg.format(html=html_str, id=0)}
        ]
        
        prompt = current_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = current_tokenizer(prompt, return_tensors="pt").to(current_model.device)
        outputs = current_model.generate(
            **inputs,
            **gen_params.generate_args
        )
        response = current_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the response after the last assistant message
        print("[**] response", response.split(f"<{TRANSLATE_TAG}_0>")[-1])
        # response = response.split(f"<{TRANSLATE_TAG}_0>")[-1].split(f"</{TRANSLATE_TAG}_0>")[0].strip()
        response = response.split(f"<{TRANSLATE_TAG}_0>")[-1].split(f"</")[0].strip()
        return response

import azure_gpt

def azure_llm_func(user_msg, html_str, model_name="gpt-4o-mini"):
    messages = [
        {"role": "system", "content": "You are an expert translator."},
        {"role": "user", "content": user_msg.format(html=html_str, id=0)}
    ]
    response = azure_gpt.get_response(messages, model_name=model_name)
    if type(response) == dict:
        response = f"{response['role']}:{response['content']}"
    else:
        response = response.choices[0].message.content
        print(f"{model_name} response:", response)
        if response is None:
            return "[None response]"
        response = response.split(f"<{TRANSLATE_TAG}_0>")[-1].split(f"</")[0].strip()
        return response

from vllm import LLM, SamplingParams
vllm_llm = None
def vllm_llm_func(user_msg, html_str, model_name="meta-llama/Llama-3.1-8B-Instruct",
                  gpu_memory_utilization=0.5, enforce_eager=False,
                  max_num_seqs=40, enable_prefix_caching=True, tokenizer_path=None, max_model_len=None):
    global vllm_llm

    sampling_params = SamplingParams(temperature=gen_params.generate_args["temperature"], max_tokens=gen_params.generate_args["max_new_tokens"])

    if vllm_llm is None:
        vllm_llm = {}  # Initialize empty dict if None
    print(model_name)
    if model_name not in vllm_llm:
        # 모델과 토크나이저 로드
        llm_params = {
            "model": model_name,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
            "max_num_seqs": max_num_seqs,
            "enable_prefix_caching": enable_prefix_caching,
            "device": "cuda",  # 명시적으로 device 지정
            "tensor_parallel_size": 1  # GPU 병렬 처리 크기 지정
        }
        if max_model_len is not None:
            llm_params["max_model_len"] = max_model_len
            
        model = LLM(**llm_params)
    
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path if tokenizer_path else model_name)
        vllm_llm[model_name] = {
            "model": model,
            "tokenizer": tokenizer
        }
    
    # Get current model and tokenizer
    current_model = vllm_llm[model_name]["model"]

    with torch.no_grad():
        current_tokenizer = vllm_llm[model_name]["tokenizer"]
        
        messages = [
            {"role": "system", "content": "You are an expert translator."},
            {"role": "user", "content": user_msg.format(html=html_str, id=0)}
        ]
        
        prompt = current_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        outputs = current_model.generate(
            prompt,
            sampling_params
        )
        response = outputs[0].outputs[0].text
        
        # Extract the response after the last assistant message
        print("[**] response", response.split(f"<{TRANSLATE_TAG}_0>")[-1])
        # response = response.split(f"<{TRANSLATE_TAG}_0>")[-1].split(f"</{TRANSLATE_TAG}_0>")[0].strip()
        response = response.split(f"<{TRANSLATE_TAG}_0>")[-1].split(f"</")[0].strip()
        return response


azure_gpt_4o_mini = lambda user_msg, html_str: azure_llm_func(user_msg, html_str, model_name="gpt-4o-mini")
azure_gpt_4o = lambda user_msg, html_str: azure_llm_func(user_msg, html_str, model_name="gpt-4o")

huggingface_llama_3_1_8b = lambda user_msg, html_str: huggingface_llm_func(user_msg, html_str, model_name="meta-llama/Llama-3.1-8B-Instruct")
huggingface_llama_3_1_70b = lambda user_msg, html_str: huggingface_llm_func(user_msg, html_str, model_name="meta-llama/Llama-3.1-70B-Instruct")
huggingface_llama_3_3_70b = lambda user_msg, html_str: huggingface_llm_func(user_msg, html_str, model_name="meta-llama/Llama-3.3-70B-Instruct")
huggingface_qwen_2_5_coder = lambda user_msg, html_str: huggingface_llm_func(user_msg, html_str, model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
huggingface_qwen_2_5_coder_7b = lambda user_msg, html_str: huggingface_llm_func(user_msg, html_str, model_name="Qwen/Qwen2.5-Coder-7B-Instruct")
huggingface_phi_4 = lambda user_msg, html_str: huggingface_llm_func(user_msg, html_str, model_name="microsoft/phi-4")

vllm_llama_3_1_8b = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="meta-llama/Llama-3.1-8B-Instruct", enforce_eager=False, enable_prefix_caching=True)
vllm_llama_3_1_8b_w8a8 = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a8", enforce_eager=False, enable_prefix_caching=True)

vllm_qwen2_5_coder_7b = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="Qwen/Qwen2.5-Coder-7B-Instruct", enforce_eager=False, enable_prefix_caching=True)
vllm_qwen2_5_coder_32b = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="Qwen/Qwen2.5-Coder-32B-Instruct", enforce_eager=False, enable_prefix_caching=True, gpu_memory_utilization=0.9, max_num_seqs=10)

vllm_qwen2_5_7b = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="Qwen/Qwen2.5-7B-Instruct", enforce_eager=False, enable_prefix_caching=True)
vllm_qwen2_5_32b = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="Qwen/Qwen2.5-32B-Instruct", enforce_eager=False, enable_prefix_caching=True, gpu_memory_utilization=0.9, max_num_seqs=5, max_model_len=15872)


# For quantization experiments
huggingface_llama_3_1_8b_gptq8 = lambda user_msg, html_str: huggingface_llm_func(user_msg, html_str, model_name="iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8")
huggingface_llama_3_1_8b_gptq4 = lambda user_msg, html_str: huggingface_llm_func(user_msg, html_str, model_name="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4")
huggingface_llama_3_1_8b_awq4 = lambda user_msg, html_str: huggingface_llm_func(user_msg, html_str, model_name="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
huggingface_qwen2_5_coder_7b_gptq8 = lambda user_msg, html_str: huggingface_llm_func(user_msg, html_str, model_name="Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int8")
huggingface_qwen2_5_coder_7b_gptq4 = lambda user_msg, html_str: huggingface_llm_func(user_msg, html_str, model_name="Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4")
huggingface_qwen2_5_coder_7b_awq4 = lambda user_msg, html_str: huggingface_llm_func(user_msg, html_str, model_name="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ")
huggingface_qwen2_5_coder_32b_gptq8 = lambda user_msg, html_str: huggingface_llm_func(user_msg, html_str, model_name="Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int8")
huggingface_qwen2_5_coder_32b_gptq4 = lambda user_msg, html_str: huggingface_llm_func(user_msg, html_str, model_name="Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4")
huggingface_qwen2_5_coder_32b_awq4 = lambda user_msg, html_str: huggingface_llm_func(user_msg, html_str, model_name="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ")

vllm_llama_3_1_8b_q8_0 = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="./quantized_models/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf", enforce_eager=False, enable_prefix_caching=True, tokenizer_path="meta-llama/Llama-3.1-8B-Instruct")
vllm_llama_3_1_8b_q4_0 = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="./quantized_models/Meta-Llama-3.1-8B-Instruct.Q4_0.gguf", enforce_eager=False, enable_prefix_caching=True, tokenizer_path="meta-llama/Llama-3.1-8B-Instruct")
vllm_llama_3_1_8b_q2_k = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="./quantized_models/Meta-Llama-3.1-8B-Instruct.Q2_K.gguf", enforce_eager=False, enable_prefix_caching=True, tokenizer_path="meta-llama/Llama-3.1-8B-Instruct")
vllm_qwen2_5_coder_7b_q8_0 = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="./quantized_models/qwen2.5-coder-7b-instruct-q8_0.gguf", enforce_eager=False, enable_prefix_caching=True, tokenizer_path="Qwen/Qwen2.5-Coder-7B-Instruct")
vllm_qwen2_5_coder_7b_q4_0 = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="./quantized_models/qwen2.5-coder-7b-instruct-q4_0.gguf", enforce_eager=False, enable_prefix_caching=True, tokenizer_path="Qwen/Qwen2.5-Coder-7B-Instruct")
vllm_qwen2_5_coder_7b_q2_k = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="./quantized_models/qwen2.5-coder-7b-instruct-q2_k.gguf", enforce_eager=False, enable_prefix_caching=True, tokenizer_path="Qwen/Qwen2.5-Coder-7B-Instruct")
vllm_qwen2_5_coder_32b_q8_0 = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="./quantized_models/qwen2.5-coder-32b-instruct-q8_0.gguf", enforce_eager=False, enable_prefix_caching=True, gpu_memory_utilization=0.9, tokenizer_path="Qwen/Qwen2.5-Coder-32B-Instruct", max_num_seqs=10)
vllm_qwen2_5_coder_32b_q4_0 = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="./quantized_models/qwen2.5-coder-32b-instruct-q4_0.gguf", enforce_eager=False, enable_prefix_caching=True, gpu_memory_utilization=0.9, tokenizer_path="Qwen/Qwen2.5-Coder-32B-Instruct", max_num_seqs=10)
vllm_qwen2_5_coder_32b_q2_k = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="./quantized_models/qwen2.5-coder-32b-instruct-q2_k.gguf", enforce_eager=False, enable_prefix_caching=True, gpu_memory_utilization=0.9, tokenizer_path="Qwen/Qwen2.5-Coder-32B-Instruct", max_num_seqs=10)


vllm_llama_3_1_8b_awq4 = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4", enforce_eager=False, enable_prefix_caching=True, tokenizer_path="meta-llama/Llama-3.1-8B-Instruct")
vllm_llama_3_1_8b_gptq4 = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4", enforce_eager=False, enable_prefix_caching=True, tokenizer_path="meta-llama/Llama-3.1-8B-Instruct")
vllm_llama_3_1_8b_gptq8 = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="./quantized_models/Meta-Llama-3.1-8B-Instruct-GPTQ-INT8", enforce_eager=False, enable_prefix_caching=True, tokenizer_path="meta-llama/Llama-3.1-8B-Instruct")
vllm_qwen2_5_coder_7b_awq4 = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ", enforce_eager=False, enable_prefix_caching=True, tokenizer_path="Qwen/Qwen2.5-Coder-7B-Instruct")
vllm_qwen2_5_coder_7b_gptq4 = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4", enforce_eager=False, enable_prefix_caching=True, tokenizer_path="Qwen/Qwen2.5-Coder-7B-Instruct")
vllm_qwen2_5_coder_7b_gptq8 = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int8", enforce_eager=False, enable_prefix_caching=True, tokenizer_path="Qwen/Qwen2.5-Coder-7B-Instruct")
vllm_qwen2_5_coder_32b_awq4 = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ", enforce_eager=False, enable_prefix_caching=True, gpu_memory_utilization=0.9, tokenizer_path="Qwen/Qwen2.5-Coder-32B-Instruct", max_num_seqs=10)
vllm_qwen2_5_coder_32b_gptq4 = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4", enforce_eager=False, enable_prefix_caching=True, gpu_memory_utilization=0.9, tokenizer_path="Qwen/Qwen2.5-Coder-32B-Instruct", max_num_seqs=10)
vllm_qwen2_5_coder_32b_gptq8 = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int8", enforce_eager=False, enable_prefix_caching=True, gpu_memory_utilization=0.9, tokenizer_path="Qwen/Qwen2.5-Coder-32B-Instruct", max_num_seqs=10)



# No "coder" qwens
vllm_qwen2_5_7b_q8_0 = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="./quantized_models/qwen2.5-7b-instruct-q8_0.gguf", enforce_eager=False, enable_prefix_caching=True, tokenizer_path="Qwen/Qwen2.5-7B-Instruct")
vllm_qwen2_5_7b_q4_0 = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="./quantized_models/qwen2.5-7b-instruct-q4_0.gguf", enforce_eager=False, enable_prefix_caching=True, tokenizer_path="Qwen/Qwen2.5-7B-Instruct")
vllm_qwen2_5_7b_q2_k = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="./quantized_models/qwen2.5-7b-instruct-q2_k.gguf", enforce_eager=False, enable_prefix_caching=True, tokenizer_path="Qwen/Qwen2.5-7B-Instruct")
vllm_qwen2_5_32b_q8_0 = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="./quantized_models/qwen2.5-32b-instruct-q8_0.gguf", enforce_eager=False, enable_prefix_caching=True, gpu_memory_utilization=0.9, tokenizer_path="Qwen/Qwen2.5-32B-Instruct", max_num_seqs=10)
vllm_qwen2_5_32b_q4_0 = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="./quantized_models/qwen2.5-32b-instruct-q4_0.gguf", enforce_eager=False, enable_prefix_caching=True, gpu_memory_utilization=0.9, tokenizer_path="Qwen/Qwen2.5-32B-Instruct", max_num_seqs=10)
vllm_qwen2_5_32b_q2_k = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="./quantized_models/qwen2.5-32b-instruct-q2_k.gguf", enforce_eager=False, enable_prefix_caching=True, gpu_memory_utilization=0.9, tokenizer_path="Qwen/Qwen2.5-32B-Instruct", max_num_seqs=10)

vllm_qwen2_5_7b_awq4 = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="Qwen/Qwen2.5-7B-Instruct-AWQ", enforce_eager=False, enable_prefix_caching=True, tokenizer_path="Qwen/Qwen2.5-Coder-7B-Instruct")
vllm_qwen2_5_7b_gptq4 = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", enforce_eager=False, enable_prefix_caching=True, tokenizer_path="Qwen/Qwen2.5-Coder-7B-Instruct")
vllm_qwen2_5_7b_gptq8 = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8", enforce_eager=False, enable_prefix_caching=True, tokenizer_path="Qwen/Qwen2.5-Coder-7B-Instruct")
vllm_qwen2_5_32b_awq4 = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="Qwen/Qwen2.5-32B-Instruct-AWQ", enforce_eager=False, enable_prefix_caching=True, gpu_memory_utilization=0.9, tokenizer_path="Qwen/Qwen2.5-Coder-32B-Instruct", max_num_seqs=10)
vllm_qwen2_5_32b_gptq4 = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4", enforce_eager=False, enable_prefix_caching=True, gpu_memory_utilization=0.9, tokenizer_path="Qwen/Qwen2.5-Coder-32B-Instruct", max_num_seqs=10)
vllm_qwen2_5_32b_gptq8 = lambda user_msg, html_str: vllm_llm_func(user_msg, html_str, model_name="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8", enforce_eager=False, enable_prefix_caching=True, gpu_memory_utilization=0.9, tokenizer_path="Qwen/Qwen2.5-Coder-32B-Instruct", max_num_seqs=10)



translation_approaches = {
    "groq_llama_3_1_70b_versatile": groq_llama_3_1_70b_versatile,
    "huggingface_llama_3_1_8b": huggingface_llama_3_1_8b,
    "huggingface_llama_3_1_70b": huggingface_llama_3_1_70b,
    "huggingface_llama_3_3_70b": huggingface_llama_3_3_70b,
    "huggingface_qwen_2_5_coder": huggingface_qwen_2_5_coder,
    "huggingface_qwen_2_5_coder_7b": huggingface_qwen_2_5_coder_7b,
    "huggingface_phi_4": huggingface_phi_4,
    "vllm_llama_3_1_8b": vllm_llama_3_1_8b,
    "azure_gpt_4o_mini": azure_gpt_4o_mini,
    "azure_gpt_4o": azure_gpt_4o,

    "vllm_qwen2_5_7b_gptq8": vllm_qwen2_5_7b_gptq8,
    "vllm_qwen2_5_7b_gptq4": vllm_qwen2_5_7b_gptq4,
    "vllm_qwen2_5_7b_awq4": vllm_qwen2_5_7b_awq4,
    "vllm_qwen2_5_32b_gptq8": vllm_qwen2_5_32b_gptq8,
    "vllm_qwen2_5_32b_gptq4": vllm_qwen2_5_32b_gptq4,
    "vllm_qwen2_5_32b_awq4": vllm_qwen2_5_32b_awq4,


    "vllm_qwen2_5_coder_7b_gptq8": vllm_qwen2_5_coder_7b_gptq8,
    "vllm_qwen2_5_coder_7b_gptq4": vllm_qwen2_5_coder_7b_gptq4,
    "vllm_qwen2_5_coder_7b_awq4": vllm_qwen2_5_coder_7b_awq4,
    "vllm_qwen2_5_coder_32b_gptq8": vllm_qwen2_5_coder_32b_gptq8,
    "vllm_qwen2_5_coder_32b_gptq4": vllm_qwen2_5_coder_32b_gptq4,
    "vllm_qwen2_5_coder_32b_awq4": vllm_qwen2_5_coder_32b_awq4,
    "vllm_llama_3_1_8b_gptq8": vllm_llama_3_1_8b_gptq8,
    "vllm_llama_3_1_8b_gptq4": vllm_llama_3_1_8b_gptq4,
    "vllm_llama_3_1_8b_awq4": vllm_llama_3_1_8b_awq4,
    "vllm_qwen2_5_coder_7b": vllm_qwen2_5_coder_7b,
    "vllm_qwen2_5_coder_32b": vllm_qwen2_5_coder_32b,

    "huggingface_qwen2_5_coder_7b_gptq8": huggingface_qwen2_5_coder_7b_gptq8,
    "huggingface_qwen2_5_coder_7b_gptq4": huggingface_qwen2_5_coder_7b_gptq4,
    "huggingface_qwen2_5_coder_7b_awq4": huggingface_qwen2_5_coder_7b_awq4,
    "huggingface_qwen2_5_coder_32b_gptq8": huggingface_qwen2_5_coder_32b_gptq8,
    "huggingface_qwen2_5_coder_32b_gptq4": huggingface_qwen2_5_coder_32b_gptq4,
    "huggingface_qwen2_5_coder_32b_awq4": huggingface_qwen2_5_coder_32b_awq4,
    "huggingface_llama_3_1_8b_gptq8": huggingface_llama_3_1_8b_gptq8,
    "huggingface_llama_3_1_8b_gptq4": huggingface_llama_3_1_8b_gptq4,
    "huggingface_llama_3_1_8b_awq4": huggingface_llama_3_1_8b_awq4,

    "vllm_llama_3_1_8b_q8_0": vllm_llama_3_1_8b_q8_0,
    "vllm_llama_3_1_8b_q4_0": vllm_llama_3_1_8b_q4_0,
    "vllm_llama_3_1_8b_q2_k": vllm_llama_3_1_8b_q2_k,

    "vllm_qwen2_5_coder_7b_q8_0": vllm_qwen2_5_coder_7b_q8_0,
    "vllm_qwen2_5_coder_7b_q4_0": vllm_qwen2_5_coder_7b_q4_0,
    "vllm_qwen2_5_coder_7b_q2_k": vllm_qwen2_5_coder_7b_q2_k,

    "vllm_qwen2_5_coder_32b_q8_0": vllm_qwen2_5_coder_32b_q8_0,
    "vllm_qwen2_5_coder_32b_q4_0": vllm_qwen2_5_coder_32b_q4_0,
    "vllm_qwen2_5_coder_32b_q2_k": vllm_qwen2_5_coder_32b_q2_k,

    "vllm_qwen2_5_7b": vllm_qwen2_5_7b,
    "vllm_qwen2_5_7b_q8_0": vllm_qwen2_5_7b_q8_0,
    "vllm_qwen2_5_7b_q4_0": vllm_qwen2_5_7b_q4_0,
    "vllm_qwen2_5_7b_q2_k": vllm_qwen2_5_7b_q2_k,

    "vllm_qwen2_5_32b": vllm_qwen2_5_32b,
    "vllm_qwen2_5_32b_q8_0": vllm_qwen2_5_32b_q8_0,
    "vllm_qwen2_5_32b_q4_0": vllm_qwen2_5_32b_q4_0,
    "vllm_qwen2_5_32b_q2_k": vllm_qwen2_5_32b_q2_k,
    

    
}


@ray.remote(num_gpus=1)
class TranslatorActor:
    def __init__(self, translation_approach):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # 필요한 import 추가
        # 모델을 한 번만 초기화
        self.translation_approach = translation_approach
        self.model = translation_approaches[translation_approach]
        
        # GPU 가용성 확인 및 출력
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"Current GPU: {torch.cuda.current_device()}")
            print(f"GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        
    def process_item(self, item: dict, translation_window_left=512, translation_window_right=512, do_clean=True) -> tuple[str, Optional[str]]:
        path = item["file_original"]
        soup = read_html_file(path)
        if soup is None:
            return item["original"], None
            
        original_text = item["original"]
        translated_text = translate_target_string(clean(soup) if do_clean else soup, original_text, self.model, translation_window_left, translation_window_right)
        return original_text, translated_text



def translate_target_string(soup, target_string, invoke_function, translation_window_left=512, translation_window_right=512):
    html_str = inject_translate_tag(soup, target_string, f"{TRANSLATE_TAG}")
    if html_str is None:
        return ""
    
    html_str = extract_translate_tag(html_str, f"{TRANSLATE_TAG}_0", translation_window_left, translation_window_right)
    if html_str is None:
        return ""

    user_msg = f"""\
- The following is a text you need to translate.
####
{{html}}
####

#Instruction

- Find a text between <{TRANSLATE_TAG}_{{id}}> and </{TRANSLATE_TAG}_{{id}}> tags and translate it into Korean (한글).
- Do not add preambles.
- Return the translated text wrapped in the same tags.
  For example, your response should be like this:
    <{TRANSLATE_TAG}_0>comments</{TRANSLATE_TAG}_0> -> <{TRANSLATE_TAG}_0>댓글</{TRANSLATE_TAG}_0>
"""

    response = invoke_function(user_msg, html_str)

  


    return response



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--translation_approach", type=str, default="huggingface_llama_3_1_8b", choices=translation_approaches.keys(), required=False)
    parser.add_argument("--translation_side_window", type=int, default=None, required=False)
    parser.add_argument("--do_clean", type=bool, default=None, required=False)
    parser.add_argument("--translation_language_pair", type=str, default="en2ko", choices=["en2ko", "es2ko", "de2ko", "zh2ko"], required=False)

    args = parser.parse_args()
    translation_approach = args.translation_approach
    
    do_clean = args.do_clean if args.do_clean is not None else True
    WINDOW_SIZE = args.translation_side_window if args.translation_side_window is not None else 512
    # Ray 초기화 - 명시적으로 GPU 리소스 지정
    ray.init(num_gpus=8, runtime_env={
        "env_vars": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
        }
    })
    
    # GPU 가용성 확인
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    data = json.loads(open(f"{DATA_PATH}/preprocessed-survey-data/{args.translation_language_pair}/webpage_chrome_manual.json", "r", encoding="utf-8").read())
    
    # 여러 GPU를 사용하는 경우
    num_gpus = min(8, torch.cuda.device_count())  # 사용할 GPU 수 (최대 8개, 실제 가용 GPU 수에 따라 조정)
    print(f"Using {num_gpus} GPUs")
    
    # 각 TranslatorActor에 GPU 리소스를 명시적으로 할당
    translators = []
    for i in range(num_gpus):
        # 각 Actor에 특정 GPU를 할당
        actor = TranslatorActor.options(num_gpus=1).remote(translation_approach)
        translators.append(actor)
    
    # Round-robin 방식으로 작업 분배
    futures = []
    futures_to_index = {} 

    for i, item in enumerate(data):
        translator = translators[i % num_gpus]
        future = translator.process_item.remote(item, translation_window_left=WINDOW_SIZE, translation_window_right=WINDOW_SIZE, do_clean=do_clean)
        futures.append(future)
        futures_to_index[future] = i

    # 결과가 완료되는 대로 출력
    while futures:
        done_id, futures = ray.wait(futures)
        index = futures_to_index[done_id[0]]

        original_text, translated_text = ray.get(done_id[0])
        if translated_text is None:
            print(f"Failed to process: {original_text}")
        elif translated_text == "":
            print(original_text, "--->", "NOT FOUND")
        else:
            print(original_text, "--->", translated_text)
        
        if args.translation_side_window is not None:
            translation_approach_with_window = translation_approach+"_"+str(WINDOW_SIZE)
        else:
            translation_approach_with_window = translation_approach
        
        if args.do_clean is not None:
            translation_approach_with_clean = translation_approach_with_window + f"_clean_{args.do_clean}"
        else:
            translation_approach_with_clean = translation_approach_with_window
        
        translation_approach_with_clean = translation_approach_with_clean + "_no_prompt_context"

        data[index]['translation_approach'][translation_approach_with_clean] = translated_text
    
    with open(f"{DATA_PATH}/preprocessed-survey-data/{args.translation_language_pair}/webpage_chrome_manual.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
ray.shutdown()
        
        
