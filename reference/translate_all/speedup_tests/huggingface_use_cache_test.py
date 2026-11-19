from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteriaList, StoppingCriteria

import torch
from typing import List
import time
import copy
import tqdm
import configs.gen_params as gen_params

# "Qwen/Qwen2.5-Coder-32B-Instruct"
#"meta-llama/Llama-3.1-8B-Instruct"
def setup_model(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    # 모델과 토크나이저 초기화
    print("model_name:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # 메모리 효율을 위한 float16 사용
        device_map="auto"
    )
    return model, tokenizer

def generate_with_cached_prompt(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    base_prompt: str,
    prompt_suffixes: List[str],
    use_cache: bool = True
) -> List[str]:
    results = []
    
    # 기본 프롬프트 토큰화 및 캐시 생성
    base_inputs = tokenizer(base_prompt, return_tensors="pt")
    base_inputs = {k: v.to(model.device) for k, v in base_inputs.items()}
    
    with torch.no_grad():
        # 기본 프롬프트에 대한 캐시된 key/value 계산
        base_outputs = model(**base_inputs, use_cache=True)
        # layers, 2 (key, values), batch, heads, seq_len, dim
        cached_key_values = base_outputs.past_key_values

    cached_key_values = copy.deepcopy(cached_key_values)

    # 각 suffix에 대해 생성
    for suffix in tqdm.tqdm(prompt_suffixes):
        full_prompt = f"{base_prompt} {suffix}"
        inputs = tokenizer(full_prompt, return_tensors="pt")
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                past_key_values=cached_key_values if use_cache else None,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=use_cache,
                **gen_params.generate_args
            )
        
        # generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        # results.append(generated_text)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append(generated_text)
    
    return results

def generate_with_cached_prompt_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    base_prompt: str,
    prompt_suffixes: List[str],
    batch_size: int = 16,
    use_cache: bool = True,
    max_new_tokens: int = None
) -> List[str]:
    
    gen_args = copy.deepcopy(gen_params.generate_args)
    if max_new_tokens is not None:
        # assert batch_size == 1, "batch_size must be 1 when max_new_tokens is provided (This option is for seq 2 seq translation)"

        gen_args["max_new_tokens"] = max_new_tokens

    results = []
    
    messages = [{
        "role":"user",
        "content":base_prompt
    }]

    base_messages_in_chat_format = tokenizer.apply_chat_template(messages, tokenize=False)

    # 기본 프롬프트 토큰화 및 캐시 생성
    base_inputs = tokenizer(base_messages_in_chat_format, return_tensors="pt")
    base_inputs = {k: v.to(model.device) for k, v in base_inputs.items()}
    
    with torch.no_grad():
        # 기본 프롬프트에 대한 캐시된 key/value 계산
        base_outputs = model(**base_inputs, use_cache=True)
        # layers, 2 (key, values), batch, heads, seq_len, dim
        cached_key_values = base_outputs.past_key_values

    batch_cached_key_values = copy.deepcopy(cached_key_values)
    batch_cached_key_values = tuple([(l[0].repeat(batch_size, 1, 1, 1), l[1].repeat(batch_size, 1, 1, 1)) for l in batch_cached_key_values])

    all_messages = []
    all_prompts = []

    for suffix_tuple in prompt_suffixes:
        # print("suffix_tuple ==> ", list(suffix_tuple))
        tmp = copy.deepcopy(messages)
        tmp[0]["content"] += suffix_tuple[0]
        tmp += [{"role":"assistant", "content":suffix_tuple[1]}]
        all_messages.append(tmp)
        all_prompts.append(tokenizer.apply_chat_template(tmp, tokenize=False, continue_final_message=True))


    # 배치 단위로 처리
    for i in tqdm.tqdm(range(0, len(all_prompts), batch_size)):
        batch_prompts = all_prompts[i:i+batch_size]
        current_batch_size = len(batch_prompts)
        
        # 마지막 배치 처리
        if current_batch_size < batch_size:
            batch_cached_key_values = copy.deepcopy(cached_key_values)
            batch_cached_key_values = tuple([(l[0].repeat(current_batch_size, 1, 1, 1), l[1].repeat(current_batch_size, 1, 1, 1)) for l in batch_cached_key_values])

        batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}
        
        with torch.no_grad():
            batch_outputs = model.generate(
                **batch_inputs,
                past_key_values=batch_cached_key_values if use_cache else None,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=use_cache,
                **gen_args
            )
        
        input_lengths = batch_inputs['input_ids'].shape[1]
        # batch_results = [tokenizer.decode(outputs[input_lengths:], skip_special_tokens=True) 
        #                 for outputs in batch_outputs]
        # results.extend(batch_results)

        batch_results = [tokenizer.decode(outputs, skip_special_tokens=True) for outputs in batch_outputs]
        results.extend(batch_results)
    
    return results




class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
        self.has_stopped = None  # 각 배치의 누적 stop 상태를 저장

    def __call__(self, input_ids, scores, **kwargs):
        # 첫 호출시 has_stopped 초기화
        if self.has_stopped is None:
            self.has_stopped = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        
        for batch_idx in range(input_ids.shape[0]):
            for stop_ids in self.stop_token_ids:
                last_tokens = input_ids[batch_idx][-len(stop_ids):]
                if len(last_tokens) < len(stop_ids):
                    continue
                if torch.all(last_tokens == torch.tensor(stop_ids).to(input_ids.device)).item():
                    self.has_stopped[batch_idx] = True  # 누적 상태 업데이트
                    break
        # 모든 배치가 한번이라도 stop 조건을 만족했는지 확인
        return torch.all(self.has_stopped).item()
        
# cache is not possible with prefix prompts.
def generate_batch_ED(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    base_prompt: str,
    prompt_per_sample: List[str],
    batch_size: int = 16,
    stop_text: str = None
) -> List[str]:
    
    stopping_criteria = StoppingCriteriaList()
    stop_on_tokens = None
    if stop_text:
        # 중단하고 싶은 텍스트를 토큰화
        stop_token_ids = tokenizer(stop_text, add_special_tokens=False, return_tensors="pt")['input_ids']
        stop_token_ids = stop_token_ids.to(model.device)
        stop_on_tokens = StopOnTokens(stop_token_ids)
        stopping_criteria.append(stop_on_tokens)

    results = []
    all_prompts = []

    for pps in prompt_per_sample:
        # print("suffix_tuple ==> ", list(suffix_tuple))
        tmp = f"{pps} {stop_text} {base_prompt}"
        all_prompts.append(tmp)

    # 배치 단위로 처리
    for i in tqdm.tqdm(range(0, len(all_prompts), batch_size)):
        batch_prompts = all_prompts[i:i+batch_size]
        if stop_on_tokens:
            stop_on_tokens.has_stopped = None

        batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        batch_inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}
        
        with torch.no_grad():
            batch_outputs = model.generate(
                **batch_inputs,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
                **gen_params.generate_args
            )
        

        batch_results = [tokenizer.decode(outputs, skip_special_tokens=True) for outputs in batch_outputs]
        results.extend(batch_results)
    
    return results


def main():
    # 모델 및 토크나이저 설정
    model, tokenizer = setup_model()
    
    
    # 기본 프롬프트 설정
    base_prompt = " ".join(["안녕하세요 저는 기상 캐스터 임꺽정입니다."]*50) # very long base prompt
    
    # 날짜 리스트 생성
    dates = [f"11월 {day}일 날씨는" for day in range(1, 31)]
    
    # 시작 시간 기록
    start_time = time.time()
    
    # 날씨 예보 생성
    model.cuda()
    forecasts = generate_with_cached_prompt(model, tokenizer, base_prompt, dates, use_cache=True)
    print("[****] SAMPLEWISE use_cache=True")
    # 결과 출력
    # for date, forecast in zip(dates, forecasts):
    #     print(f"Date: {date}")
    #     print(f"Forecast: {forecast}\n")

    forecasts = generate_with_cached_prompt(model, tokenizer, base_prompt, dates, use_cache=False)
    print("[****] SAMPLEWISE use_cache=False")
    # 결과 출력
    # for date, forecast in zip(dates, forecasts):
    #     print(f"Date: {date}")
    #     print(f"Forecast: {forecast}\n")

    forecasts = generate_with_cached_prompt_batch(model, tokenizer, base_prompt, dates, use_cache=True)
    print("[****] BATCHWISE use_cache=True")
    # 결과 출력
    # for date, forecast in zip(dates, forecasts):
    #     print(f"Date: {date}")
    #     print(f"Forecast: {forecast}\n")

    forecasts = generate_with_cached_prompt_batch(model, tokenizer, base_prompt, dates, use_cache=False)
    print("[****] BATCHWISE use_cache=False")
    # 결과 출력
    # for date, forecast in zip(dates, forecasts):
    #     print(f"Date: {date}")
    #     print(f"Forecast: {forecast}\n")


    # 총 소요 시간 출력
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
