NUMBER_OF_CACHE_BLOCKS=2
GPU_MEMORY_UTILIZATION=0.9
MAX_NUM_SEQS=32
for model in "qwen/Qwen2.5-Coder-32B-Instruct" #"qwen/Qwen2.5-Coder-7B-Instruct" "meta-llama/Llama-3.1-8B-Instruct" #"qwen/Qwen2.5-Coder-32B-Instruct"
do
    model_path_name=${model//\//_}
    for no_cache in False  True
    do
        for context_size_in_tags in 256 128 64 32 #128 64 32 16 8 1 
        do
            if [ "$no_cache" = "True" ]; then
                python translate_all/main_vllm.py --model-name $model --no-cache \
                                                                     --context-size-in-tags $context_size_in_tags \
                                                                     --max-num-seqs ${MAX_NUM_SEQS} \
                                                                     --num-parallel-cache-blocks ${NUMBER_OF_CACHE_BLOCKS} \
                                                                     --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
                                                                     --translate-all True 2>&1 | tee  ./results/qa_translated/log_${model_path_name}_${no_cache}_${NUMBER_OF_CACHE_BLOCKS}_${context_size_in_tags}.txt
            else
                python translate_all/main_vllm.py --model-name $model \
                                                                     --context-size-in-tags $context_size_in_tags \
                                                                     --max-num-seqs ${MAX_NUM_SEQS} \
                                                                     --num-parallel-cache-blocks ${NUMBER_OF_CACHE_BLOCKS} \
                                                                     --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
                                                                     --translate-all True 2>&1 | tee  ./results/qa_translated/log_${model_path_name}_${no_cache}_${NUMBER_OF_CACHE_BLOCKS}_${context_size_in_tags}.txt
            fi
        done
    done
done


# CUDA_VISIBLE_DEVICES="0" python translate_all/main_vllm.py --model-name "qwen/Qwen2.5-Coder-32B-Instruct" --no-cache --context-size-in-tags 128 --max-num-seqs 32 --num-parallel-cache-blocks 2 --gpu-memory-utilization 0.9 --translate-all True 2>&1 | tee  ./results/qa_translated/log_qwen_Qwen2.5-Coder-32B-Instruct_True_2_128.txt
# CUDA_VISIBLE_DEVICES="1" python translate_all/main_vllm.py --model-name "qwen/Qwen2.5-Coder-32B-Instruct" --no-cache --context-size-in-tags 64 --max-num-seqs 32 --num-parallel-cache-blocks 2 --gpu-memory-utilization 0.9 --translate-all True 2>&1 | tee  ./results/qa_translated/log_qwen_Qwen2.5-Coder-32B-Instruct_True_2_64.txt
# CUDA_VISIBLE_DEVICES="2" python translate_all/main_vllm.py --model-name "qwen/Qwen2.5-Coder-32B-Instruct" --no-cache --context-size-in-tags 32 --max-num-seqs 32 --num-parallel-cache-blocks 2 --gpu-memory-utilization 0.9 --translate-all True 2>&1 | tee  ./results/qa_translated/log_qwen_Qwen2.5-Coder-32B-Instruct_True_2_32.txt