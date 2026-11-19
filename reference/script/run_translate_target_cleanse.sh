for translation_approach in azure_gpt_4o_mini azure_gpt_4o huggingface_llama_3_1_8b huggingface_qwen_2_5_coder huggingface_qwen_2_5_coder_7b
do
    for translation_language_pair in es2ko en2ko de2ko zh2ko
    do
        for context_size in 0 32 64 128 256 512
        do
            python translate_target_string/ray_version.py --translation_side_window $context_size --translation_approach $translation_approach --translation_language_pair $translation_language_pair --do_clean
        done
    done
done

for translation_approach in azure_gpt_4o_mini azure_gpt_4o huggingface_llama_3_1_8b huggingface_qwen_2_5_coder huggingface_qwen_2_5_coder_7b
do
    for translation_language_pair in en2ko es2ko de2ko zh2ko 
    do
        for context_size in 0 
        do
            python translate_target_string/ray_version_no_prompt_context.py --translation_side_window $context_size --translation_approach $translation_approach --translation_language_pair $translation_language_pair --do_clean
        done
    done
done