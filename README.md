# LLM as a jailbreak judge

<a href="https://wandb.ai/romaingrx/llm-as-a-judge/reports/LLM-as-a-judge--Vmlldzo5MjcwNTMw">
<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Wandb" width="120"/>
</a>

> [!WARNING]  
> This research is conducted for educational and defensive purposes only, aiming to improve AI safety and security.

This repository contains the code which evaluates the performance of small open-source LMs as judges. Specifically, this evaluation centers on [Mistral Nemo 13B model](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407).


The code is divided into 3 main components:
1. [Basic judge](./src/basic_judge.py): A judge that uses a LM to judge the harmfulness of a given conversation in one try
2. [Prompt optimization](./src/prompt_optimization.py): A script that optimizes the judge's CoT and prompts to improve its performance in judging the harmfulness of a given conversation
3. [Multifaceted judge](./src/multifaceted_judge.py): A judge that uses a combination of metrics (harmfulness, relevance and informativeness) to judge the harmfulness of a given content

>[!INFO]
> The dataset used in this evaluation is the data produced during the [HarmBench evaluation](https://github.com/romaingrx/Harmbench) of the [Red Teamer Mistral Nemo](https://github.com/romaingrx/red-teamer-mistral-nemo) against [zephyr-7b](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta).

## Basic judge

In this example, I simply reused the prompt from [Patrick Chao's repository](https://github.com/patrickrchao/JailbreakingLLMs/blob/main/system_prompts.py) to assess the basic capabilities of the model to judge the harmfulness of a given conversation and compared its performance to GPT-4 equipped with the same prompt.

## Prompt optimization

For this technique, I leveraged DSPY's efficiency to prompt-tune my implementation with few-shot examples. To do so, I created a simple 2-layer module. The first layer is a chain-of-thought that asks the model to reason about the dialogue and go through the rules one by one, determining if they were matched or not. Then, in the second layer, it analyzes the reasoning and return True if any of the rules were matched.

## Multifaceted judge

For the final technique, I drew inspiration from recent papers [[1]](#1)[[2]](#2) approach. This method goes beyond simply assessing whether an answer adheres to ethical, legal, or safety guidelines. It also evaluates if the response provides relevant information and adequately addresses the initial prompt.

To implement this, I recreated the [Multifaceted Jailbreak Evaluation framework](https://github.com/controllability/jailbreak-evaluation) using DSPY only. This implementation includes modular components that can extract different levels of granularity from the texts. The approach considers multiple facets of the response, including:

1. Adherence to safety guidelines
2. Relevance to the initial prompt
3. Informativeness of the content
4. Potential for harmful implications

By evaluating these multiple aspects, we can gain a more comprehensive understanding of the model's output, helping to identify both overt and subtle forms of jailbreaking attempts.


## References

<a id="1">[1]</a> Souly, A., Lu, Q., Bowen, D., Trinh, T., Hsieh, E., Pandey, S., Abbeel, P., Svegliato, J., Emmons, S., Watkins, O., & Toyer, S. (2024). A StrongREJECT for Empty Jailbreaks. arXiv preprint arXiv:2402.10260. https://arxiv.org/abs/2402.10260

<a id="2">[2]</a> Cai, H., Arunasalam, A., Lin, L. Y., Bianchi, A., & Celik, Z. B. (2024). Rethinking How to Evaluate Language Model Jailbreak. arXiv preprint arXiv:2404.06407. https://arxiv.org/abs/2404.06407