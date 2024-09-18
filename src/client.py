from dsp.modules.hf import HFModel
from openai import OpenAI
from inspect import signature
from loguru import logger


class OpenAIClientVLLM(HFModel):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model=model, is_client=True)
        self.client = OpenAI(*args, **kwargs)

    def _generate(self, prompt, **kwargs):
        create_params = signature(self.client.chat.completions.create).parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in create_params}
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": str(prompt)}], **filtered_kwargs
        )
        response = {
            "prompt": prompt,
            "choices": [
                {"text": choice.message.content} for choice in response.choices
            ],
        }
        return response
