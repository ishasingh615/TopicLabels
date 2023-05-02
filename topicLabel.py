# install transformers
# install sentencepiece
# install accelerate
# install bitsandbytes

import torch
from transformers import GenerationConfig
from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM


class TopicLabels:
    def __init__(self, sentences):
        self.sentences = sentences
        self.instructionPrompt = "Determine one common topic of the following sentences: "
        self.tokenizer = LlamaTokenizer.from_pretrained("chainyo/alpaca-lora-7b")
        self.model = LlamaForCausalLM.from_pretrained(
            "chainyo/alpaca-lora-7b",
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.generation_config = GenerationConfig(
            temperature=0.2,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=128,
        )

    def generate_prompt(self, instruction: str, input_ctxt: str = None) -> str:
        if input_ctxt:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

      ### Instruction:
      {instruction}

      ### Input:
      {input_ctxt}

    ### Response:"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

  ### Response:"""

    def topic_label_inference(self):
        input_ctxt = None  # For some tasks, you can provide an input context to help the model generate a better response.

        instruction = self.instructionPrompt + self.sentences
        prompt = self.generate_prompt(instruction, input_ctxt)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )

        response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        print(response.split("is")[-1])



