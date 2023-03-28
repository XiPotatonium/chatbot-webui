from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor, nn

from transformers import (
    PreTrainedModel, PretrainedConfig, AutoModel, AutoModelForSeq2SeqLM,
    Blip2ForConditionalGeneration, Blip2VisionModel, Blip2QFormerModel,
    Blip2VisionConfig, Blip2QFormerConfig
)
from .modeling_chatglm import ChatGLMForConditionalGeneration, InvalidScoreLogitsProcessor
from .configuration_chatglm import ChatGLMConfig

import copy
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig


logger = logging.get_logger(__name__)



class Blip2ChatGLMConfig(PretrainedConfig):
    """Mainly based on Blip2Config

    Args:
        PretrainedConfig (_type_): _description_
    """
    is_composition = True

    def __init__(self, vision_config=None, qformer_config=None, text_config=None, num_query_tokens=32, **kwargs):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the Blip2VisionConfig with default values.")

        if qformer_config is None:
            qformer_config = {}
            logger.info("qformer_config is None. Initializing the Blip2QFormerConfig with default values.")

        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the text config with default values (`OPTConfig`).")

        self.vision_config = Blip2VisionConfig(**vision_config)
        self.qformer_config = Blip2QFormerConfig(**qformer_config)
        # text_model_type = text_config["model_type"] if "model_type" in text_config else "opt"
        # self.text_config = CONFIG_MAPPING[text_model_type](**text_config)
        self.text_config = ChatGLMConfig(**text_config)

        # self.tie_word_embeddings = self.text_config.tie_word_embeddings
        self.tie_word_embeddings = False                # I don't know what this is
        # self.is_encoder_decoder = self.text_config.is_encoder_decoder
        self.is_encoder_decoder = True                  # chatglm is an encoder-decoder model

        self.num_query_tokens = num_query_tokens
        self.qformer_config.encoder_hidden_size = self.vision_config.hidden_size
        # self.use_decoder_only_language_model = self.text_config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        self.use_decoder_only_language_model = False    # chatglm is an encoder-decoder model
        self.initializer_factor = 1.0
        self.initializer_range = 0.02

    @classmethod
    def from_vision_qformer_text_configs(
        cls,
        vision_config: Blip2VisionConfig,
        qformer_config: Blip2QFormerConfig,
        text_config: PretrainedConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`Blip2Config`] (or a derived class) from a BLIP-2 vision model, Q-Former and language model
        configurations.

        Returns:
            [`Blip2Config`]: An instance of a configuration object
        """

        return cls(
            vision_config=vision_config.to_dict(),
            qformer_config=qformer_config.to_dict(),
            text_config=text_config.to_dict(),
            **kwargs,
        )

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["qformer_config"] = self.qformer_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output


class Blip2ForChatGLM(PreTrainedModel):
    config_class = Blip2ChatGLMConfig

    def __init__(self, config: Blip2ChatGLMConfig):
        super().__init__(config)

        self.vision_model = Blip2VisionModel(config.vision_config)

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)


class Blip2ChatGLM(nn.Module):
    def __init__(self, blip2: Blip2ForChatGLM, lm: ChatGLMForConditionalGeneration) -> None:
        super().__init__()
        self.blip2 = blip2
        self.language = lm

    @torch.no_grad()
    def stream_chat(
        self,
        tokenizer,
        query: Union[str, Tuple[str, torch.Tensor]],
        history: List[Tuple[Union[str, Tuple[str, torch.Tensor]], str]]=[],
        num_beams=5,
        max_length=128,
        top_p=0.9,
        do_sample=True,
        temperature=1,
    ):
        device = self.blip2.device
        # 1. Prepare token ids
        images = []
        image_slots = []

        nvtokens = self.blip2.query_tokens.size(1)
        if history:
            input_ids = tokenizer(f"[Round {len(history)}]\n问：", add_special_tokens=False).input_ids
            slot_offset = len(input_ids)
            if isinstance(query, tuple):
                qtext, qimg = query
                # image slot, embedding will be replaced by image embeddings
                input_ids.extend([tokenizer.unk_token_id] * nvtokens)
            else:
                qtext = query
                qimg = None
            input_ids += tokenizer(qtext + f"\n答：").input_ids
            if qimg is not None:
                images.append(qimg)
                image_slots.append(len(input_ids) - slot_offset)       # count from backward

            for ri, (q, r) in enumerate(reversed(history)):
                if len(input_ids) >= max_length:
                    break
                i = len(history) - ri - 1
                cur_input_ids: List[int] = tokenizer(f"[Round {i}]\n问：", add_special_tokens=False).input_ids
                slot_offset = len(cur_input_ids)
                if isinstance(q, tuple):
                    qtext, qimg = q
                    # image slot, embedding will be replaced by image embeddings
                    cur_input_ids.extend([tokenizer.unk_token_id] * nvtokens)
                else:
                    qtext = q
                    qimg = None
                cur_input_ids += tokenizer(qtext + f"\n答：{r}\n", add_special_tokens=False).input_ids
                input_ids = cur_input_ids + input_ids
                if qimg is not None:
                    images.append(qimg)
                    image_slots.append(len(input_ids) - slot_offset)       # count from backward
        else:
            input_ids = []
            if isinstance(query, tuple):
                qtext, qimg = query
                # image slot, embedding will be replaced by image embeddings
                input_ids.extend([tokenizer.unk_token_id] * nvtokens)
            else:
                qtext = query
                qimg = None
            input_ids += tokenizer(qtext).input_ids
            if qimg is not None:
                images.append(qimg)
                image_slots.append(len(input_ids))       # count from backward

        if len(input_ids) >= max_length:
            # truncate
            if image_slots[-1] > max_length and image_slots[-1] - nvtokens < max_length:
                # A non-intact image slot is not allowed
                input_ids = input_ids[-(image_slots[-1] - nvtokens):]
            else:
                input_ids = input_ids[-max_length:]
            if image_slots[-1] > max_length:
                image_slots.pop()
                images.pop()

        # 2. Prepare image embeddings
        if len(images) != 0:
            image = torch.cat(list(images), dim=0)
            vision_outputs = self.blip2.vision_model.forward(image)
            image_embeds = vision_outputs[0]
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.blip2.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs = self.blip2.qformer.forward(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
            )
            query_output = query_outputs[0]

            vtokens = self.blip2.language_projection(query_output)
        else:
            vtokens = []

        # 3. Place image embeddings into slots
        input_ids = torch.as_tensor(input_ids, dtype=torch.long).to(device).unsqueeze(0)
        inputs_embeds = self.language.transformer.word_embeddings(input_ids)
        for slot, vimg in zip(image_slots, vtokens):
            inputs_embeds[0][-slot:-slot+nvtokens,:] = vimg

        logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                    "temperature": temperature, "logits_processor": logits_processor}

        for outputs in self.language.mm_stream_generate(input_ids=input_ids, inputs_embeds=inputs_embeds, **gen_kwargs):
            outputs = outputs.tolist()[0][len(input_ids[0]):]
            response = tokenizer.decode(outputs)
            response = self.language.process_response(response)
            new_history = history + [(query, response)]
            yield response, new_history
