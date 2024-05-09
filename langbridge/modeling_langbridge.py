# code adapted from flamingo-mini https://github.com/dhansmair/flamingo-mini

from __future__ import annotations
from abc import ABC
from typing import Any, Dict
import contextlib
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pdb
from transformers import PreTrainedModel, AutoModel, AutoConfig, PreTrainedTokenizer, MT5EncoderModel, UMT5EncoderModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast
)
from transformers import AutoTokenizer, AutoModelForCausalLM, MT5ForConditionalGeneration, MT5Tokenizer, MT5Model, MT5Config

from .configuration_langbridge import LangBridgeConfig
from .alignment_modules import LinearWithAddedEos, PerceiverResampler, FFNWithAddedEos, Linear


@contextlib.contextmanager
def suppress_model_loading_warnings(suppress: bool = True):
    if suppress:
        logger = logging.getLogger('transformers.modeling_utils')
        level = logger.level
        logger.setLevel(logging.CRITICAL)
        yield
        logger.setLevel(level)
    else:
        yield


class LBBaseModel(ABC, PreTrainedModel):

    config: LangBridgeConfig
    enc: PreTrainedModel
    lm: PreTrainedModel
    lm_head: nn.Linear
    embeddings: nn.Embedding
    multi_lm: PreTrainedModel
    config_class = LangBridgeConfig

    def __init__(self, config: LangBridgeConfig, random_init=True, suppress_warnings=True):
        super().__init__(config)
        if 'umt5' in config.enc.lower():
            enc_class = UMT5EncoderModel
        elif 'mt5' in config.enc.lower():
            enc_class = MT5EncoderModel
        else:
            enc_class = AutoModel

        with suppress_model_loading_warnings(suppress_warnings):
            if random_init:
                enc_config = AutoConfig.from_pretrained(
                    config.enc)
                self.enc = enc_class(config=enc_config)
            else:
                print('loading encoder from pretrained')
                # print(config.dim_lm,config.dim_multi_lm)
                self.enc = enc_class.from_pretrained(config.enc)

        # self.enc.gradient_checkpointing_enable(
            # gradient_checkpointing_kwargs={'use_reentrant': False})

        if config.alignments == 'linear':  # default
            self.alignment = LinearWithAddedEos(
                dim=config.dim_enc, out_dim=config.dim_lm)
        elif config.alignments == 'ffn':  # mlp
            self.alignment = FFNWithAddedEos(
                dim=config.dim_enc, out_dim=config.dim_lm)
        elif config.alignments == 'latent':
            self.alignment = PerceiverResampler(
                dim=config.dim_enc, out_dim=config.dim_lm, num_latents=config.num_latents)
        else:
            raise ValueError(
                f'unknown alignment type {config.alignments}')

        if config.alignments1 == 'linear':  # default
            self.alignment1 = Linear(
                in_features=config.dim_lm, out_features=config.dim_multi_lm)
        if "mt5" in config.multi_lm:
            # class MT5DecoderWrapper(MT5ForConditionalGeneration):
            #     def __init__(self, config, *args, **kwargs):
            #         super().__init__(config=config, *args, **kwargs)
            #         self.model = self.decoder  # Directly access the decoder
                    
            #     @classmethod
            #     def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
            #         model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
            #         return cls(config=model.config, **kwargs)
            class MT5DecoderOnlyWrapper(MT5ForConditionalGeneration):
                def __init__(self, config):
                    super().__init__(config)

                @classmethod
                def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
                    model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
                    config = MT5Config.from_pretrained(pretrained_model_name_or_path)
                    config.is_encoder_decoder = False  # Remove encoder
                    model = MT5DecoderOnlyWrapper(config)
                    model = model.decoder
                    model.load_state_dict(model.state_dict(), strict=False)  # Load decoder weights
                    return model
            self.multi_lm = MT5DecoderOnlyWrapper.from_pretrained(config.multi_lm)
        else:
            self.multi_lm = AutoModelForCausalLM.from_pretrained(config.multi_lm)
        self.multi_lm.to(self.device)
        self.multi_embeddings = self.multi_lm.get_input_embeddings()
        for param in self.multi_lm.parameters():
            param.requires_grad = False

    def freeze_encoder(self):
        """freeze vision model """
        for param in self.enc.parameters():
            param.requires_grad = False

    def freeze_lm(self):
        for param in self.lm.parameters():
            param.requires_grad = False

    def unfreeze_lm(self):
        for param in self.lm.parameters():
            param.requires_grad = True

    def freeze_multi_lm(self):
        for param in self.multi_lm.parameters():
            param.requires_grad = False

    def unfreeze_multi_lm(self):
        for param in self.multi_lm.parameters():
            param.requires_grad = True

    # get soft prompts
    def get_encoder_features(self, enc_ids: torch.Tensor, enc_mask: torch.Tensor) -> torch.Tensor:
        if self.config.freeze_encoder:
            with torch.no_grad():
                enc_features = self.enc(
                    input_ids=enc_ids, attention_mask=enc_mask).last_hidden_state  # (b, s, d)
        else:
            enc_features = self.enc(
                input_ids=enc_ids, attention_mask=enc_mask).last_hidden_state
        enc_features = self.alignment(enc_features, enc_mask)
        return enc_features

    def forward(
        self,
        enc_ids: torch.Tensor | None = None,
        enc_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        multi_input_ids: torch.Tensor | None = None,
        multi_attention_mask: torch.Tensor | None = None,
        use_cache: bool = True,
        past_key_values: tuple | None = None,
        return_dict: bool = True,
        labels: torch.Tensor | None = None,
        multi_labels: torch.Tensor | None = None,
        loss_reduction: str = 'mean',
        **kwargs
    ) -> CausalLMOutputWithPast:
        # sanity check
        assert return_dict, "can only use return_dict=True at the moment!"

        # find the input shape
        batch_size, seq_length = input_ids.shape[:
                                                 2] if input_ids is not None else enc_ids.shape[:2]
        device = input_ids.device if input_ids is not None else enc_ids.device
        lm_past_key_values = None if past_key_values is None else past_key_values[0]
        # lm_past_key_values_multi= None if past_key_values is None else past_key_values[0]
        if input_ids is not None:
            embeddings = self.embeddings(input_ids)
        bos_shifted = False
        if lm_past_key_values is None:
            assert enc_ids.size(0) == batch_size

            enc_features = self.get_encoder_features(
                enc_ids, enc_mask)

            if input_ids is not None:
                first_input_ids = input_ids[:, 0]
                if all(first_input_ids == self.lm.config.bos_token_id):
                    # move bos embedding to the front
                    bos_shifted = True
                    embeddings = torch.cat(
                        [embeddings[:, 0].unsqueeze(dim=1), enc_features, embeddings[:, 1:]], dim=1)
                else:
                    embeddings = torch.cat([enc_features, embeddings], dim=1)
            else:
                embeddings = enc_features
            enc_feature_length = enc_features.shape[1]
        else:
            enc_feature_length = past_key_values[1]
        
        if input_ids is not None:
            if self.config.alignments not in ['linear', 'ffn']:
                attn_mask = torch.cat(
                    [torch.ones((batch_size, enc_feature_length), device=device, dtype=torch.long), attention_mask], dim=1)
            else:
                # use the encoder masks for the soft prompts
                if bos_shifted:
                    # TODO: messy code
                    # torch.ones are there since the alignment adds a single learnable eos token at the enc
                    attn_mask = torch.cat(
                        [attention_mask[:, 0].unsqueeze(dim=1), enc_mask, torch.ones((batch_size, 1), device=device, dtype=torch.long), attention_mask[:, 1:]], dim=1)
                else:
                    attn_mask = torch.cat(
                        [enc_mask, torch.ones((batch_size, 1), device=device, dtype=torch.long), attention_mask], dim=1)
        else:
            attn_mask = enc_mask
        # pass through LM
        # out: BaseModelOutputWithPast = self.lm(attention_mask=attn_mask,inputs_embeds=embeddings,use_cache=use_cache,past_key_values=lm_past_key_values,return_dict=True,**kwargs)
        # pdb.set_trace()
        out: BaseModelOutputWithPast = self.lm(
            attention_mask=attn_mask, #[4,1001]
            inputs_embeds=embeddings, #[4,1001,4096]
            use_cache=False,
            # past_key_values=lm_past_key_values, 
            return_dict=True,
            **kwargs
        )
        # out has two element -> first of size [4,1001,4096] other one is tuple 32 tuples i think each layer output
        outputs=self.alignment1(out.last_hidden_state)
        # attn_mask=self.alignment1(attn_mask)
        # pdb.set_trace()
        # self.freeze_multi_lm()
        # final_out=self.multi_lm.transformer(attention_mask=attn_mask,inputs_embeds=outputs,use_cache=use_cache,past_key_values=lm_past_key_values,return_dict=True,**kwargs)

        if multi_input_ids is not None:
            multi_embeddings = self.multi_embeddings(multi_input_ids)

        attn_mask=torch.cat([attn_mask,multi_attention_mask],dim=1)
        multi_embeddings=torch.cat([outputs,multi_embeddings],dim=1)
        
        final_out=self.multi_lm.transformer(
            attention_mask=attn_mask,
            inputs_embeds=multi_embeddings,
            use_cache=use_cache,
            past_key_values=lm_past_key_values,
            # return_dict=False,
            # **kwargs
            )

        final_logits: torch.Tensor = self.multi_lm.lm_head(final_out.last_hidden_state)
        # logits: torch.Tensor = self.lm_head(out.last_hidden_state)
        # logits shape ---> [4,1001,32003]
        loss = None
            
            # label shape [4,109] ---> labels only for output
        if labels is not None:
            # no loss for soft prompts
            no_loss_labels = torch.zeros(
                (batch_size, enc_feature_length + labels.shape[1]), device=device, dtype=torch.long) + -100
            # labels for inputs ---> [4,892]
            bos_shifted=False
            if bos_shifted:
                full_labels = torch.cat(
                    [labels[:, 0].unsqueeze(dim=1), no_loss_labels, labels[:, 1:]], dim=1)
                    # full_labels = labels(org) + unlabel(-100) ---> [4,1001]
            else:
                full_labels = torch.cat(
                    [no_loss_labels, multi_labels], dim=1)
            # logits shape (batch, seq_length, #words)
            shift_logits = final_logits[..., :-1, :].contiguous() #[4,1000,32003]
            # labels shape (batch, seq_length)
            shift_labels = full_labels[..., 1:].contiguous() #[4,1000] ---> both for input and output.

            # Flatten the tokens
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), #[4000,32003]
                                shift_labels.view(-1), reduction=loss_reduction) #[4000]
            if loss_reduction == 'none':
                # CrossEntropyLoss will flatten all dimensions by default
                loss = rearrange(loss, '(b s) -> b s', b=batch_size)
                # remove soft promtps from loss
                loss = loss[:, enc_feature_length+labels.shape[1]:]
        # pdb.set_trace()
        return CausalLMOutputWithPast(
            loss=loss,
            logits=final_logits,
            past_key_values=(final_out.past_key_values,
                             enc_feature_length+labels.shape[1]) if use_cache else None,
            # past_key_values_multi=(final_out.past_key_values,
            #         enc_feature_length) if use_cache else None,
            hidden_states=final_out.hidden_states,
            attentions=final_out.attentions,
        )


# used for debbuging with opt-125m
class LBOPT(LBBaseModel):
    config: LangBridgeConfig

    def __init__(self, config: LangBridgeConfig, random_init=True):
        from transformers import OPTForCausalLM, OPTModel
        super().__init__(config, random_init=random_init)

        if random_init:
            model_config = AutoConfig.from_pretrained(config.lm)
            base_lm: OPTForCausalLM = OPTForCausalLM(config=model_config)
        else:
            print('loading lm from pretrained')
            base_lm: OPTForCausalLM = OPTForCausalLM.from_pretrained(
                config.lm)
        assert self.config.dim_lm == base_lm.config.hidden_size, \
            f"specified {self.config.dim_lm=} in LangBridgeConfig, but {config.lm} has hidden size={base_lm.config.hidden_size}"
        self.lm: OPTModel = base_lm.model
        self.lm_head = base_lm.lm_head
        self.embeddings = base_lm.get_input_embeddings()


class LBLlama(LBBaseModel):
    config: LangBridgeConfig

    def __init__(self, config: LangBridgeConfig, random_init=True):
        from transformers import LlamaForCausalLM, LlamaModel
        super().__init__(config, random_init=random_init)

        if random_init:
            model_config = AutoConfig.from_pretrained(config.lm)
            try:
                model_config.attn_implementation = 'flash_attention_2'
                base_lm: LlamaForCausalLM = LlamaForCausalLM(
                    config=model_config)
            except ImportError:
                print('Not using Flash Attention!')
                base_lm: LlamaForCausalLM = LlamaForCausalLM(
                    config=model_config)
        else:
            print('loading lm from pretrained')
            try:
                base_lm: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(
                    config.lm, use_flash_attention_2=True)
            except ImportError:
                print('Not using Flash Attention!')
                base_lm: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(
                    config.lm)

        assert self.config.dim_lm == base_lm.config.hidden_size, \
            f"specified {self.config.dim_lm=} in LangBridgeConfig, but {config.lm} has hidden size={base_lm.config.hidden_size}"
        self.lm: LlamaModel = base_lm.model
        self.lm_head = base_lm.lm_head
        self.embeddings = base_lm.get_input_embeddings()


class LBMistral(LBBaseModel):
    config: LangBridgeConfig

    def __init__(self, config: LangBridgeConfig, random_init=True):
        from transformers import MistralForCausalLM, MistralModel
        super().__init__(config, random_init=random_init)

        if random_init:
            model_config = AutoConfig.from_pretrained(config.lm)
            try:
                model_config.attn_implementation = 'flash_attention_2'
                base_lm: MistralForCausalLM = MistralForCausalLM(
                    config=model_config)
            except ImportError:
                print('Not using Flash Attention!')
                base_lm: MistralForCausalLM = MistralForCausalLM(
                    config=model_config)
        else:
            try:
                base_lm: MistralForCausalLM = MistralForCausalLM.from_pretrained(
                    config.lm, use_flash_attention_2=True)
            except ImportError:
                print('Not using Flash Attention!')
                base_lm: MistralForCausalLM = MistralForCausalLM.from_pretrained(
                    config.lm)
        assert self.config.dim_lm == base_lm.config.hidden_size, \
            f"specified {self.config.dim_lm=} in LangBridgeConfig, but {config.lm} has hidden size={base_lm.config.hidden_size}"

        self.lm: MistralModel = base_lm.model
        self.lm_head = base_lm.lm_head
        self.embeddings = base_lm.get_input_embeddings()


class LangBridgeModel(PreTrainedModel):
    config: LangBridgeConfig
    config_class = LangBridgeConfig

    _LANGUAGE_MODEL_VERSIONS = {
        'facebook/opt': LBOPT,
        'EleutherAI/llemma': LBLlama,
        'codellama/CodeLlama': LBLlama,
        'microsoft/Orca-2': LBLlama,
        'meta-math/MetaMath': LBLlama,
        'meta-llama/Llama-2-7b-hf': LBLlama,
        'mistralai/Mistral-7B-v0.1': LBMistral,
    }

    def __init__(self, config: LangBridgeConfig, random_init=True, model_class=None):
        super().__init__(config)

        if model_class is None:
            model_class = self._find_lm_class(config.lm)
        self.lb: LBBaseModel = model_class(config, random_init=random_init)
        # self.decoder
        if config.freeze_language_model:
            self.freeze_lm()

        if config.freeze_encoder:
            self.freeze_encoder()

    @classmethod
    def _find_lm_class(cls, language_model_id: str):
        for prefix, lm_class in cls._LANGUAGE_MODEL_VERSIONS.items():
            if language_model_id.startswith(prefix):
                return lm_class
        raise ValueError(f'unsupported language model {language_model_id}')

    def freeze_encoder(self):
        self.lb.freeze_encoder()

    def freeze_lm(self):
        self.lb.freeze_lm()

    def unfreeze_lm(self):
        self.lb.unfreeze_lm()

    def forward(
        self,
        enc_ids: torch.Tensor | None = None,
        enc_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = True,
        past_key_values: tuple | None = None,
        return_dict: bool = True,
        labels: torch.Tensor | None = None,
        loss_reduction: str = 'mean',
        **kwargs
    ) -> CausalLMOutputWithPast:

        return self.lb(
            enc_ids=enc_ids,
            enc_mask=enc_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            labels=labels,
            loss_reduction=loss_reduction,
            **kwargs
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        enc_ids: torch.Tensor | None = None,
        enc_mask: torch.Tensor | None = None,
        past=None,
        past_key_values=None,
        **kwargs
    ) -> Dict[str, Any]:
        """ hf specific function. Overridden from PreTrainedModel for text generation purposes.

        if use_cache is used, past is not None, then only the last column will be passed as input_ids.
        TODO was `past` renamed to `past_key_values` in transformers 4.26?
        """

        if past_key_values is not None or past is not None:
            input_ids = input_ids[:, -1:]

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            enc_ids=enc_ids,
            enc_mask=enc_mask,
            past_key_values=past_key_values if past_key_values is not None else past,
            **kwargs
        )

    def _reorder_cache(self, past, beam_idx):
        """ hf specific function. Overridden from PreTrainedModel.

        this is required for beam search in combination with use_cache.

        Args: 
            past is a tuple of past_key_values of the xattn layers, and of the LM layers.
            beam_idx: index of the beam
        """
        xattn_past, lm_past = past

        xattn_past_beam = tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device))
                  for past_state in layer_past)
            for layer_past in xattn_past
        )

        lm_past_beam = tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device))
                  for past_state in layer_past)
            for layer_past in lm_past
        )

        return xattn_past_beam, lm_past_beam

    # a simple function to test the model
    @torch.no_grad()
    def generate_from_prefix(
        self,
        enc_tokenizer: PreTrainedTokenizer,
        lm_tokenizer: PreTrainedTokenizer,
        prefix: str,
        max_length: int = 150,
        num_beams: int = 1,
        **kwargs
    ):
        enc_input = enc_tokenizer(prefix, return_tensors='pt', padding=True)
        enc_ids = enc_input['input_ids'].to(self.device)
        enc_mask = enc_input['attention_mask'].to(self.device)

        input_ids = torch.LongTensor([lm_tokenizer.bos_token_id])
        input_ids = input_ids.repeat(enc_ids.shape[0], 1).to(self.device)
        attention_mask = torch.ones_like(input_ids)

        out_ids = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            enc_ids=enc_ids,
            enc_mask=enc_mask,
            num_beams=num_beams,
            early_stopping=True,
            use_cache=True,
            bos_token_id=lm_tokenizer.bos_token_id,
            eos_token_id=lm_tokenizer.eos_token_id,
            pad_token_id=lm_tokenizer.eos_token_id,
            max_length=max_length,
            **kwargs
        )

        captions = lm_tokenizer.batch_decode(
            out_ids, skip_special_tokens=True)
        return captions
