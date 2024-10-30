from transformers import BartPretrainedModel,BartModel,BartConfig #,shift_tokens_right
from transformers.utils import ModelOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from dataclasses import dataclass


class Linear_Block(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Linear_Block,self).__init__()
        self.selu= nn.Tanh() # nn.SiLU() #nn.Sigmoid() # 
        self.linear=nn.Linear(in_channels, out_channels)
        self.batchNorm=nn.BatchNorm1d(out_channels)
  
    def forward(self,x):
        x_skip=self.linear(x)
        x_skip=self.batchNorm(x_skip)
        x=self.selu(x_skip)
        return x,x_skip


class BartPooler(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(2048, 1024)
        self.activation = nn.Tanh() # nn.SiLU() # nn.Sigmoid() # 

    def forward(self, hidden_states,turns,parts):
        batch_list=[]
        for i in range(len(hidden_states)):
            start=1
            for j in range(turns[i]):
                end=torch.sum(parts[i,:j+1])+1
                slice_=hidden_states[i,start:end,:]
                max_pool=torch.amax(slice_, dim=0,keepdim=True)
                avg_pool=torch.mean(slice_, dim=0,keepdim=True)
                batch_list.append(torch.cat([max_pool,avg_pool],dim=-1))
                start=end

        first_token_tensor = torch.cat(batch_list,dim=0)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BypassModel(nn.Module):
    def __init__(self):
        super(BypassModel,self).__init__()
        self.bart_pooler=BartPooler()
        self.LB_pool=Linear_Block(1024,1024)
        
        self.LB_1_emotions=Linear_Block(1024,1024)
        self.drop_1_emotions=nn.Dropout(p=0.5)
        self.LB_2_emotions=Linear_Block(1024,1024)
        self.drop_2_emotions=nn.Dropout(p=0.5)
        self.LB_3_emotions=Linear_Block(1024,1024)
        self.drop_3_emotions=nn.Dropout(p=0.5)
        self.LB_4_emotions=Linear_Block(1024,1024)
        self.LB_5_emotions=nn.Linear(1024,7)
        
        
        self.LB_1_acts=Linear_Block(1024,1024)
        self.drop_1_acts=nn.Dropout(p=0.5)
        self.LB_2_acts=Linear_Block(1024,1024)
        self.drop_2_acts=nn.Dropout(p=0.5)
        self.LB_3_acts=Linear_Block(1024,1024)
        self.drop_3_acts=nn.Dropout(p=0.5)
        self.LB_4_acts=Linear_Block(1024,1024)
        self.LB_5_acts=nn.Linear(1024,5)
        
        
        self.LB_1_intents=Linear_Block(1024,1024)
        self.drop_1_intents=nn.Dropout(p=0.5)
        self.LB_2_intents=Linear_Block(1024,1024)
        self.drop_2_intents=nn.Dropout(p=0.5)
        self.LB_3_intents=Linear_Block(1024,1024)
        self.drop_3_intents=nn.Dropout(p=0.5)
        self.LB_4_intents=Linear_Block(1024,1024)
        self.LB_5_intents=nn.Linear(1024,102)

        
    def forward(self,hidden_states,turns,parts):

        pooled_output=self.bart_pooler(hidden_states,turns,parts)
        pooled_output,_=self.LB_pool(pooled_output)
        
        emotion,_=self.LB_1_emotions(pooled_output)
        emotion=self.drop_1_emotions(emotion)
        emotion,_=self.LB_2_emotions(emotion)
        emotion=self.drop_2_emotions(emotion)
        emotion,_=self.LB_3_emotions(emotion)
        emotion=self.drop_3_emotions(emotion)
        emotion,_=self.LB_4_emotions(emotion)
        emotion=self.LB_5_emotions(emotion)
        
        act,_=self.LB_1_acts(pooled_output)
        act=self.drop_1_acts(act)
        act,_=self.LB_2_acts(act)
        act=self.drop_2_acts(act)
        act,_=self.LB_3_acts(act)
        act=self.drop_3_acts(act)
        act,_=self.LB_4_acts(act)
        act=self.LB_5_acts(act)
        
        intent,_=self.LB_1_intents(pooled_output)
        intent=self.drop_1_intents(intent)
        intent,_=self.LB_2_intents(intent)
        intent=self.drop_2_intents(intent)
        intent,_=self.LB_3_intents(intent)
        intent=self.drop_3_intents(intent)
        intent,_=self.LB_4_intents(intent)
        intent=self.LB_5_intents(intent)
        
        return emotion,act,intent
    

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

@dataclass
class Seq2SeqLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    loss_emotions: Optional[torch.FloatTensor] = None
    loss_acts: Optional[torch.FloatTensor] = None
    loss_intents: Optional[torch.FloatTensor] = None
    prediction_emotions: Optional[torch.FloatTensor] = None
    prediction_acts: Optional[torch.FloatTensor] = None
    prediction_intents: Optional[torch.FloatTensor] = None
   
@dataclass
class Seq2SeqLMOutput_2(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    
        
def get_bypass_loss(prediction_emotions,target_emotions,prediction_acts,target_acts,prediction_intents,target_intents,total_turns,loss_func):

    target_masked_emotions=[]
    target_masked_acts=[]
    target_masked_intents=[]
    
    for i in range(len(target_emotions)):
        target_masked_emotions.append(target_emotions[i,:total_turns[i]])
        target_masked_acts.append(target_acts[i,:total_turns[i]])
        target_masked_intents.append(target_intents[i,:total_turns[i]])
    
    target_masked_emotions = torch.cat(target_masked_emotions,dim=-1)
    target_masked_acts = torch.cat(target_masked_acts,dim=-1)
    target_masked_intents = torch.cat(target_masked_intents,dim=-1)
    

    loss_emotions=loss_func(prediction_emotions,target_masked_emotions)
    loss_acts=loss_func(prediction_acts,target_masked_acts)
    loss_intents=loss_func(prediction_intents,target_masked_intents)

    
    return loss_emotions,loss_acts,loss_intents


class BartForConditionalGeneration(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.BartBypass=None
        self.cross_entropy_loss=nn.CrossEntropyLoss()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        emotions: Optional[torch.Tensor] = None,
        acts: Optional[torch.Tensor] = None,
        intents: Optional[torch.Tensor] = None,
        turns: Optional[torch.Tensor] = None,
        parts: Optional[torch.Tensor] = None,
        
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        
        if turns is not None:
            encoder_ouptput=outputs.encoder_last_hidden_state
            prediction_emotions,prediction_acts,prediction_intents=self.BartBypass(encoder_ouptput,turns,parts)
            loss_emotions,loss_acts,loss_intents=get_bypass_loss(prediction_emotions,emotions,prediction_acts,acts,prediction_intents,intents,turns,self.cross_entropy_loss)

            return Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
                loss_emotions=loss_emotions,
                loss_acts=loss_acts,
                loss_intents=loss_intents,
                prediction_emotions=prediction_emotions,
                prediction_acts=prediction_acts,
                prediction_intents=prediction_intents,
            )
        else:
            return Seq2SeqLMOutput_2(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )
        
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
    


def get_unified_bert(model_checkpoint, unified_model = None):
    model = BartForConditionalGeneration.from_pretrained(model_checkpoint)
    model.BartBypass=BypassModel()
    if unified_model:
        model.load_state_dict(torch.load(unified_model))
    return model

    