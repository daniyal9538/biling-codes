import json
import os
from typing import List, Optional, Dict, Union
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BertConfig, BertModel, BertForMaskedLM
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, MaskedLMOutput

def max_len_pad(*sequences, max_length: int, pad_value=0):
    padded_sequences = []
    
    for seq in sequences:
        if len(seq) < max_length:
            padded_seq = seq + [pad_value] * (max_length - len(seq))
        else:
            padded_seq = seq[:max_length]
        padded_sequences.append(padded_seq)
    return padded_sequences

class BillingTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.json"}

    def __init__(
        self, 
        vocab_file=None, 
        unk_token="[UNK]", 
        sep_token="[SEP]", 
        pad_token="[PAD]", 
        cls_token="[CLS]", 
        mask_token="[MASK]", 
        add_cls = False,
        age_ids = False,
        segment_ids = False,
        Lo7_task = False,
        **kwargs
    ):
        # 1. INITIALIZE VOCABULARY
        self.add_cls = add_cls
        self.age_ids = age_ids
        self.segment_ids = segment_ids
        self.Lo7_task = Lo7_task
        self.vocab = {}
        if vocab_file:
            with open(vocab_file, encoding="utf-8") as f:
                self.vocab = json.load(f)
        else:
            # Default vocab
            self.vocab = {
                pad_token: 0,
                unk_token: 1,
                cls_token: 2,
                sep_token: 3,
                mask_token: 4,
            }
        
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

        # 2. CALL PARENT INIT
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            vocab_file=vocab_file,
            **kwargs
        )

    def __call__(self, text, **kwargs):
        # This is the main entry point for tokenization
        # We can handle the custom lists here before passing to the parent

        return super().__call__(text, **kwargs)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text, **kwargs):
        if isinstance(text, list):
            if self.add_cls:
                return [self.cls_token] + [str(t) for t in text]
            return [str(t) for t in text]
        if self.add_cls:
            return [self.cls_token] + text.split() 
        return text.split()
    
    def tokenize_df(self, df: pd.DataFrame,  code_column: str = 'icd_code_mapped') -> List[List[str]]:
        codes = ' '.join(df[code_column].tolist())
        return self._tokenize(codes)

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )
        
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)
            
        return (vocab_file,)

    def train_from_list(self, code_list: List[List[str]]):
        unique_codes = set()
        for codes in code_list:
    
                unique_codes.add(str(codes))
        
        sorted_codes = sorted(list(unique_codes))
        next_id = len(self.vocab)
        
        for code in sorted_codes:
            if code not in self.vocab:
                self.vocab[code] = next_id
                self.ids_to_tokens[next_id] = code
                next_id += 1
                
        print(f"Vocabulary trained. Size: {len(self.vocab)}")



    def tokenize_dict(self, data: Union[Dict, List[Dict]], code_column: str = 'icd_code_mapped', age_column: str = 'age_month', **kwargs):
        return_tensors = kwargs.get('return_tensors', None)
        device = kwargs.get('device', 'cpu')
        padding = kwargs.get('padding', False)
        max_length = kwargs.get('max_length', self.model_max_length)

        if isinstance(data, list):
            codes = [' '.join(d[code_column]) for d in data]
        
                
            
            if self.add_cls:
                segments_raw = [d['segment'][0] + d['segment'] for d in data]
            else:
                segments_raw = [d['segment'] for d in data]
            
            if self.add_cls:
                ages_raw = [d[age_column][0] + d[age_column] for d in data]
            else:
                ages_raw = [d[age_column] for d in data]
            Lo7_raw = [d['Lo7'][0] for d in data] 
            if padding:
                for i in range(len(codes)):
                    segments_raw[i], ages_raw[i] = max_len_pad(segments_raw[i], ages_raw[i], max_length=max_length)
            is_list = True
        else:
            codes = ' '.join(data[code_column])
            segments_raw = data['segment']
            if self.add_cls:
                segments_raw = [segments_raw[0]] + segments_raw
            
            ages_raw = data[age_column]
            if self.add_cls:
                ages_raw = [ages_raw[0]] + ages_raw
            Lo7_raw = data['Lo7'][0] 
            if padding:
                segments_raw, ages_raw = max_len_pad(segments_raw, ages_raw, max_length=max_length)
            is_list = False

        if return_tensors == 'pt':
            segments = torch.tensor(segments_raw, device=device)
            age_list = torch.tensor(ages_raw, device=device)
            Lo7_list = torch.tensor(Lo7_raw, device=device) 
            
            if not is_list:
                segments = segments.unsqueeze(0)
                age_list = age_list.unsqueeze(0)
                Lo7_list = Lo7_list.unsqueeze(0)
        else:
            segments = segments_raw
            age_list = ages_raw
            Lo7_list = Lo7_raw
            
        output = self(codes, **kwargs)
        if self.segment_ids:
            output['segment_ids'] = segments
        if self.age_ids:
            output['age_ids'] = age_list
        if self.Lo7_task:
            output['cls_labels'] = Lo7_list
        return output


    def tokenize_df(self, df: Union[pd.DataFrame, List[pd.DataFrame]], code_column: str = 'icd_code_mapped', **kwargs):
        if isinstance(df, list):
            data_list = []
            for single_df in df:
                data_list.append(single_df.to_dict(orient='list'))
            output = self.tokenize_dict(data_list, code_column=code_column, **kwargs)
        else:
            data_dict = df.to_dict(orient='list')
            output = self.tokenize_dict(data_dict, code_column=code_column, **kwargs)
        return output



class BillingDataset(Dataset):
    def __init__(self, data, tokenizer, **tokenizer_kwargs):
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length
        self.tokenizer_kwargs = tokenizer_kwargs
        # Pre-process data into a list of dictionaries for efficient access during training
        if isinstance(data, pd.core.groupby.DataFrameGroupBy):
            print("Converting grouped data to list of dictionaries...")
            self.data = [group.to_dict(orient='list') for _, group in data]
        elif isinstance(data, list):
            self.data = data
        else:
            raise ValueError("Data must be a pandas DataFrameGroupBy object or a list of dictionaries")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize the dictionary
        # We request PyTorch tensors. Based on the custom tokenizer implementation,
        # passing a single dict usually results in tensors with shape [1, seq_len].
        encoded = self.tokenizer.tokenize_dict(
            item, 
            padding='max_length', 
            max_length=self.max_length, 
            return_tensors='pt',
            truncation=True,
            **self.tokenizer_kwargs
        )
        
        # Squeeze the tensors to remove the batch dimension (size 1) added by the tokenizer
        # The DataLoader will add a new batch dimension.
        item_output = {key: val.squeeze(0) if torch.is_tensor(val) else val for key, val in encoded.items()}
        
        return item_output
    
class IteratorBillingDataset(Dataset):
    def __init__(self, data_iterator, tokenizer):
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length
        # Convert the iterator to a list to support indexing (__getitem__)
        self.data = list(data_iterator)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize the dictionary using the custom tokenizer
        encoded = self.tokenizer.tokenize_dict(
            item, 
            padding='max_length', 
            max_length=self.max_length, 
            return_tensors='pt',
            truncation=True
        )
        
        # Remove the batch dimension [1, seq_len] -> [seq_len]
        return {key: val.squeeze(0) if torch.is_tensor(val) else val for key, val in encoded.items()}


class BCBERTConfig(BertConfig):
    """
    Configuration class for Med-BERT based on the paper specifications.
    
    Paper specifications:
    - Layers (L): 6 [cite: 816]
    - Attention Heads (A): 6 [cite: 816]
    - Hidden Dimension (H): 192 [cite: 816]
    - Feed-forward/filter size: 64 [cite: 816] 
      (Note: The paper states 64 for 'feed-forward/filter size'. 
       Standard BERT usually sets intermediate size to 4x hidden. 
       We will set intermediate_size to 64 as explicitly stated.)
    - Vocab Size: 82,603 (ICD-9 + ICD-10) [cite: 122]
    """
    def __init__(
        self,
        vocab_size=13481, 
        hidden_size=192,
        num_hidden_layers=6,
        num_attention_heads=6,
        intermediate_size=64, # [cite: 816]
        max_position_embeddings=120, # [cite: 817]
        type_vocab_size=2, # Expanded to handle Visit IDs (conceptually unlimited, but set to max seq len)
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        cls_pooler= False,
        return_dict=True,
        age_vocab_size=38806,
        enable_age_ids=False,
        enable_segment_ids=False,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            return_dict=return_dict,
            **kwargs
        )
        self.age_vocab_size = age_vocab_size
        self.cls_pooler = cls_pooler
        self.enable_age_ids = enable_age_ids
        self.enable_segment_ids = enable_segment_ids
        
class BCBERTEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.age_vocab_size = getattr(config, "age_vocab_size", 38806)
        self.cls_pooler = getattr(config, "cls_pooler", False)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.enable_age_ids = getattr(config, "enable_age_ids", False)
        self.enable_segment_ids = getattr(config, "enable_segment_ids", False)
        # Custom embeddings for Age and Segment
        if self.enable_age_ids:
            self.age_embeddings = nn.Embedding(config.age_vocab_size, config.hidden_size) # Similar to position embedding since age is sequential in nature
        if self.enable_segment_ids:
            self.segment_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size) #  Similar to position embedding since segment is sequential in nature

        # self.LayerNorm is not snake-cased to keep with original BERT implementation from transformers
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0,
        age_ids=None, segment_ids=None
    ):

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting defaults for device
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        # If token_type_ids is None, standard BERT sets to 0.

        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        # Standard BERT embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            # print(f"Position IDs shape: {position_ids.shape}, Position Embeddings shape: {position_embeddings.shape}m Token Type IDs shape: {token_type_ids.shape}, Token Type Embeddings shape: {token_type_embeddings.shape}")
            embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        else:
            embeddings = inputs_embeds + token_type_embeddings
        # print(f"Embeddings shape before custom additions: {embeddings.shape}")
        # print(self.age_embeddings,self.segment_embeddings)
        # Add Custom Embeddings if provided
        if self.enable_age_ids and age_ids is not None:
            age_embeds = self.age_embeddings(age_ids)
            embeddings += age_embeds
        
        if self.enable_segment_ids and segment_ids is not None:
            segment_embeds = self.segment_embeddings(segment_ids)
            embeddings += segment_embeds

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BCBERT(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.cls_pooler = getattr(config, "cls_pooler", False)
        
        # Override the standard embeddings with our custom one
        self.embeddings = BCBERTEmbeddings(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        age_ids=None,      # Custom argument
        segment_ids=None,  # Custom argument
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        # print(input_ids.shape,age_ids.shape,segment_ids.shape)
        # Run the embeddings layer with custom arguments
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            age_ids=age_ids,
            segment_ids=segment_ids
        )

        # The rest of the forward pass in BertModel
        # We handle get_extended_attention_mask manually as the base class method expects inputs we have
        
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = encoder_outputs[0]
        pooled_output = None
        if self.cls_pooler:
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        else:
            if attention_mask is not None:
                # Expand mask: [Batch, Seq] -> [Batch, Seq, Hidden]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
                sum_embeddings = torch.sum(sequence_output * input_mask_expanded, dim=1)
                mean_embeddings = sum_embeddings / input_mask_expanded.sum(dim=1).clamp(min=1e-9)
            else:
                sum_embeddings = torch.sum(sequence_output, dim=1)
                mean_embeddings = sum_embeddings / sequence_output.size(1)
            pooled_output = mean_embeddings
                
            
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=None,
        )






class BCBERTForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        # Replace the standard BERT module with our custom BCBERT
        self.bert = BCBERT(config, add_pooling_layer=False)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        age_ids=None,      # Custom argument
        segment_ids=None,  # Custom argument
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass custom arguments (age_ids, segment_ids) to the base model
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            age_ids=age_ids,
            segment_ids=segment_ids,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )






# --- 1. Define the Custom CSV Logger ---
from transformers import TrainerCallback
import pandas as pd
import os
class CSVLogCallback(TrainerCallback):
    """
    A custom callback that logs training and evaluation metrics 
    to CSV files in real-time.
    """
    def __init__(self, output_dir,name):
        self.train_log_path = os.path.join(output_dir, f"train_log.csv")
        self.eval_log_path = os.path.join(output_dir, f"eval_log.csv")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Triggered whenever the Trainer logs something (e.g., training loss or eval metrics).
        """
        if logs is None:
            return

        # Separate training logs from evaluation logs based on keys
        # Training logs usually have 'loss'; Eval logs start with 'eval_'
        is_eval = any(k.startswith("eval_") for k in logs.keys())
        
        target_path = self.eval_log_path if is_eval else self.train_log_path
        
        # Prepare the DataFrame
        df = pd.DataFrame([logs])
        
        # Ensure 'epoch' and 'step' are always recorded for plotting later
        if 'epoch' not in df.columns:
            df['epoch'] = state.epoch
        if 'step' not in df.columns:
            df['step'] = state.global_step
            
        # Write to CSV (Append mode)
        # We write the header only if the file doesn't exist yet
        header = not os.path.exists(target_path)
        df.to_csv(target_path, mode='a', header=header, index=False)