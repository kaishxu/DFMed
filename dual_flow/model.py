import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForPreTraining
from typing import List, Optional
from GAT import GAT, LayerType

class CrossAttention(nn.Module):
    def __init__(self, hidden_size):
        super(CrossAttention, self).__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, key_value_states, attention_mask=None, add_original_input=False):
        query_states = self.query(hidden_states)
        key_states = self.key(key_value_states)
        value_states = self.value(key_value_states)

        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.hidden_size)

        if attention_mask is not None:
            converted_attention_mask = (1.0 - attention_mask) * torch.finfo(attention_scores.dtype).min
            attention_scores = attention_scores + converted_attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_probs, value_states)

        if add_original_input:
            hidden_states = self.norm(hidden_states + attention_output)
            return hidden_states
        else:
            return attention_output

class ActEntityModel(BertForPreTraining):
    def __init__(self, config):
        super(ActEntityModel, self).__init__(config)
        self.hidden_size = config.hidden_size
        self.act_embeds = nn.Parameter(torch.empty((7, config.hidden_size)))
        self.none_entity_embeds = nn.Parameter(torch.empty((1, config.hidden_size)))
        self.none_act_embeds = nn.Parameter(torch.empty((1, config.hidden_size)))
        self.context_act_attn = CrossAttention(config.hidden_size)
        self.context_entity_attn = CrossAttention(config.hidden_size)
        self.act_entity_attn = CrossAttention(config.hidden_size)
        self.entity_act_attn = CrossAttention(config.hidden_size)

        gru_layer = 2
        self.gru_act = nn.GRU(config.hidden_size * 3, config.hidden_size, gru_layer, batch_first=True)
        self.gru_entity = nn.GRU(config.hidden_size * 3, config.hidden_size, gru_layer, batch_first=True)
        self.w = nn.Linear(config.hidden_size * gru_layer, config.hidden_size)

        self.act_mlp = nn.Linear(config.hidden_size, 7)
        self.entity_mlp = nn.Linear(config.hidden_size, config.n_entity)
        self.sigmoid = nn.Sigmoid()
        self.bceloss = nn.BCELoss()
        self.act_weight = 1
        self.entity_weight = 1

        # Initialize weights and apply final processing
        self.post_init()

        # Initialize GAT
        self.entity_gat = GAT(num_of_layers=1, num_heads_per_layer=[4], num_features_per_layer=[config.hidden_size//4], add_skip_connection=True, dropout=0.1, layer_type=LayerType.IMP2)

    def pooling(self, x):
        return torch.mean(x, dim=-2, keepdim=True)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        context_idx: Optional[List] = None,
        entity_ids: Optional[torch.Tensor] = None,
        entity_mask: Optional[torch.Tensor] = None,
        entity_embeds: Optional[torch.Tensor] = None,
        entity_matrix: Optional[torch.Tensor] = None,
        act_turn_idx: Optional[List] = None,
        entity_turn_idx: Optional[List] = None,
        batch_entity_turn_idx: Optional[List] = None,
        act_labels: Optional[torch.Tensor] = None,
        entity_labels: Optional[torch.Tensor] = None,
        target_kg_entity_idx: Optional[List] = None,
        neg_kg_entity_idx: Optional[List] = None,
    ):

        if entity_ids != None:  # only to calculate entity representation
            encoder_outputs = self.bert(
                input_ids=entity_ids,
                attention_mask=entity_mask)
            entity_state = encoder_outputs.last_hidden_state  #M, L, D
            entity_state = (entity_state * entity_mask[:,:,None]).sum(dim=1) / entity_mask.sum(dim=1)[:,None]
            return entity_state

        encoder_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask)
        context_state = encoder_outputs.last_hidden_state  #B, L, D
        batch_size, _, _ = context_state.shape

        # get all entity embeds in batch (idx based on all entity list)
        batch_entity_embeds = entity_embeds[batch_entity_turn_idx]
        # GAT for entity embeds if training
        # no need to update entity embeds if not training, since embeds have been updated by GAT before evaluation loop.
        if self.training:
            batch_entity_matrix = entity_matrix[batch_entity_turn_idx,:][:,batch_entity_turn_idx]
            batch_entity_matrix = batch_entity_matrix.to(entity_embeds.device)
            batch_entity_embeds, _ = self.entity_gat((batch_entity_embeds, batch_entity_matrix))

        act_loss = 0
        entity_loss = 0
        act_scores_batch = []
        entity_scores_batch = []
        pos_embeds = []
        neg_embeds = []
        entity_hiddens = []
        entity_hiddens_batch = []
        for i in range(batch_size):

            final_act_state_lst = []
            final_entity_state_lst = []
            act_state_turn_lst = []
            act_state_turn_one_lst = []
            entity_state_turn_lst = []
            entity_state_turn_one_lst = []
            n_turn = len(context_idx[i])
            for k in range(n_turn):

                if entity_turn_idx[i][k] != []:
                    turn_entity_embeds = batch_entity_embeds[entity_turn_idx[i][k]]  # (idx based on batch entity list)

                    entity_state_turn_lst.append(turn_entity_embeds)
                    entity_state_turn_one_lst.append(turn_entity_embeds)
                else:
                    entity_state_turn_lst.append(self.none_entity_embeds)
                    entity_state_turn_one_lst.append(self.none_entity_embeds)

                if act_turn_idx[i][k] != []:
                    act_state_turn_lst.append(self.act_embeds[act_turn_idx[i][k]])
                    act_state_turn_one_lst.append(self.act_embeds[act_turn_idx[i][k]])
                else:
                    if act_state_turn_lst == []:  #there is no act in the first turn
                        act_state_turn_lst.append(self.none_act_embeds)
                        act_state_turn_one_lst.append(self.none_act_embeds)
                    act_state_turn = torch.cat(act_state_turn_lst, dim=0)  #act in previous turns
                    act_state_turn_one = torch.cat(act_state_turn_one_lst, dim=0)  #act in current turn
                    entity_state_turn = torch.cat(entity_state_turn_lst, dim=0)  #entity in previous turn
                    entity_state_turn_one = torch.cat(entity_state_turn_one_lst, dim=0)  #entity in current turn
                    context_state_turn = context_state[i][:context_idx[i][k][1]]  #context

                    context_act_state = self.context_act_attn(self.pooling(context_state_turn), act_state_turn_one, add_original_input=True)
                    context_entity_state = self.context_entity_attn(self.pooling(context_state_turn), entity_state_turn_one, add_original_input=True)

                    entity_act_state = self.entity_act_attn(self.pooling(entity_state_turn_one), act_state_turn, add_original_input=True)
                    act_entity_state = self.act_entity_attn(self.pooling(act_state_turn_one), entity_state_turn, add_original_input=True)

                    final_act_state_turn = torch.cat([self.pooling(act_state_turn_one), context_act_state, act_entity_state], dim=-1)
                    final_entity_state_turn = torch.cat([self.pooling(entity_state_turn_one), context_entity_state, entity_act_state], dim=-1)

                    final_act_state_lst.append(final_act_state_turn)
                    final_entity_state_lst.append(final_entity_state_turn)

                    entity_state_turn_one_lst = []  #reset
                    act_state_turn_one_lst = []  #reset

            final_act_state = torch.stack(final_act_state_lst, dim=1)
            final_entity_state = torch.stack(final_entity_state_lst, dim=1)
            _, act_hidden = self.gru_act(final_act_state)
            _, entity_hidden = self.gru_entity(final_entity_state)
            act_hidden = self.w(act_hidden.reshape(1, -1))
            entity_hidden = self.w(entity_hidden.reshape(1, -1))

            entity_hiddens_batch.append(entity_hidden)
            if target_kg_entity_idx[i] != []:
                entity_hiddens.append(entity_hidden.repeat(len(target_kg_entity_idx[i]), 1))
                pos_embeds.append(entity_embeds[target_kg_entity_idx[i]])
                neg_embeds.append(entity_embeds[neg_kg_entity_idx[i]])

            # act loss
            act_scores = self.sigmoid(self.act_mlp(act_hidden))
            act_loss_tmp = self.bceloss(act_scores, act_labels[i:i+1])
            act_loss += act_loss_tmp
            act_scores_batch.append(act_scores)

            # entity loss (for meddg 160)
            if entity_labels != None:
                entity_scores = self.sigmoid(self.entity_mlp(entity_hidden))
                entity_loss_tmp = self.bceloss(entity_scores, entity_labels[i:i+1])
                entity_loss += entity_loss_tmp
                entity_scores_batch.append(entity_scores)

        entity_loss = entity_loss / batch_size
        act_loss = act_loss / batch_size

        # entity loss (for ranking)
        if entity_hiddens != [] and entity_labels == None:
            # cosine similarity
            entity_hiddens = torch.cat(entity_hiddens, dim=0)
            pos_embeds = torch.cat(pos_embeds, dim=0)
            neg_embeds = torch.cat(neg_embeds, dim=0)
            extra_neg = torch.matmul(entity_hiddens, neg_embeds.T)

            logit_matrix = torch.cat([(entity_hiddens * pos_embeds).sum(-1).unsqueeze(1), extra_neg], dim=1)  # [B, 1 + B]
            lsm = F.log_softmax(logit_matrix, dim=1)
            entity_loss = (-1.0 * lsm[:, 0]).mean()

        loss = act_loss * self.act_weight + entity_loss * self.entity_weight
        if entity_scores_batch == []:
            return (loss, act_loss, entity_loss, 
                    (torch.cat(act_scores_batch, dim=0), act_labels, None, None), 
                    torch.cat(entity_hiddens_batch, dim=0))
        else:
            return (loss, act_loss, entity_loss, 
                    (torch.cat(act_scores_batch, dim=0), act_labels, torch.cat(entity_scores_batch, dim=0), entity_labels), 
                    None)
