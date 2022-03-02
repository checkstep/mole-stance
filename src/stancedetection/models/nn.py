from copy import copy

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertModel, BertPreTrainedModel, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_roberta import RobertaClassificationHead, RobertaPreTrainedModel

def mask_tensor(tensor, mask, dtype=None):
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    # Masking the unrelated positions
    mask = (1.0 - mask) * -10000.0
    if dtype:
        mask = mask.to(dtype=dtype)  # fp16 compatibility
    tensor = tensor + mask

    return tensor

class LabelEmbeddingLayer(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()

        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.label_embedding = nn.Embedding(num_labels, hidden_size)

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5

            def find_tensor_attributes(module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def forward(self, h, mask=None, return_non_masked=False):
        # label_embedding.weight.shape = [num_labels,  hidden_size]
        label_emb = self.label_embedding.weight
        h_size = h.shape[1]

        if h_size > self.hidden_size:
            pad_size = h.shape[1] - self.hidden_size
            padding = torch.zeros((self.num_labels, pad_size)).to(self.dtype)
            label_emb = torch.cat((label_emb, padding), dim=1)

        # label_emb.shape = [1, hidden_size, num_labels]
        label_emb = label_emb.permute([1, 0]).unsqueeze(0)

        # h.shape = [batch_size, 1, hidden_size]
        h = h.unsqueeze(1)

        # scores.shape = [batch_size, num_labels]
        comb_repr = torch.matmul(h, label_emb).squeeze(1)

        scores = comb_repr
        if mask is not None:
            scores = mask_tensor(scores, mask, dtype=self.dtype)

        result = (scores,)
        if return_non_masked:
            result = result + (comb_repr,)

        return result


class LabelTransferNetwork(nn.Module):
    def __init__(
        self,
        embedding_hidden_size,
        num_labels,
        task2labels=None,
        use_ltn=True,
        return_ltn_scores=False,
        return_ltn_loss=False,
    ):
        super().__init__()
        self.use_ltn = use_ltn
        self.lel = LabelEmbeddingLayer(embedding_hidden_size, num_labels)
        self.return_ltn_scores = return_ltn_scores
        self.return_ltn_loss = return_ltn_loss
        if use_ltn:
            # This is a 2D array [num_tasks, ], where the rows contains the activate labels ids for each task
            # Should start with zero
            self.task2labels = task2labels
            self.num_tasks = len(self.task2labels)
            self.proj = nn.Linear(self.num_tasks * self.lel.hidden_size, num_labels)

    def forward(self, h, lel_mask=None):
        scores, non_masked_scores = self.lel(h, mask=lel_mask, return_non_masked=True)
        result = (scores,)

        if self.use_ltn and (self.return_ltn_loss or self.return_ltn_scores):
            task_embeddings = []
            for label_ids in self.task2labels:
                # task_label_embeddings = [batch_size, num_labels_(task)]
                task_scores = non_masked_scores[:, label_ids]
                task_weights = torch.softmax(task_scores, dim=-1)

                # task_label_embeddings = [num_labels_(task), hidden_size]
                task_label_embeddings = self.lel.label_embedding.weight[label_ids]

                # Get the weighted average for the each task
                # task_embedding.shape = [batch_size, hidden_size]
                task_embedding = torch.matmul(task_weights, task_label_embeddings)

                task_embeddings.append(task_embedding)

            task_embeddings = torch.cat(task_embeddings, dim=1)
            ltn_scores = self.proj(task_embeddings)

            if self.return_ltn_scores:
                # Mask the labels that are not relevant for the task.
                scores = mask_tensor(ltn_scores, lel_mask, dtype=self.lel.dtype)
                result = (scores,)

            if self.return_ltn_loss:
                mse_loss = torch.nn.MSELoss()
                ltn_loss = mse_loss(ltn_scores, non_masked_scores)
                result = result + (ltn_loss,)

        return result


class LTNRoberta(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = LabelTransferNetwork(
            self.config.label_embedding_hidden,
            num_labels=self.num_labels,
            task2labels=config.task2labels,
            return_ltn_scores=config.return_ltn_scores,
            return_ltn_loss=config.return_ltn_loss,
        )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        lel_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        ltn_outputs = self.classifier(pooled_output, lel_mask)
        logits = ltn_outputs[0]

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if self.classifier.return_ltn_loss:
            loss += ltn_outputs[1]

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RobertaForSequenceClassificationMTL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = BertModel(config)
        self.classifiers = torch.nn.ModuleList()
        for labels in config.task2labels:
            task_config = copy(config)
            task_config.num_labels = len(labels)
            self.classifiers.append(RobertaClassificationHead(task_config))

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        task_name=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[1]
        task_id = self.config.task2id[task_name]
        classifier = self.classifiers[task_id]
        logits = classifier(sequence_output)

        loss = None
        if labels is not None:
            num_labels = len(self.config.task2labels[task_id])
            if num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BERTForSequenceClassificationMTL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.classifiers = torch.nn.ModuleList()
        for labels in config.task2labels:
            task_config = copy(config)
            task_config.num_labels = len(labels)
            self.classifiers.append(nn.Linear(config.hidden_size, task_config.num_labels))

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        task_name=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[1]
        task_id = self.config.task2id[task_name]
        classifier = self.classifiers[task_id]
        logits = classifier(sequence_output)

        loss = None
        if labels is not None:
            num_labels = len(self.config.task2labels[task_id])
            if num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
