from copy import copy

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import RobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_roberta import RobertaLayer, RobertaPooler

from stancedetection.models.nn import LabelTransferNetwork


class GradientReversal(torch.autograd.Function):
    """
    Basic layer for doing gradient reversal
    """

    lambd = 1.0

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReversal.lambd * grad_output.neg()


class DomainAdaptationHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.encoder = RobertaLayer(config)
        self.pooler = RobertaPooler(config)

    def forward(
        self,
        hidden_states=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
    ):
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )[0]

        return self.pooler(encoder_outputs)


class MultiViewRobertaShared(RobertaForSequenceClassification):
    def __init__(self, config):
        assert config.num_domains > 0
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_domains = config.num_domains

        #         self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.adapters = nn.ModuleList(
            [DomainAdaptationHead(copy(config)) for _ in range(self.num_domains)]
        )
        self.shared_adapter = DomainAdaptationHead(copy(config))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = LabelTransferNetwork(
            self.config.label_embedding_hidden,
            num_labels=self.num_labels,
            task2labels=config.task2labels,
            return_ltn_scores=config.return_ltn_scores,
            return_ltn_loss=config.return_ltn_loss,
        )

        self.lambda_ = 0.5
        self.gamma = 0.5

        if self.config.is_domain_adversarial:
            self.num_tasks = len(config.task2labels)
            self.domain_classifier = nn.Linear(config.hidden_size, self.num_tasks)
            self.adversarial_scalar = 1e-03
            # divisor = min(1, 2 * (len(outputs[1]) - self.supervision_layer))

        self.init_weights()

    def _forward_domain_adapt(
        self,
        adapter,
        sequence_output,
        extended_attention_mask=None,
        head_mask=None,
        lel_mask=None,
        output_attentions=False,
        output_hidden=False,
    ):
        domain_adapter_output = adapter(
            sequence_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        domain_adapter_output = self.dropout(domain_adapter_output)
        ltn_outputs = self.classifier(domain_adapter_output, lel_mask)

        outputs = (ltn_outputs[0],)
        if output_hidden:
            outputs += (domain_adapter_output,)

        return outputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        domain_name=None,
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

        is_domain_adversarial = (
            domain_name is not None and self.config.is_domain_adversarial and self.training
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

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
        sequence_output = outputs[0]

        device = input_ids.device if input_ids is not None else sequence_output.device
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        shared_tuple = self._forward_domain_adapt(
            self.shared_adapter,
            sequence_output,
            extended_attention_mask=extended_attention_mask,
            head_mask=head_mask,
            lel_mask=lel_mask,
            output_attentions=output_attentions,
            output_hidden=True,
        )

        shared_logits = shared_tuple[0]
        shared_outputs = shared_tuple[1]

        tasks_logits = []
        for domain_id in range(self.num_domains):
            task_logits = self._forward_domain_adapt(
                self.adapters[domain_id],
                sequence_output,
                extended_attention_mask=extended_attention_mask,
                head_mask=head_mask,
                lel_mask=lel_mask,
                output_attentions=output_attentions,
                output_hidden=False,
            )[0]
            tasks_logits.append(task_logits)

        print(shared_logits.shape)
        all_logits = torch.stack(tasks_logits + [shared_logits], dim=1)
        print(all_logits.shape)
        all_logits = torch.mean(all_logits, dim=1)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(all_logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = (1 - self.lambda_) * loss_fct(
                    all_logits.view(-1, self.num_labels), labels.view(-1)
                )
                loss += self.gamma * loss_fct(
                    shared_logits.view(-1, self.num_labels), labels.view(-1)
                )

                if domain_name is not None:
                    target_domain_id = self.config.domain2id[domain_name]
                    if is_domain_adversarial:
                        shared_outputs = GradientReversal.apply(shared_outputs)
                        domain_adversarial_logits = self.domain_classifier(shared_outputs)

                        domain_labels = torch.LongTensor(
                            np.fromiter(
                                (self.config.label2task[label.item()] for label in labels),
                                dtype=np.long,
                            )
                        ).to(labels.device)
                        loss_domain_adv = loss_fct(
                            domain_adversarial_logits.view(-1, self.num_tasks),
                            domain_labels.view(-1),
                        )
                        loss += self.adversarial_scalar * loss_domain_adv

                    task_logits = tasks_logits[target_domain_id]
                    loss += self.lambda_ * loss_fct(
                        task_logits.view(-1, self.num_labels), labels.view(-1)
                    )

        if not return_dict:
            output = (all_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=all_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
