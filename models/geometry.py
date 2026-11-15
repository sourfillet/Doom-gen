"""Shared autoregressive model of complete variable-length map records.

This has no source slots/scaffold or per-map geometry parameters. Its next-token
distribution owns list lengths, coordinates, incidence, materials and actions.
Syntax masks only describe the reversible binary-record contract.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from data.geometry_tokens import BOS, END_LIST, PAD, RECORD
from data.learned_geometry import validate_conditioning

CHECKPOINT_KIND = 'geometry'


def condition_tags(context):
    """Namespace explicit game/tag context for the existing TagConditioner."""
    validate_conditioning(context)
    tags = {f'tag:{name}': weight for name, weight in context['tags'].items()}
    if context['target_game']:
        tags[f'game:{context["target_game"]}'] = 1.0
    return tags


class GeometryRNN(nn.Module):
    def __init__(self, hidden=128, embedding=64, layers=2, context_dim=64, prefix_features=False,
                 codec_config=None, record_memory=False, memory_read=False, record_anchors=False,
                 relative_pointers=False, spatial_pointers=False, dependency_memory=False,
                 record_values=False):
        super().__init__()
        if min(hidden, embedding, layers, context_dim) < 1:
            raise ValueError('geometry model dimensions must be positive')
        self.hidden, self.layers = hidden, layers
        self.prefix_features = prefix_features
        from data.geometry_codec import codec_for
        self.codec = codec_for(codec_config)
        self.vocab_size = self.codec.vocab_size
        self.embedding = nn.Embedding(self.vocab_size, embedding, padding_idx=PAD)
        self.initial = nn.Linear(context_dim, hidden * layers)
        if prefix_features:
            self.format_embedding = nn.Embedding(self.codec.prefix_states, 16)
            self.register_buffer('count_scales', torch.tensor([1.0, 10.0, 100.0, 1000.0]))
        self.rnn = nn.GRU(embedding + context_dim + (56 if prefix_features else 0), hidden, layers, batch_first=True)
        self.output = nn.Linear(hidden, self.vocab_size)
        self.record_memory = record_memory
        self.record_anchors = record_anchors
        self.record_values = record_values
        if memory_read and not record_memory:
            raise ValueError('memory read requires record memory')
        if record_anchors and not memory_read:
            raise ValueError('record anchors require the memory-read model')
        if relative_pointers and not record_memory:
            raise ValueError('relative pointers require record memory')
        if spatial_pointers and not record_anchors:
            raise ValueError('spatial pointers require record anchors')
        if dependency_memory and (not spatial_pointers or not getattr(self.codec, 'unordered_memory', False)):
            raise ValueError('dependency memory requires interleaved tokens and spatial pointers')
        if record_values and not record_anchors:
            raise ValueError('record values require record anchors')
        if record_memory:
            if not prefix_features:
                raise ValueError('record memory requires causal prefix features')
            from models.geometry_memory import GeometryRecordReadout
            self.memory_readout = GeometryRecordReadout(hidden, self.codec, memory_read=memory_read,
                                                       record_anchors=record_anchors,
                                                       relative_pointers=relative_pointers,
                                                       spatial_pointers=spatial_pointers,
                                                       dependency_memory=dependency_memory,
                                                       record_values=record_values)

    def initial_state(self, context):
        return torch.tanh(self.initial(context)).view(-1, self.layers, self.hidden).transpose(0, 1).contiguous()

    def forward(self, tokens, context, state=None, features=None, *, memory=None, references=None,
                encoded_memory=None, anchors=None, record_values=None):
        embedding = self.embedding(tokens)
        values = torch.cat([embedding, context[:, None].expand(-1, tokens.shape[1], -1)], dim=-1)
        if self.prefix_features:
            if features is None or features.shape != (*tokens.shape, 6):
                raise ValueError('geometry prefix features must come from each causal token prefix')
            positions = self.format_embedding(features[:, :, 0])
            angles = features[:, :, 1:].to(values.dtype).unsqueeze(-1) / self.count_scales
            counts = torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)
            values = torch.cat([values, positions, counts], dim=-1)
        hidden, state = self.rnn(values, self.initial_state(context) if state is None else state)
        logits = self.output(hidden)
        if self.record_memory:
            if memory is None or references is None:
                raise ValueError('record-memory model requires its causal emitted-record bank and queries')
            logits = self.memory_readout(hidden, logits, features, memory, references,
                                         encoded=encoded_memory, anchors=anchors, record_values=record_values)
        elif (memory is not None or references is not None or encoded_memory is not None
              or anchors is not None or record_values is not None):
            raise ValueError('legacy geometry model does not consume record memory')
        return logits, state

    def loss(self, tokens, context, allowed, features=None, *, stop_weight=1.0, memory=None, references=None,
             anchors=None, record_values=None, loss_groups=None, ending_loss_weight=0.0):
        """Map-balanced next-token NLL; forced syntax does not inflate accuracy.

        ``allowed`` is built only from each token prefix. Padding is excluded;
        masked positions with a single legal symbol carry no learning claim.
        An explicit stop_weight can upweight rare learned list endings; it
        supplies no target record counts or inference stopping rules.
        """
        if isinstance(stop_weight, bool) or not math.isfinite(stop_weight) or stop_weight <= 0:
            raise ValueError('stop weight must be finite and positive')
        if isinstance(ending_loss_weight, bool) or not math.isfinite(ending_loss_weight) or ending_loss_weight < 0:
            raise ValueError('ending loss weight must be finite and nonnegative')
        if ending_loss_weight and not hasattr(self.codec, 'continuation_tokens'):
            raise ValueError('ending auxiliary loss requires interleaved headers')
        if not self.record_memory and (memory is not None or references is not None
                                       or anchors is not None or record_values is not None):
            raise ValueError('legacy geometry model does not consume record memory')
        kwargs = {'memory': memory, 'references': references} if self.record_memory else {}
        if anchors is not None:
            kwargs['anchors'] = anchors
        if record_values is not None:
            kwargs['record_values'] = record_values
        logits, _ = self(tokens[:, :-1], context, features=features, **kwargs)
        targets = tokens[:, 1:]
        if allowed.shape != logits.shape or allowed.dtype != torch.bool:
            raise ValueError('allowed token masks must match the causal logits')
        valid = targets != PAD
        if not torch.all(allowed.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[valid]):
            raise ValueError('supervised token is illegal under its own prefix')
        # Padding rows need a finite logit to keep softmax/backward well defined.
        masks = allowed.clone()
        masks[:, :, PAD] |= ~valid
        masked = logits.masked_fill(~masks, -torch.inf)
        losses = F.cross_entropy(masked.transpose(1, 2), targets, reduction='none', ignore_index=PAD)
        decisions = (allowed.sum(-1) > 1) & valid
        correct = masked.argmax(-1) == targets
        if hasattr(self.codec, 'continuation_tokens'):
            choices = list(self.codec.continuation_tokens)
            endings = allowed[:, :, choices].any(-1) & allowed[:, :, self.codec.stop_token] & valid
            stops = endings & (targets == self.codec.stop_token)
            continuations = endings & (targets != self.codec.stop_token)
        else:
            endings = allowed[:, :, RECORD] & allowed[:, :, END_LIST] & valid
            stops = endings & (targets == END_LIST)
            continuations = endings & (targets == RECORD)
        if loss_groups is not None:
            if stop_weight != 1.0:
                raise ValueError('field-balanced loss requires stop weight 1; boundary classes are already balanced')
            from models.geometry_objective import balanced_loss
            loss = balanced_loss(losses, decisions, loss_groups)
        elif stop_weight == 1.0:
            # Keep the original default objective and reduction exactly.
            lengths = decisions.sum(-1).clamp_min(1)
            loss = ((losses * decisions).sum(-1) / lengths).mean()
        else:
            weights = decisions.to(losses.dtype) * torch.where(stops, stop_weight, 1.0)
            loss = ((losses * weights).sum(-1) / weights.sum(-1).clamp_min(1)).mean()

        if ending_loss_weight:
            # The conditional binary hazard shares the actual next-token
            # logits. Equal class means prevent hundreds of continuation
            # headers from drowning the single global ending. The normal
            # token/type/value objective is retained; inference is unchanged.
            # Select eligible rows BEFORE logsumexp/subtraction: unavailable
            # EOS/continuation logits elsewhere can be -inf or forced syntax.
            terms = []
            for batch in range(tokens.shape[0]):
                selected = masked[batch, endings[batch]]
                gold = stops[batch, endings[batch]]
                if not gold.any():
                    raise ValueError('ending auxiliary loss requires a complete global ending per map')
                score = selected[:, self.codec.stop_token] - torch.logsumexp(selected[:, choices], -1)
                binary = F.binary_cross_entropy_with_logits(score, gold.to(score.dtype), reduction='none')
                classes = [binary[gold].mean()]
                if (~gold).any():
                    classes.append(binary[~gold].mean())
                terms.append(torch.stack(classes).mean())
            ending_auxiliary = torch.stack(terms).mean()
            loss = loss + ending_loss_weight * ending_auxiliary

        def accuracy(selection):
            return (correct & selection).sum() / selection.sum().clamp_min(1)

        raw = F.cross_entropy(logits.transpose(1, 2), targets, reduction='none', ignore_index=PAD)
        metrics = {'decision_accuracy': accuracy(decisions),
                   'ending_accuracy': accuracy(endings),
                   'list_stop_recall': accuracy(stops),
                   'list_continue_recall': accuracy(continuations),
                   'list_stop_count': stops.sum(),
                   'list_continue_count': continuations.sum(),
                   'value_accuracy': accuracy(decisions & ~endings),
                   'forced_fraction': ((~decisions) & valid).sum() / valid.sum().clamp_min(1),
                   'unmasked_nll': (raw * valid).sum() / valid.sum().clamp_min(1)}
        if hasattr(self.codec, 'continuation_tokens'):
            # Type prediction is a separate categorical task; continuation
            # recall asks whether any record type was chosen instead of EOS.
            headers = allowed[:, :, choices].any(-1) & valid
            guesses = masked.argmax(-1)
            binary_correct = (guesses == self.codec.stop_token) == (targets == self.codec.stop_token)
            metrics['record_type_accuracy'] = accuracy(headers & (targets != self.codec.stop_token))
            metrics['ending_accuracy'] = (binary_correct & endings).sum() / endings.sum().clamp_min(1)
            metrics['list_continue_recall'] = ((guesses != self.codec.stop_token) & continuations).sum() / continuations.sum().clamp_min(1)
            metrics['value_accuracy'] = accuracy(decisions & ~headers)
        if ending_loss_weight:
            metrics['ending_auxiliary_loss'] = ending_auxiliary
        return loss, {name: value.detach() for name, value in metrics.items()}

    @torch.no_grad()
    def sample(self, context, *, seed=0, max_tokens=4096, temperature=1.0, syntax_masks=True):
        """Free-running inference from context alone, retaining incomplete streams.

        No teacher prefix/target, requested source counts or source coordinates
        enter this API. Budgets reject; they never synthesize missing endings.
        """
        if context.ndim != 2 or context.shape[0] != 1:
            raise ValueError('sampling requires one context vector')
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 2:
            raise ValueError('max_tokens must be an integer of at least two')
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError('temperature must be finite and nonnegative')
        self.eval()
        parser = self.codec.state()
        parser.push(BOS)
        memory_builder, bank, encoded_bank = None, None, None
        if self.record_memory:
            from data.geometry_memory import (RecordMemory, current_record_anchors,
                                              current_record_values, reference_query)
            memory_builder = RecordMemory()
            bank = torch.empty((1, 0, 9), dtype=torch.long, device=context.device)
            encoded_bank = self.memory_readout.prepare_records(bank)
        tokens, state, forced = [BOS], None, 0
        generator = torch.Generator(device=context.device).manual_seed(seed)
        report = {'schema': 'doom_geometry_token_sample_v1', 'seed': seed,
                  'temperature': temperature, 'max_tokens': max_tokens,
                  'syntax_masks': syntax_masks, 'status': 'budget_exhausted',
                  'tokens': tokens, 'forced_tokens': 0}
        while len(tokens) < max_tokens:
            features = (torch.tensor([[parser.features()]], device=context.device)
                        if self.prefix_features else None)
            kwargs = {}
            if self.record_memory:
                previous = len(memory_builder.rows)
                memory_builder.observe(parser)
                if len(memory_builder.rows) > previous:
                    new = torch.tensor([memory_builder.rows[previous:]], dtype=torch.long, device=context.device)
                    addition = (self.memory_readout.prepare_records(new, history=bank, encoded_history=encoded_bank)
                                if self.memory_readout.dependency_memory else self.memory_readout.prepare_records(new))
                    bank = torch.cat([bank, new], 1)
                    encoded_bank = tuple(torch.cat([before, after], 1) for before, after in zip(encoded_bank, addition))
                kwargs = {'memory': bank, 'encoded_memory': encoded_bank,
                          'references': torch.tensor([[reference_query(parser)]], device=context.device)}
                if self.record_anchors:
                    kwargs['anchors'] = torch.tensor([[current_record_anchors(parser)]], device=context.device)
                if self.record_values:
                    kwargs['record_values'] = torch.tensor([[current_record_values(parser)]], device=context.device)
            logits, state = self(torch.tensor([[tokens[-1]]], device=context.device), context, state,
                                 features=features, **kwargs)
            values = logits[0, -1].clone()
            allowed = parser.allowed_tokens()
            if syntax_masks:
                mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=context.device)
                mask[list(allowed)] = True
                values.masked_fill_(~mask, -torch.inf)
                forced += len(allowed) == 1
            if not torch.isfinite(values).any() or torch.isnan(values).any() or torch.isposinf(values).any():
                report.update(status='invalid_logits', reason='no finite next-token distribution')
                break
            token = (int(values.argmax()) if temperature == 0 else
                     int(torch.multinomial(torch.softmax(values / temperature, -1), 1, generator=generator)))
            tokens.append(token)
            try:
                parser.push(token)
            except ValueError as exc:
                report.update(status='invalid_token', reason=str(exc))
                break
            if parser.done:
                report['status'] = 'complete'
                break
        report['forced_tokens'] = forced
        return report
