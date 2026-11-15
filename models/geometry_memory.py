"""Learned reads and reference heads over causally visible authoring records.

Reference scores depend on learned queries and numeric record embeddings. The
only deterministic projection maps a selected record ID to its field-token
encoding. There are no wall-length/shape/connectivity/intersection rules here.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from data.geometry_memory import ANCHOR_SLOTS, RECORD_WIDTH, REFERENCE_WIDTH, VALUE_SLOTS, require_memory_codec
from data.geometry_tokens import BYTE_OFFSET, FIELD_TYPES
from data.learned_geometry import AUTHORING_FIELDS


def grouped_logsumexp(scores, indices, size):
    """Sum learned candidate mass for IDs sharing a next encoding token."""
    maxima = scores.new_full((*scores.shape[:-1], size), -torch.inf)
    maxima.scatter_reduce_(-1, indices, scores, reduce='amax', include_self=True)
    selected = maxima.gather(-1, indices)
    finite = torch.isfinite(scores)
    # Never form (-inf)-(-inf): masked future/padding rows must have zero,
    # finite gradients even when a token group has no visible candidates.
    shifted = scores.masked_fill(~finite, 0) - selected.masked_fill(~torch.isfinite(selected), 0)
    mass = shifted.masked_fill(~finite, 0).exp() * finite
    sums = scores.new_zeros(maxima.shape).scatter_add(-1, indices, mass)
    result = sums.clamp_min(torch.finfo(scores.dtype).tiny).log() + maxima.masked_fill(~torch.isfinite(maxima), 0)
    return result.masked_fill(~torch.isfinite(maxima), -torch.inf)


class GeometryRecordReadout(nn.Module):
    def __init__(self, hidden, codec, *, memory_read=False, record_anchors=False, relative_pointers=False,
                 spatial_pointers=False, dependency_memory=False, record_values=False):
        super().__init__()
        require_memory_codec(codec)
        self.hidden, self.vocab_size = hidden, codec.vocab_size
        self.unordered_memory = getattr(codec, 'unordered_memory', False)
        self.relative_references = getattr(codec, 'relative_references', False)
        self.kind_embedding = nn.Embedding(5, 8)
        self.register_buffer('value_scales', torch.tensor([1., 16., 256., 4096.]))
        self.encoder = nn.Sequential(nn.Linear(79, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.readout = nn.Linear(hidden, codec.vocab_size, bias=False)
        self.pointer_query = nn.Linear(hidden, hidden, bias=False)
        self.pointer_key = nn.Linear(hidden, hidden, bias=False)
        self.empty_memory = nn.Parameter(torch.zeros(hidden))
        self.null_reference = nn.Parameter(torch.zeros(hidden))
        indices = torch.arange(65536)
        first = BYTE_OFFSET + (indices & 255)
        for value, token in codec.integer_ids.items():
            first[value] = token
        self.register_buffer('first_tokens', first, persistent=False)
        scalar_slots = torch.zeros(len(AUTHORING_FIELDS) + 1, VALUE_SLOTS, dtype=torch.bool)
        signed_slots = torch.zeros(len(AUTHORING_FIELDS) + 1, VALUE_SLOTS, dtype=torch.bool)
        for section, name in enumerate(AUTHORING_FIELDS):
            for slot, kind in enumerate(FIELD_TYPES[name]):
                scalar_slots[section, slot] = kind in {'signed', 'unsigned'}
                signed_slots[section, slot] = kind == 'signed'
        self.register_buffer('scalar_slots', scalar_slots, persistent=False)
        self.register_buffer('signed_slots', signed_slots, persistent=False)
        self.memory_read = memory_read
        if memory_read:
            # Append after all v1 parameters and start at the identical v1
            # function. The extra learned path can turn retrieved geometry
            # into a reference query without changing the legacy head.
            self.pointer_read = nn.Linear(hidden, hidden, bias=False)
            nn.init.zeros_(self.pointer_read.weight)
        self.record_anchors = record_anchors
        if record_anchors:
            self.anchor_read = nn.Linear(ANCHOR_SLOTS * (hidden + 1), hidden, bias=False)
            nn.init.zeros_(self.anchor_read.weight)
        self.relative_pointers = relative_pointers
        if relative_pointers:
            # Zero initialization preserves the v4 function and constructor RNG.
            self.relative_pointer = nn.Parameter(torch.zeros(hidden, 9))
        self.spatial_pointers = spatial_pointers
        if spatial_pointers:
            self.spatial_pointer = nn.Parameter(torch.zeros(hidden, 18))
        self.dependency_memory = dependency_memory
        if dependency_memory:
            # Sector -> side; v1/v2/right/left -> wall. These are the file's
            # already emitted references, not inferred spatial relationships.
            # Zero initialization preserves the previous function and RNG.
            self.dependency_messages = nn.Parameter(torch.zeros(5, hidden + 1, hidden))
        self.record_values = record_values
        if record_values:
            # Completed scalar fields of the current partial record. Zero
            # initialization preserves the previous function and constructor RNG.
            self.record_value = nn.Parameter(torch.zeros(VALUE_SLOTS * 10, hidden))

    def encode_records(self, records):
        if records.ndim != 3 or records.shape[-1] != RECORD_WIDTH:
            raise ValueError('record memory must have shape [batch, records, 9]')
        kinds = records[:, :, 0].long()
        raw = records[:, :, 2:].to(self.value_scales.dtype)
        angles = raw.unsqueeze(-1) / self.value_scales
        values = torch.cat([raw / 32768., angles.sin().flatten(-2), angles.cos().flatten(-2)], -1)
        indices = records[:, :, 1].to(raw.dtype).unsqueeze(-1) / self.value_scales
        position = torch.cat([indices.sin(), indices.cos()], -1)
        return self.encoder(torch.cat([self.kind_embedding(kinds.clamp(0, 4)), position, values], -1))

    def dependency_values(self, values, records, queries, before):
        """Gather parents only when they precede the child in emission order.

        Full teacher banks can include future rows with arbitrary attributes.
        Missing/forward parents in such rows contribute zero, not information
        about a later prefix. Real emitted records obey the codec's reference
        grammar. No action/physical-connectivity inference is performed here.
        """
        batch, count, slots, _ = queries.shape
        sentinel = torch.iinfo(torch.long).max
        keys = records[:, :, 0].long() * (1 << 32) + records[:, :, 1].long()
        keys = keys.masked_fill(records[:, :, 0] < 0, sentinel)
        keys, order = keys.sort(-1)
        keys = torch.cat([keys, keys.new_full((batch, 1), sentinel)], -1)
        order = torch.cat([order, order.new_full((batch, 1), records.shape[1])], -1)
        kind, index = queries[:, :, :, 0], queries[:, :, :, 1]
        present = (kind >= 0) & (index >= 0) & (index < 65536)
        sought = (kind * (1 << 32) + index).masked_fill(~present, sentinel)
        positions = torch.searchsorted(keys.contiguous(), sought.reshape(batch, -1).contiguous())
        found = keys.gather(1, positions).reshape_as(sought)
        rows = order.gather(1, positions).reshape_as(sought)
        present = present & (found == sought) & (rows < before[:, :, None])
        values = torch.cat([values, values.new_zeros((batch, 1, self.hidden))], 1)
        parents = values.gather(1, rows.reshape(batch, -1, 1).expand(-1, -1, self.hidden))
        parents = parents.reshape(batch, count, slots, self.hidden).masked_fill(~present[:, :, :, None], 0)
        return torch.cat([parents, present.to(values.dtype)[:, :, :, None]], -1)

    def prepare_records(self, records, *, history=None, encoded_history=None):
        """Encode a full bank, or append records using the immutable past cache.

        The side and wall passes traverse only declared binary dependencies.
        A side reads its sector; a wall reads its vertices and enriched sides.
        No record can read itself or a future child/parent. Old cached embeddings
        never change, making training and incremental inference equivalent.
        """
        encoded = self.encode_records(records)
        if self.dependency_memory:
            if history is None:
                if encoded_history is not None:
                    raise ValueError('dependency cache requires its past record bank')
                history = records[:, :0]
                previous = encoded[:, :0]
            else:
                if (history.ndim != 3 or history.shape[0] != records.shape[0] or history.shape[2] != RECORD_WIDTH
                        or encoded_history is None or len(encoded_history) != 3
                        or any(part.shape != (*history.shape[:2], self.hidden) for part in encoded_history)):
                    raise ValueError('dependency cache must match its past record bank')
                previous = encoded_history[0]
            bank = torch.cat([history, records], 1)
            before = torch.arange(records.shape[1], device=records.device)[None] + history.shape[1]
            before = before.expand(records.shape[0], -1)
            queries = records.new_full((*records.shape[:2], 5, 2), -1, dtype=torch.long)
            sides, walls = records[:, :, 0] == 2, records[:, :, 0] == 3
            queries[:, :, 0, 0] = torch.where(sides, 1, -1)
            queries[:, :, 0, 1] = records[:, :, 7].long()  # side.sector
            parents = self.dependency_values(torch.cat([previous, encoded], 1), bank, queries, before)
            encoded = encoded + torch.einsum('bnsi,sih->bnh', parents, self.dependency_messages)
            queries.fill_(-1)
            for slot, field, kind in ((1, 2, 0), (2, 3, 0), (3, 7, 2), (4, 8, 2)):
                queries[:, :, slot, 0] = torch.where(walls, kind, -1)
                queries[:, :, slot, 1] = records[:, :, field].long()
            parents = self.dependency_values(torch.cat([previous, encoded], 1), bank, queries, before)
            encoded = encoded + torch.einsum('bnsi,sih->bnh', parents, self.dependency_messages)
        return encoded, self.key(encoded), self.pointer_key(encoded)

    def read_anchors(self, encoded, records, features, anchors):
        """Look up already chosen IDs; learned layers choose the next reference."""
        values, present = self.gather_anchors(encoded, records, features, anchors)
        return torch.cat([values, present.to(values.dtype)[:, :, :, None]], -1).flatten(-2)

    def gather_anchors(self, encoded, records, features, anchors):
        """Generic causal lookup for encoded or raw emitted-record attributes."""
        batch, length, _ = features.shape
        width = encoded.shape[-1]
        if anchors is None or anchors.shape != (batch, length, ANCHOR_SLOTS, 2):
            raise ValueError('anchor model requires causal partial-record reference slots')
        kind, index = anchors[:, :, :, 0].long(), anchors[:, :, :, 1].long()
        present = (kind >= 0) & (index >= 0)
        counts = features[:, :, 1:].gather(-1, kind.clamp(0, 4))
        if torch.any(present & ((kind > 2) | (index >= counts))):
            raise ValueError('anchor refers to a record not yet emitted')
        sentinel = torch.iinfo(torch.long).max
        keys = records[:, :, 0].long() * (1 << 32) + records[:, :, 1].long()
        keys = keys.masked_fill(records[:, :, 0] < 0, sentinel)
        if self.unordered_memory:
            keys, order = keys.sort(-1)
            encoded = encoded.gather(1, order[:, :, None].expand(-1, -1, width))
        elif torch.any(keys[:, 1:] < keys[:, :-1]):
            raise ValueError('anchor memory requires records in authoring order')
        keys = torch.cat([keys, keys.new_full((batch, 1), sentinel)], -1)
        sought = (kind * (1 << 32) + index).masked_fill(~present, sentinel)
        positions = torch.searchsorted(keys.contiguous(), sought.reshape(batch, -1).contiguous())
        found = keys.gather(-1, positions).reshape_as(sought)
        if torch.any(present & (found != sought)):
            raise ValueError('anchor record is missing from emitted memory')
        values = torch.cat([encoded, encoded.new_zeros((batch, 1, width))], 1)
        values = values.gather(1, positions[:, :, None].expand(-1, -1, width))
        values = values.reshape(batch, length, ANCHOR_SLOTS, width).masked_fill(~present[:, :, :, None], 0)
        return values, present

    def spatial_scores(self, query, records, features, anchors, references):
        """Learned relative XY basis for candidate vertices and a known vertex.

        No distance preference, self-edge prohibition, angle, length or closure
        rule is imposed. The score is zero until learned coefficients change.
        """
        coordinates = records[:, :, 2:4].to(query.dtype)
        values, present = self.gather_anchors(coordinates, records, features, anchors)
        known = present & (anchors[:, :, :, 0] == 0)
        first = known.long().argmax(-1)
        origin = values.gather(2, first[:, :, None, None].expand(-1, -1, 1, 2)).squeeze(2)
        delta = coordinates[:, None] - origin[:, :, None]
        angles = delta[:, :, :, :, None] / self.value_scales
        basis = torch.cat([delta[:, :, :, :, None] / 65536., angles.sin(), angles.cos()], -1).flatten(-2)
        scores = torch.einsum('btnf,btf->btn', basis, query @ self.spatial_pointer)
        active = known.any(-1) & (references[:, :, 0] == 0)
        return scores.masked_fill(~(active[:, :, None] & (records[:, None, :, 0] == 0)), 0)

    def encode_record_values(self, record_values, features):
        """Encode completed scalar fields of the current partial record.

        Only scalar slots already completed before the current prefix position
        may be marked present. Half-read byte escapes, unknown future fields,
        names and reference IDs are rejected if marked present and masked to
        zero when absent.
        """
        batch, length, _ = features.shape
        if record_values is None or record_values.shape != (batch, length, VALUE_SLOTS, 2):
            raise ValueError('record-value model requires causal partial-record scalar slots')
        presence = record_values[:, :, :, 1]
        if torch.any((presence != 0) & (presence != 1)):
            raise ValueError('record-value presence bits must be 0 or 1')
        present = presence == 1
        section = (features[:, :, 0].long() // 64).clamp(0, len(AUTHORING_FIELDS))
        position = features[:, :, 0].long() % 64
        field_index = torch.where(position > 0, (position - 1) // 9, 0)
        slot_indices = torch.arange(VALUE_SLOTS, device=features.device)[None, None, :]
        allowed_slots = (position[:, :, None] > 0) & (slot_indices < field_index[:, :, None]) & self.scalar_slots[section]
        if torch.any(present & ~allowed_slots):
            raise ValueError('record value exposes an incomplete, future or non-scalar field')
        raw = record_values[:, :, :, 0].to(self.value_scales.dtype)
        signed = self.signed_slots[section]
        low_bound = torch.where(signed, -32768., 0.)
        high_bound = torch.where(signed, 32767., 65535.)
        if torch.any(present & ((raw < low_bound) | (raw > high_bound))):
            raise ValueError('record-value scalar is out of 16-bit authoring range')
        raw = raw.masked_fill(~present, 0)
        angles = raw.unsqueeze(-1) / self.value_scales
        encoded = torch.cat([
            raw.unsqueeze(-1) / 32768.,
            angles.sin(),
            angles.cos(),
            present.to(raw.dtype).unsqueeze(-1),
        ], -1)
        return encoded.masked_fill(~present[:, :, :, None], 0).flatten(-2)

    def forward(self, hidden, logits, features, records, references, *, encoded=None, anchors=None,
                record_values=None, chunk_size=256):
        batch, length, _ = hidden.shape
        if (records.ndim != 3 or records.shape[0] != batch or records.shape[-1] != RECORD_WIDTH
                or references.shape != (batch, length, REFERENCE_WIDTH)):
            raise ValueError('causal records/reference queries do not match geometry logits')
        if encoded is None:
            encoded = self.prepare_records(records)
        if len(encoded) != 3 or any(part.shape != (batch, records.shape[1], self.hidden) for part in encoded):
            raise ValueError('encoded record memory shape is incompatible')
        encoded, keys, pointer_keys = encoded
        if self.record_anchors:
            anchor_values = self.read_anchors(encoded, records, features, anchors)
        elif anchors is not None:
            raise ValueError('this memory model does not consume partial-record anchors')
        if self.record_values:
            scalar_values = self.encode_record_values(record_values, features)
        elif record_values is not None:
            raise ValueError('this memory model does not consume partial-record scalar values')
        kinds, ids = records[:, :, 0].long(), records[:, :, 1].long()
        chunks = []
        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)
            counts = features[:, start:end, 1:]
            limits = counts.gather(-1, kinds.clamp(0, 4)[:, None].expand(-1, end-start, -1))
            visible = ((kinds[:, None] >= 0) & (ids[:, None] >= 0) & (ids[:, None] < limits))
            query = self.query(hidden[:, start:end])
            scores = torch.einsum('bth,bnh->btn', query, keys) / math.sqrt(self.hidden)
            empty = torch.einsum('bth,h->bt', query, self.empty_memory).unsqueeze(-1) / math.sqrt(self.hidden)
            weights = torch.cat([scores.masked_fill(~visible, -torch.inf), empty], -1).softmax(-1)
            context = torch.einsum('btn,bnh->bth', weights[:, :, :-1], encoded)
            context = context + weights[:, :, -1:] * self.empty_memory
            values = logits[:, start:end] + self.readout(context)
            if self.record_values:
                value_context = scalar_values[:, start:end] @ self.record_value
                values = values + self.readout(value_context)

            reference = references[:, start:end]
            active = reference[:, :, 0] >= 0
            nullable = reference[:, :, 3].bool()
            byte = reference[:, :, 1]
            low = reference[:, :, 2]
            encoded_ids = ((limits - 1 - ids[:, None]) if self.relative_references else
                           ids[:, None].expand(-1, end - start, -1)).clamp(0, 65535)
            candidates = visible & (kinds[:, None] == reference[:, :, :1])
            candidates &= (byte[:, :, None] == 0) | ((encoded_ids & 255) == low[:, :, None])
            null_allowed = nullable & ((byte == 0) | (low == 255))
            if torch.any(active & ~candidates.any(-1) & ~null_allowed):
                raise ValueError('reference query has no visible emitted record')
            pointer_query = self.pointer_query(hidden[:, start:end])
            if self.memory_read:
                pointer_query = pointer_query + self.pointer_read(context)
            if self.record_anchors:
                pointer_query = pointer_query + self.anchor_read(anchor_values[:, start:end])
            if self.record_values:
                pointer_query = pointer_query + value_context
            pointer_scores = torch.einsum('bth,bnh->btn', pointer_query, pointer_keys) / math.sqrt(self.hidden)
            if self.relative_pointers:
                # Relative emitted-list index, not spatial distance or an
                # endpoint preference. Every legal candidate remains available.
                age = (limits - ids[:, None] - 1).to(pointer_query.dtype)
                angles = age[:, :, :, None] / self.value_scales
                position = torch.cat([age[:, :, :, None] / 65536., angles.sin(), angles.cos()], -1)
                coefficients = pointer_query @ self.relative_pointer
                pointer_scores = pointer_scores + torch.einsum('btnf,btf->btn', position, coefficients)
            if self.spatial_pointers:
                pointer_scores = pointer_scores + self.spatial_scores(pointer_query, records, features[:, start:end],
                                                                     anchors[:, start:end], reference)
            null_score = torch.einsum('bth,h->bt', pointer_query, self.null_reference).unsqueeze(-1) / math.sqrt(self.hidden)
            pointer_scores = torch.cat([pointer_scores.masked_fill(~candidates, -torch.inf),
                                       null_score.masked_fill(~(null_allowed | ~active).unsqueeze(-1), -torch.inf)], -1)
            all_ids = torch.cat([encoded_ids, ids.new_full((batch, end - start, 1), 65535)], -1)
            first = self.first_tokens[all_ids]
            high = BYTE_OFFSET + (all_ids >> 8)
            encoding = torch.where(byte[:, :, None] == 0, first, high)
            pointers = grouped_logsumexp(pointer_scores, encoding, self.vocab_size)
            chunks.append(torch.where(active[:, :, None], pointers, values))
        return torch.cat(chunks, 1)
