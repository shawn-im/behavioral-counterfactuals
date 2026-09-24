from __future__ import annotations

"""Shared conditional-support output projection for training and analysis."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchedSupportOnlyOutputHead:
    """Project response positions only onto each record's cached support."""

    def __init__(
        self,
        output_embeddings: nn.Module,
        support_ids: list[torch.Tensor],
        *,
        prompt_lengths: list[int],
        response_lengths: list[int],
        chunk_positions: int = 2048,
    ):
        if not isinstance(output_embeddings, nn.Linear):
            raise TypeError(
                "batched support-only projection requires a linear output head; "
                f"got {type(output_embeddings).__name__}"
            )
        if not (
            len(support_ids) == len(prompt_lengths) == len(response_lengths)
        ):
            raise ValueError("support/prompt/response batch sizes must match")
        self.module = output_embeddings
        self.support_ids = [value.detach().long().cpu() for value in support_ids]
        self.prompt_lengths = [int(value) for value in prompt_lengths]
        self.response_lengths = [int(value) for value in response_lengths]
        self.chunk_positions = max(1, int(chunk_positions))
        self.original_forward = None

    def _forward(self, hidden: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if args or kwargs:
            raise TypeError("support-only output projection expects hidden states only")
        if hidden.ndim != 3:
            raise ValueError(
                "support-only output projection expects [batch, sequence, hidden], "
                f"got {tuple(hidden.shape)}"
            )
        batch, sequence, hidden_size = hidden.shape
        if int(batch) != len(self.support_ids):
            raise ValueError("LM-head batch size does not match support batch")
        if int(self.module.weight.shape[1]) != int(hidden_size):
            raise ValueError("LM-head hidden size does not match model hidden states")

        max_width = max(
            (int(value.shape[1]) for value in self.support_ids), default=0
        )
        max_response = max(self.response_lengths, default=0)
        if max_width <= 0 or max_response <= 0:
            return hidden.new_zeros((batch, max_response, max_width))

        batch_index_parts = []
        sequence_index_parts = []
        support_parts = []
        width_parts = []
        for batch_index, (ids, prompt_len, response_len) in enumerate(
            zip(self.support_ids, self.prompt_lengths, self.response_lengths)
        ):
            if int(ids.shape[0]) != int(response_len):
                raise ValueError("support rows do not match response length")
            first = int(prompt_len) - 1
            last = first + int(response_len)
            if first < 0 or last > int(sequence):
                raise ValueError(
                    "response prediction positions are outside the LM-head sequence: "
                    f"[{first}, {last}) vs sequence={sequence}"
                )
            width = int(ids.shape[1])
            if width < max_width:
                ids = F.pad(ids, (0, max_width - width), value=0)
            batch_index_parts.append(
                torch.full((response_len,), batch_index, dtype=torch.long)
            )
            sequence_index_parts.append(torch.arange(first, last, dtype=torch.long))
            support_parts.append(ids)
            width_parts.append(
                torch.full((response_len,), width, dtype=torch.long)
            )

        if not support_parts:
            return hidden.new_zeros((batch, max_response, max_width))

        batch_indices = torch.cat(batch_index_parts).to(
            hidden.device, non_blocking=True
        )
        sequence_indices = torch.cat(sequence_index_parts).to(
            hidden.device, non_blocking=True
        )
        packed_support = torch.cat(support_parts, dim=0).to(
            self.module.weight.device, non_blocking=True
        )
        widths = torch.cat(width_parts).to(hidden.device, non_blocking=True)
        selected_hidden = hidden[batch_indices, sequence_indices]

        total = int(packed_support.shape[0])
        flat_output = hidden.new_zeros((total, max_width))
        weight = self.module.weight
        bias = self.module.bias
        columns = torch.arange(max_width, device=hidden.device).unsqueeze(0)
        for start in range(0, total, self.chunk_positions):
            end = min(start + self.chunk_positions, total)
            ids = packed_support[start:end]
            selected_weight = weight.index_select(0, ids.reshape(-1)).reshape(
                end - start, max_width, hidden_size
            )
            values = torch.einsum(
                "cd,ckd->ck", selected_hidden[start:end], selected_weight
            )
            if bias is not None:
                selected_bias = bias.index_select(0, ids.reshape(-1)).reshape(
                    end - start, max_width
                )
                values = values + selected_bias
            valid = columns < widths[start:end].unsqueeze(1)
            flat_output[start:end] = values.masked_fill(~valid, 0.0)

        output = hidden.new_zeros((batch, max_response, max_width))
        offset = 0
        for batch_index, response_len in enumerate(self.response_lengths):
            next_offset = offset + int(response_len)
            output[batch_index, :response_len] = flat_output[offset:next_offset]
            offset = next_offset
        return output

    def __enter__(self):
        self.original_forward = self.module.forward
        self.module.forward = self._forward
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.original_forward is not None:
            self.module.forward = self.original_forward
            self.original_forward = None
