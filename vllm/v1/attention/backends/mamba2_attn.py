# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from dataclasses import dataclass, replace
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.config.mamba import MambaPrefillBackendEnum
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import (
    AttentionBackend,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec


@dataclass(frozen=True)
class FlashInferSSDMetadataCPU:
    """Host-side metadata for one packed Mamba2 prefill iteration."""

    seq_idx: list[int]
    chunk_indices: list[int]
    chunk_offsets: list[int]
    seq_chunk_cumsum: list[int]
    segment_seq_ids: list[int]
    segment_state_block_indices: list[int]
    valid_seqlen: int
    padded_seqlen: int


def compute_flashinfer_ssd_metadata(
    query_start_loc: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    *,
    chunk_size: int,
    mamba_block_size: int,
) -> FlashInferSSDMetadataCPU:
    """Build packed-varlen metadata for FlashInfer's Mamba2 SSD kernel.

    Logical segments are emitted in packed traversal order and never cross a
    sequence boundary, a physical ``chunk_size`` boundary, or a Mamba cache
    checkpoint. ``segment_state_block_indices`` contains the per-request block
    table column that must receive the post-segment state, or ``-1`` when the
    state is not needed by vLLM.

    Both inputs must be CPU tensors. Keeping this construction on the host
    avoids a device synchronization in the per-iteration metadata builder.
    """
    if query_start_loc.device.type != "cpu" or num_computed_tokens.device.type != "cpu":
        raise ValueError("FlashInfer SSD metadata inputs must be CPU tensors")
    if query_start_loc.ndim != 1 or num_computed_tokens.ndim != 1:
        raise ValueError("FlashInfer SSD metadata inputs must be one-dimensional")
    if query_start_loc.numel() != num_computed_tokens.numel() + 1:
        raise ValueError("query_start_loc must contain one more entry than contexts")
    if chunk_size <= 0 or mamba_block_size <= 0:
        raise ValueError("chunk and Mamba block sizes must be positive")
    if int(query_start_loc[0].item()) != 0:
        raise ValueError("query_start_loc must start at zero")

    starts = query_start_loc[:-1].tolist()
    ends = query_start_loc[1:].tolist()
    computed = num_computed_tokens.tolist()
    valid_seqlen = int(ends[-1]) if ends else 0
    padded_seqlen = (
        ((valid_seqlen + chunk_size - 1) // chunk_size) * chunk_size
        if valid_seqlen
        else 0
    )

    seq_idx: list[int] = []
    chunk_indices: list[int] = []
    chunk_offsets: list[int] = []
    seq_chunk_cumsum: list[int] = [0]
    segment_seq_ids: list[int] = []
    segment_state_block_indices: list[int] = []

    for seq_id, (packed_start, packed_end, context_len) in enumerate(
        zip(starts, ends, computed)
    ):
        packed_start = int(packed_start)
        packed_end = int(packed_end)
        context_len = int(context_len)
        query_len = packed_end - packed_start
        if query_len < 0 or context_len < 0:
            raise ValueError("sequence and context lengths must be nonnegative")

        seq_idx.extend([seq_id] * query_len)
        if query_len == 0:
            seq_chunk_cumsum.append(len(chunk_indices))
            continue

        cuts = {packed_start, packed_end}

        # Physical packed-input chunk boundaries.
        boundary = ((packed_start // chunk_size) + 1) * chunk_size
        while boundary < packed_end:
            cuts.add(boundary)
            boundary += chunk_size

        # State-cache checkpoint boundaries are relative to the original
        # sequence, not to the packed input or scheduler iteration.
        final_position = context_len + query_len
        checkpoint = ((context_len // mamba_block_size) + 1) * mamba_block_size
        while checkpoint < final_position:
            cuts.add(packed_start + checkpoint - context_len)
            checkpoint += mamba_block_size

        ordered_cuts = sorted(cuts)
        for segment_start, segment_end in zip(ordered_cuts, ordered_cuts[1:]):
            chunk_indices.append(segment_start // chunk_size)
            chunk_offsets.append(segment_start % chunk_size)
            segment_seq_ids.append(seq_id)

            endpoint_position = context_len + segment_end - packed_start
            if segment_end == packed_end:
                # The final scheduled state may be a partial cache block.
                state_block_idx = (final_position - 1) // mamba_block_size
            elif endpoint_position % mamba_block_size == 0:
                state_block_idx = endpoint_position // mamba_block_size - 1
            else:
                state_block_idx = -1
            segment_state_block_indices.append(state_block_idx)

        seq_chunk_cumsum.append(len(chunk_indices))

    # The padded tail is masked by valid_seqlen, but must still carry a valid
    # sequence id because FlashInfer accepts a physical chunk-aligned tensor.
    if padded_seqlen > valid_seqlen:
        seq_idx.extend(
            [len(starts) - 1 if starts else 0] * (padded_seqlen - valid_seqlen)
        )

    return FlashInferSSDMetadataCPU(
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        seq_chunk_cumsum=seq_chunk_cumsum,
        segment_seq_ids=segment_seq_ids,
        segment_state_block_indices=segment_state_block_indices,
        valid_seqlen=valid_seqlen,
        padded_seqlen=padded_seqlen,
    )


def compute_varlen_chunk_metadata(
    query_start_loc: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build chunk-aligned, variable-length metadata used by Mamba2 SSD kernels.

    Given per-sequence cumulative token starts `query_start_loc` of shape [B+1]
    and a physical `chunk_size`, returns three tensors on the same device:
      - cu_chunk_seqlens:  (nchunks+1,) int32   exclusive prefix-sum of
        logical-chunk lengths (each logical chunk never crosses a sequence or
        physical-chunk boundary).
      - last_chunk_indices: (B,)       int32   index of the last logical chunk
        for each sequence (=-1 for empty sequences).
      - seq_idx_chunks:     (nchunks,) int32   sequence index for each logical
        chunk in order.

    This is intentionally lightweight and CPU-side; it mirrors the metadata
    produced by the V1 Mamba2 meta-data builder and is exported so tests
    (and other callers) can avoid duplicating the logic.
    """
    assert query_start_loc.ndim == 1, "query_start_loc must be 1-D [B+1]"
    assert int(query_start_loc[0].item()) == 0, "query_start_loc[0] must be 0"
    device = query_start_loc.device

    qsl64 = query_start_loc.to(torch.int64)
    starts = qsl64[:-1].tolist()
    ends = qsl64[1:].tolist()
    total = int(qsl64[-1].item())

    chunk_lens: list[int] = []
    seq_idx_chunks: list[int] = []
    last_chunk_indices: list[int] = [-1] * len(starts)

    for b, (s, e) in enumerate(zip(starts, ends)):
        if e <= s:
            # empty sequence
            continue
        pos = s
        while pos < e:
            # split at both sequence boundaries and physical chunk boundaries
            room = chunk_size - (pos % chunk_size)
            take = min(room, e - pos)
            chunk_lens.append(int(take))
            seq_idx_chunks.append(b)
            last_chunk_indices[b] = len(chunk_lens) - 1
            pos += take

    # Exclusive prefix sum over logical-chunk lengths
    if chunk_lens:
        cu_chunk_seqlens_list = [0] + list(itertools.accumulate(chunk_lens))
        # Final boundary must equal total tokens (check on host to avoid a sync)
        assert cu_chunk_seqlens_list[-1] == total
    else:
        cu_chunk_seqlens_list = [0]
    cu_chunk_seqlens = async_tensor_h2d(
        cu_chunk_seqlens_list, dtype=torch.int32, device=device
    )

    # last_chunk_indices is empty when there are no sequences (len(starts) == 0).
    last_chunk_indices_t = async_tensor_h2d(
        last_chunk_indices, dtype=torch.int32, device=device
    )
    seq_idx_chunks_t = async_tensor_h2d(
        seq_idx_chunks, dtype=torch.int32, device=device
    )
    return cu_chunk_seqlens, last_chunk_indices_t, seq_idx_chunks_t


class Mamba2AttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "MAMBA2_ATTN"

    @staticmethod
    def get_builder_cls() -> type["Mamba2AttentionMetadataBuilder"]:
        return Mamba2AttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class Mamba2AttentionMetadata(BaseMambaAttentionMetadata):
    prep_initial_states: bool = False
    chunk_size: int = 0

    # Chunk-related metadata (only for prefill)
    seq_idx_p: torch.Tensor | None = None

    # FlashInfer packed-varlen metadata. These tensors are computed once by
    # the metadata builder and reused by every Mamba2 layer in the iteration.
    fi_seq_idx_p: torch.Tensor | None = None
    fi_chunk_indices_p: torch.Tensor | None = None
    fi_chunk_offsets_p: torch.Tensor | None = None
    fi_seq_chunk_cumsum_p: torch.Tensor | None = None
    fi_intermediate_state_indices_p: torch.Tensor | None = None
    fi_valid_seqlen: int = 0
    fi_padded_seqlen: int = 0
    fi_num_seqs: int = 0


class Mamba2AttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[Mamba2AttentionMetadata]
):
    metadata_cls = Mamba2AttentionMetadata

    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        chunk_size = vllm_config.model_config.get_mamba_chunk_size()
        assert chunk_size is not None, (
            "chunk_size needs to be set in the model config for Mamba2 models"
        )
        self.chunk_size: int = chunk_size
        self.use_flashinfer_prefill = (
            vllm_config.mamba_config.prefill_backend
            == MambaPrefillBackendEnum.FLASHINFER
        )
        self.mamba_block_size = kv_cache_spec.block_size

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs: Any,
    ) -> Mamba2AttentionMetadata:
        common = self._compute_common_metadata(
            common_attn_metadata,
            num_accepted_tokens=kwargs.get("num_accepted_tokens"),
            prev_last_scheduled_idx=kwargs.get("prev_last_scheduled_idx"),
        )

        seq_idx_p = None
        cu_chunk_seqlen_p = None
        last_chunk_indices_p = None
        prep_initial_states = False
        fi_seq_idx_p = None
        fi_chunk_indices_p = None
        fi_chunk_offsets_p = None
        fi_seq_chunk_cumsum_p = None
        fi_intermediate_state_indices_p = None
        fi_valid_seqlen = 0
        fi_padded_seqlen = 0
        fi_num_seqs = 0

        # Compute seq_idx for prefill only
        if common.num_prefills > 0:
            prep_initial_states = (
                torch.any(common.has_initial_states_p).item()
                if common.has_initial_states_p is not None
                else False
            )

            cu_chunk_seqlen_p, seq_idx_p, last_chunk_indices_p = (
                self._build_chunk_metadata_tensors(
                    self.chunk_size,
                    common,
                    common_attn_metadata,
                )
            )

            if self.use_flashinfer_prefill:
                seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
                assert seq_lens_cpu is not None
                query_start_loc_p_cpu = (
                    common_attn_metadata.query_start_loc_cpu[
                        -common.num_prefills - 1 :
                    ]
                    - common.num_decode_tokens
                )
                query_lens_cpu = (
                    query_start_loc_p_cpu[1:] - query_start_loc_p_cpu[:-1]
                )
                num_computed_tokens_p_cpu = (
                    seq_lens_cpu[
                        common.num_reqs - common.num_prefills : common.num_reqs
                    ]
                    - query_lens_cpu
                )
                fi_cpu = compute_flashinfer_ssd_metadata(
                    query_start_loc_p_cpu,
                    num_computed_tokens_p_cpu,
                    chunk_size=self.chunk_size,
                    mamba_block_size=self.mamba_block_size,
                )
                device = common_attn_metadata.query_start_loc.device
                fi_seq_idx_p = async_tensor_h2d(
                    fi_cpu.seq_idx, dtype=torch.int32, device=device
                ).unsqueeze(0)
                fi_chunk_indices_p = async_tensor_h2d(
                    fi_cpu.chunk_indices, dtype=torch.int32, device=device
                )
                fi_chunk_offsets_p = async_tensor_h2d(
                    fi_cpu.chunk_offsets, dtype=torch.int32, device=device
                )
                fi_seq_chunk_cumsum_p = async_tensor_h2d(
                    fi_cpu.seq_chunk_cumsum, dtype=torch.int32, device=device
                )

                # Convert per-segment block-table columns into physical cache
                # rows once. Every layer has its own state tensor but shares
                # this row namespace and therefore reuses the same mapping.
                segment_seq_ids = async_tensor_h2d(
                    fi_cpu.segment_seq_ids, dtype=torch.int64, device=device
                )
                segment_block_indices = async_tensor_h2d(
                    fi_cpu.segment_state_block_indices,
                    dtype=torch.int64,
                    device=device,
                )
                assert common.state_indices_tensor_p is not None
                selected = segment_block_indices >= 0
                safe_block_indices = segment_block_indices.clamp_min(0)
                gathered_state_indices = common.state_indices_tensor_p[
                    segment_seq_ids, safe_block_indices
                ]
                fi_intermediate_state_indices_p = torch.where(
                    selected,
                    gathered_state_indices,
                    torch.full_like(gathered_state_indices, -1),
                ).to(torch.int32)
                fi_valid_seqlen = fi_cpu.valid_seqlen
                fi_padded_seqlen = fi_cpu.padded_seqlen
                fi_num_seqs = common.num_prefills

        return replace(
            common,
            prep_initial_states=prep_initial_states,
            chunk_size=self.chunk_size,
            seq_idx_p=seq_idx_p,
            cu_chunk_seqlen_p=cu_chunk_seqlen_p,
            last_chunk_indices_p=last_chunk_indices_p,
            fi_seq_idx_p=fi_seq_idx_p,
            fi_chunk_indices_p=fi_chunk_indices_p,
            fi_chunk_offsets_p=fi_chunk_offsets_p,
            fi_seq_chunk_cumsum_p=fi_seq_chunk_cumsum_p,
            fi_intermediate_state_indices_p=fi_intermediate_state_indices_p,
            fi_valid_seqlen=fi_valid_seqlen,
            fi_padded_seqlen=fi_padded_seqlen,
            fi_num_seqs=fi_num_seqs,
        )
