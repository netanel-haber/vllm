# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backends.mamba2_attn import (
    compute_flashinfer_ssd_metadata,
)


def test_flashinfer_ssd_metadata_splits_all_boundary_types():
    # Sequence 0 crosses a physical 128-token boundary. Sequence 1 begins at
    # packed offset 200 with an existing 1100-token state, so its 1152-token
    # cache checkpoint lies at packed offset 252, before the next physical
    # boundary at 256.
    metadata = compute_flashinfer_ssd_metadata(
        torch.tensor([0, 200, 400], dtype=torch.int32),
        torch.tensor([0, 1100], dtype=torch.int32),
        chunk_size=128,
        mamba_block_size=1152,
    )

    assert metadata.valid_seqlen == 400
    assert metadata.padded_seqlen == 512
    assert metadata.chunk_indices == [0, 1, 1, 1, 2, 3]
    assert metadata.chunk_offsets == [0, 0, 72, 124, 0, 0]
    assert metadata.seq_chunk_cumsum == [0, 2, 6]
    assert metadata.segment_seq_ids == [0, 0, 1, 1, 1, 1]
    assert metadata.segment_state_block_indices == [-1, 0, 0, -1, -1, 1]
    assert metadata.seq_idx[:200] == [0] * 200
    assert metadata.seq_idx[200:400] == [1] * 200
    # Padded tokens carry a valid id but are masked by valid_seqlen.
    assert metadata.seq_idx[400:] == [1] * 112


@pytest.mark.parametrize("total", [32343, 32344, 32346, 32768])
def test_flashinfer_ssd_metadata_supports_target_packed_lengths(total: int):
    metadata = compute_flashinfer_ssd_metadata(
        torch.tensor([0, total], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        chunk_size=128,
        mamba_block_size=1152,
    )

    expected_padded = ((total + 127) // 128) * 128
    assert metadata.valid_seqlen == total
    assert metadata.padded_seqlen == expected_padded
    assert len(metadata.seq_idx) == expected_padded
    assert metadata.seq_chunk_cumsum == [0, len(metadata.chunk_indices)]
    assert all(
        0 <= offset < 128 for offset in metadata.chunk_offsets
    )
    assert all(
        later > earlier
        for earlier, later in zip(
            metadata.chunk_indices, metadata.chunk_indices[1:]
        )
    )

    # Every completed 1152-token cache block and the final partial/full block
    # receives exactly one post-segment state destination.
    selected = [
        block for block in metadata.segment_state_block_indices if block >= 0
    ]
    assert selected == list(range((total - 1) // 1152 + 1))


def test_flashinfer_ssd_metadata_mixed_prior_and_partial_finals():
    query_lens = [1, 127, 128, 129, 1151, 1152, 1153]
    query_start_loc = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=torch.int32
    )
    contexts = torch.tensor([0, 1, 127, 128, 1151, 1152, 1153])

    metadata = compute_flashinfer_ssd_metadata(
        query_start_loc,
        contexts,
        chunk_size=128,
        mamba_block_size=1152,
    )

    assert metadata.seq_chunk_cumsum[-1] == len(metadata.chunk_indices)
    for seq_id in range(len(query_lens)):
        begin = metadata.seq_chunk_cumsum[seq_id]
        end = metadata.seq_chunk_cumsum[seq_id + 1]
        assert end > begin
        # The last logical segment of every sequence writes its partial or
        # full final state to the block containing the final position.
        expected_final_block = (
            int(contexts[seq_id]) + query_lens[seq_id] - 1
        ) // 1152
        assert metadata.segment_state_block_indices[end - 1] == expected_final_block


@pytest.mark.parametrize(
    "query_start_loc,contexts,error",
    [
        ([1, 2], [0], "start at zero"),
        ([0, 2, 1], [0, 0], "nonnegative"),
        ([0, 1], [0, 0], "one more entry"),
    ],
)
def test_flashinfer_ssd_metadata_rejects_invalid_inputs(
    query_start_loc, contexts, error
):
    with pytest.raises(ValueError, match=error):
        compute_flashinfer_ssd_metadata(
            torch.tensor(query_start_loc),
            torch.tensor(contexts),
            chunk_size=128,
            mamba_block_size=1152,
        )
