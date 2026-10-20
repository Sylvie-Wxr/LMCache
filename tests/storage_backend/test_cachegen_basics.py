# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the CacheGen tensor container."""

# Third Party
from safetensors import SafetensorError
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors
import pytest
import torch

# First Party
from lmcache.storage_backend.serde.cachegen_basics import (
    CacheGenEncoderOutput,
    CacheGenGPUBytestream,
    CacheGenGPUEncoderOutput,
)


def _make_gpu_output() -> CacheGenGPUEncoderOutput:
    """Create a small, internally consistent CacheGen output on CPU."""
    lengths = torch.ones((4, 4), dtype=torch.int32)
    return CacheGenGPUEncoderOutput(
        data_chunks=[
            CacheGenGPUBytestream(
                bytestream=torch.arange(16, dtype=torch.uint8),
                bytestream_lengths=lengths,
                ntokens=3,
            )
        ],
        cdf=torch.arange(4 * 4 * 33, dtype=torch.int16).reshape(4, 4, 33),
        max_tensors_key=torch.ones((2, 3, 1)),
        max_tensors_value=torch.full((2, 3, 1), 2.0),
        num_heads=2,
        head_size=2,
    )


def test_encoder_output_round_trip() -> None:
    """Preserve the CPU encoder fields through the data-only container."""
    output = CacheGenEncoderOutput(
        bytestream=b"cachegen",
        start_indices=torch.tensor([0, 4], dtype=torch.int32),
        cdf=torch.arange(8, dtype=torch.int16),
        max_tensors_key=torch.ones((1, 2, 1)),
        max_tensors_value=torch.full((1, 2, 1), 2.0),
        num_heads=2,
        head_size=4,
    )

    restored = CacheGenEncoderOutput.from_bytes(output.to_bytes())

    assert restored.bytestream == output.bytestream
    assert restored.num_heads == output.num_heads
    assert restored.head_size == output.head_size
    torch.testing.assert_close(restored.start_indices, output.start_indices)
    torch.testing.assert_close(restored.cdf, output.cdf)
    torch.testing.assert_close(restored.max_tensors_key, output.max_tensors_key)
    torch.testing.assert_close(restored.max_tensors_value, output.max_tensors_value)


def test_gpu_encoder_output_round_trip() -> None:
    """Preserve every tensor and scalar field through the container."""
    output = _make_gpu_output()

    restored = CacheGenGPUEncoderOutput.from_bytes(output.to_bytes())

    assert restored.num_heads == output.num_heads
    assert restored.head_size == output.head_size
    assert len(restored.data_chunks) == 1
    assert restored.data_chunks[0].ntokens == 3
    torch.testing.assert_close(restored.cdf, output.cdf)
    torch.testing.assert_close(restored.max_tensors_key, output.max_tensors_key)
    torch.testing.assert_close(restored.max_tensors_value, output.max_tensors_value)
    torch.testing.assert_close(
        restored.data_chunks[0].bytestream, output.data_chunks[0].bytestream
    )
    torch.testing.assert_close(
        restored.data_chunks[0].bytestream_lengths,
        output.data_chunks[0].bytestream_lengths,
    )


def test_gpu_encoder_output_rejects_untyped_bytes() -> None:
    """Reject bytes that are not a valid tensor container."""
    with pytest.raises(SafetensorError):
        CacheGenGPUEncoderOutput.from_bytes(b"not-a-tensor-container")


def test_gpu_encoder_output_rejects_unknown_fields() -> None:
    """Reject fields outside the versioned CacheGen schema."""
    tensors = load_tensors(_make_gpu_output().to_bytes())
    tensors["unexpected"] = torch.tensor(1)

    with pytest.raises(ValueError, match="fields"):
        CacheGenGPUEncoderOutput.from_bytes(save_tensors(tensors))


def test_gpu_encoder_output_rejects_inconsistent_token_count() -> None:
    """Reject chunk metadata that disagrees with the encoded tensor shapes."""
    tensors = load_tensors(_make_gpu_output().to_bytes())
    tensors["chunk_0_ntokens"] = torch.tensor(2, dtype=torch.int64)

    with pytest.raises(ValueError, match="tokens"):
        CacheGenGPUEncoderOutput.from_bytes(save_tensors(tensors))
