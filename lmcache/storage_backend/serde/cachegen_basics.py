# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import List

# Third Party
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors
from transformers import AutoConfig
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate

logger = init_logger(__name__)

CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK = 256
_CACHEGEN_FORMAT_VERSION = 1


def _metadata_tensor(value: int) -> torch.Tensor:
    """Create a scalar tensor for an integer format field.

    Args:
        value: Integer metadata value to encode.

    Returns:
        A scalar ``int64`` CPU tensor.
    """
    return torch.tensor(value, dtype=torch.int64)


def _read_metadata_tensor(tensors: dict[str, torch.Tensor], name: str) -> int:
    """Read and validate an integer format field.

    Args:
        tensors: Decoded tensor mapping.
        name: Required scalar field name.

    Returns:
        The decoded integer value.

    Raises:
        ValueError: If the field is missing or is not a scalar ``int64`` tensor.
    """
    tensor = tensors.get(name)
    if tensor is None or tensor.dtype != torch.int64 or tensor.shape != torch.Size([]):
        raise ValueError(f"Invalid CacheGen metadata field: {name}")
    return int(tensor.item())


def _load_tensor_mapping(data: bytes) -> dict[str, torch.Tensor]:
    """Decode a CacheGen tensor mapping and validate its format version.

    Args:
        data: Bytes produced by ``safetensors.torch.save``.

    Returns:
        The decoded CPU tensor mapping.

    Raises:
        ValueError: If the format version is unsupported.
        safetensors.SafetensorError: If the payload is malformed.
    """
    tensors = load_tensors(data)
    version = _read_metadata_tensor(tensors, "format_version")
    if version != _CACHEGEN_FORMAT_VERSION:
        raise ValueError(f"Unsupported CacheGen format version: {version}")
    return tensors


@dataclass
class QuantizationSpec:
    start_layer: int
    end_layer: int
    bins: int

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)


@dataclass
class CacheGenConfig:
    # TODO: move this class to another file like "cachegen_basics.py"
    nlayers: int
    kspecs: List[QuantizationSpec]
    vspecs: List[QuantizationSpec]

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)

    @staticmethod
    def from_model_name(model_name: str) -> "CacheGenConfig":
        family_7b = [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "lmsys/longchat-7b-16k",
            "Qwen/Qwen-7B",
        ]
        family_8b = ["meta-llama/Llama-3.1-8B-Instruct"]
        family_9b = ["THUDM/glm-4-9b-chat"]
        if model_name in family_7b:
            return CacheGenConfig(
                nlayers=32,
                kspecs=[
                    QuantizationSpec(start_layer=0, end_layer=10, bins=32),
                    QuantizationSpec(start_layer=10, end_layer=32, bins=16),
                ],
                vspecs=[
                    QuantizationSpec(start_layer=0, end_layer=2, bins=32),
                    QuantizationSpec(start_layer=2, end_layer=32, bins=16),
                ],
            )
        elif model_name in family_8b:
            return CacheGenConfig(
                nlayers=32,
                kspecs=[
                    QuantizationSpec(start_layer=0, end_layer=10, bins=32),
                    QuantizationSpec(start_layer=10, end_layer=32, bins=16),
                ],
                vspecs=[
                    QuantizationSpec(start_layer=0, end_layer=2, bins=32),
                    QuantizationSpec(start_layer=2, end_layer=32, bins=16),
                ],
            )
        # TODO(Jiayi): needs tuning for better quality
        elif model_name in family_9b:
            return CacheGenConfig(
                nlayers=40,
                kspecs=[
                    QuantizationSpec(start_layer=0, end_layer=10, bins=32),
                    QuantizationSpec(start_layer=10, end_layer=40, bins=16),
                ],
                vspecs=[
                    QuantizationSpec(start_layer=0, end_layer=2, bins=32),
                    QuantizationSpec(start_layer=2, end_layer=40, bins=16),
                ],
            )
        else:
            try:
                config = AutoConfig.from_pretrained(model_name)
                # Default name caught by num_hidden_layers
                if config.num_hidden_layers is None:
                    raise ValueError(
                        f"num_hidden_layers is None for model {model_name}"
                    )
                if config.num_hidden_layers < 10:
                    return CacheGenConfig(
                        nlayers=config.num_hidden_layers,
                        kspecs=[
                            QuantizationSpec(
                                start_layer=0,
                                end_layer=config.num_hidden_layers,
                                bins=32,
                            ),
                        ],
                        vspecs=[
                            QuantizationSpec(
                                start_layer=0,
                                end_layer=config.num_hidden_layers,
                                bins=32,
                            ),
                        ],
                    )
                else:
                    return CacheGenConfig(
                        nlayers=config.num_hidden_layers,
                        kspecs=[
                            QuantizationSpec(start_layer=0, end_layer=10, bins=32),
                            QuantizationSpec(
                                start_layer=10,
                                end_layer=config.num_hidden_layers,
                                bins=16,
                            ),
                        ],
                        vspecs=[
                            QuantizationSpec(start_layer=0, end_layer=2, bins=32),
                            QuantizationSpec(
                                start_layer=2,
                                end_layer=config.num_hidden_layers,
                                bins=16,
                            ),
                        ],
                    )
            except Exception as e:
                raise ValueError(
                    f"Model {model_name} not supported by CacheGenConfig"
                ) from e


@dataclass
class CacheGenEncoderOutput:
    # TODO: maybe use numpy array so that we can directly tobytes() and
    # frombuffer() to have a better performance
    bytestream: bytes
    start_indices: torch.Tensor
    cdf: torch.Tensor
    max_tensors_key: torch.Tensor
    max_tensors_value: torch.Tensor
    num_heads: int
    head_size: int

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)

    def to_bytes(self) -> bytes:
        """Encode this output as a data-only tensor container.

        Returns:
            A versioned byte payload containing tensors and scalar metadata.
        """
        return save_tensors(
            {
                "format_version": _metadata_tensor(_CACHEGEN_FORMAT_VERSION),
                "bytestream": torch.frombuffer(
                    bytearray(self.bytestream), dtype=torch.uint8
                ),
                "start_indices": self.start_indices.contiguous(),
                "cdf": self.cdf.contiguous(),
                "max_tensors_key": self.max_tensors_key.contiguous(),
                "max_tensors_value": self.max_tensors_value.contiguous(),
                "num_heads": _metadata_tensor(self.num_heads),
                "head_size": _metadata_tensor(self.head_size),
            }
        )

    @staticmethod
    def from_bytes(bs: bytes) -> "CacheGenEncoderOutput":
        """Decode a data-only tensor container.

        Args:
            bs: Bytes produced by :meth:`to_bytes`.

        Returns:
            A CacheGen encoder output whose tensors reside on CPU.

        Raises:
            ValueError: If the payload fields do not match the expected schema.
            safetensors.SafetensorError: If the payload is malformed.
        """
        tensors = _load_tensor_mapping(bs)
        expected_fields = {
            "format_version",
            "bytestream",
            "start_indices",
            "cdf",
            "max_tensors_key",
            "max_tensors_value",
            "num_heads",
            "head_size",
        }
        if tensors.keys() != expected_fields:
            raise ValueError("Invalid CacheGen encoder output fields")
        bytestream = tensors["bytestream"]
        if bytestream.dtype != torch.uint8 or bytestream.ndim != 1:
            raise ValueError("Invalid CacheGen byte stream tensor")
        return CacheGenEncoderOutput(
            bytestream=bytestream.numpy().tobytes(),
            start_indices=tensors["start_indices"],
            cdf=tensors["cdf"],
            max_tensors_key=tensors["max_tensors_key"],
            max_tensors_value=tensors["max_tensors_value"],
            num_heads=_read_metadata_tensor(tensors, "num_heads"),
            head_size=_read_metadata_tensor(tensors, "head_size"),
        )


@dataclass
class CacheGenGPUBytestream:
    bytestream: torch.Tensor
    bytestream_lengths: torch.Tensor  # [nlayers, nchannels, bytestream_length]
    ntokens: int

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)


@dataclass
class CacheGenGPUEncoderOutput:
    data_chunks: List[CacheGenGPUBytestream]
    cdf: torch.Tensor
    max_tensors_key: torch.Tensor
    max_tensors_value: torch.Tensor
    num_heads: int
    head_size: int

    def __getitem__(self, key: str) -> int:
        return getattr(self, key)

    @_lmcache_nvtx_annotate
    def to_bytes(self) -> bytes:
        """Encode this output as a data-only tensor container.

        Returns:
            A versioned byte payload containing tensors and scalar metadata.
        """
        tensors = {
            "format_version": _metadata_tensor(_CACHEGEN_FORMAT_VERSION),
            "chunk_count": _metadata_tensor(len(self.data_chunks)),
            "cdf": self.cdf.contiguous(),
            "max_tensors_key": self.max_tensors_key.contiguous(),
            "max_tensors_value": self.max_tensors_value.contiguous(),
            "num_heads": _metadata_tensor(self.num_heads),
            "head_size": _metadata_tensor(self.head_size),
        }
        for index, chunk in enumerate(self.data_chunks):
            tensors[f"chunk_{index}_bytestream"] = chunk.bytestream.contiguous()
            tensors[f"chunk_{index}_lengths"] = chunk.bytestream_lengths.contiguous()
            tensors[f"chunk_{index}_ntokens"] = _metadata_tensor(chunk.ntokens)
        return save_tensors(tensors)

    @staticmethod
    @_lmcache_nvtx_annotate
    def from_bytes(
        bs: bytes, device: torch.device | str = "cpu"
    ) -> "CacheGenGPUEncoderOutput":
        """Decode a data-only tensor container on the requested device.

        Args:
            bs: Bytes produced by :meth:`to_bytes`.
            device: Device that will consume the decoded CacheGen tensors.

        Returns:
            A validated CacheGen encoder output on ``device``.

        Raises:
            ValueError: If the payload fields or tensor relationships are invalid.
            safetensors.SafetensorError: If the payload is malformed.
        """
        tensors = _load_tensor_mapping(bs)
        chunk_count = _read_metadata_tensor(tensors, "chunk_count")
        if chunk_count <= 0:
            raise ValueError("CacheGen output must contain at least one data chunk")

        expected_fields = {
            "format_version",
            "chunk_count",
            "cdf",
            "max_tensors_key",
            "max_tensors_value",
            "num_heads",
            "head_size",
        }
        for index in range(chunk_count):
            expected_fields.update(
                {
                    f"chunk_{index}_bytestream",
                    f"chunk_{index}_lengths",
                    f"chunk_{index}_ntokens",
                }
            )
        if tensors.keys() != expected_fields:
            raise ValueError("Invalid CacheGen GPU encoder output fields")

        num_heads = _read_metadata_tensor(tensors, "num_heads")
        head_size = _read_metadata_tensor(tensors, "head_size")
        if num_heads <= 0 or head_size <= 0:
            raise ValueError("CacheGen head dimensions must be positive")

        cdf = tensors["cdf"]
        max_tensors_key = tensors["max_tensors_key"]
        max_tensors_value = tensors["max_tensors_value"]
        if cdf.ndim != 3 or cdf.dtype != torch.int16:
            raise ValueError("Invalid CacheGen CDF tensor")
        if max_tensors_key.ndim != 3 or max_tensors_value.ndim != 3:
            raise ValueError("Invalid CacheGen maximum-value tensors")
        if max_tensors_key.shape[1:] != max_tensors_value.shape[1:]:
            raise ValueError("CacheGen maximum-value tensor shapes do not match")
        if cdf.shape[0] != max_tensors_key.shape[0] + max_tensors_value.shape[0]:
            raise ValueError("CacheGen layer metadata does not match the CDF")
        if cdf.shape[1] != num_heads * head_size:
            raise ValueError("CacheGen head metadata does not match the CDF")

        data_chunks = []
        total_tokens = 0
        for index in range(chunk_count):
            bytestream = tensors[f"chunk_{index}_bytestream"]
            lengths = tensors[f"chunk_{index}_lengths"]
            ntokens = _read_metadata_tensor(tensors, f"chunk_{index}_ntokens")
            if bytestream.dtype != torch.uint8 or bytestream.ndim != 1:
                raise ValueError("Invalid CacheGen chunk byte stream")
            if lengths.dtype != torch.int32 or lengths.shape != cdf.shape[:2]:
                raise ValueError("Invalid CacheGen chunk length tensor")
            if ntokens <= 0 or ntokens > CACHEGEN_GPU_MAX_TOKENS_PER_CHUNK:
                raise ValueError("Invalid CacheGen chunk token count")
            if bool(torch.any(lengths < 0)) or int(lengths.sum()) != bytestream.numel():
                raise ValueError("CacheGen chunk lengths do not match its byte stream")
            total_tokens += ntokens
            data_chunks.append(
                CacheGenGPUBytestream(
                    bytestream=bytestream.to(device),
                    bytestream_lengths=lengths.to(device),
                    ntokens=ntokens,
                )
            )
        if total_tokens != max_tensors_key.shape[1]:
            raise ValueError("CacheGen chunk tokens do not match tensor metadata")

        return CacheGenGPUEncoderOutput(
            data_chunks=data_chunks,
            cdf=cdf.to(device),
            max_tensors_key=max_tensors_key.to(device),
            max_tensors_value=max_tensors_value.to(device),
            num_heads=num_heads,
            head_size=head_size,
        )

    def debug_print_device(self):
        logger.debug("bytestream device: %s", self.data_chunks[0].bytestream.device)
        logger.debug(
            "bytestream_lengths device: %s",
            self.data_chunks[0].bytestream_lengths.device,
        )
        logger.debug("cdf device: %s", self.cdf.device)
        logger.debug("max_tensors_key device: %s", self.max_tensors_key.device)
        logger.debug("max_tensors_value device: %s", self.max_tensors_value.device)
