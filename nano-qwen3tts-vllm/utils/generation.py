"""Prepare talker inputs for Qwen3-TTS generate.

Extracts the input preparation logic from Qwen3TTSForConditionalGeneration.generate()
so that talker_input_embeds, trailing_text_hiddens, tts_pad_embed, and attention_mask
can be computed once and then fed into the talker's generate method (e.g. for
vLLM-style batching or separate preprocessing).

Use prepare_inputs_from_components() when you have talker + config.
Use prepare_inputs_from_config() when you have only config + embedding/projection
callables and device (no talker).
"""

from __future__ import annotations

import torch
from typing import Callable, Optional, Union, List

def prepare_inputs(
    config,
    input_ids: list[torch.Tensor],
    instruct_ids: Optional[list[torch.Tensor]] = None,
    ref_ids: Optional[list[torch.Tensor]] = None,
    voice_clone_prompt: Optional[dict] = None,
    languages: Optional[list[str]] = None,
    speakers: Optional[list[str]] = None,
    non_streaming_mode: bool = False,
    *,
    text_embedding: Callable,
    input_embedding: Callable,
    text_projection: Callable,
    device: Union[torch.device, str],
    voice_clone_spk_embeds: Optional[list[torch.Tensor]] = None,
    generate_speaker_prompt_fn: Optional[Callable] = None,
    generate_icl_prompt_fn: Optional[Callable] = None,
):
    """Prepare talker inputs from config + embedding/projection callables (no talker required).

    Use this when you only have config and the embedding/projection layers (or
    callables). No full model or talker needed.

    Args:
        config: Config with .talker_config and .tts_bos_token_id, .tts_eos_token_id, .tts_pad_token_id.
        input_ids: List of token tensors, one per batch item.
        instruct_ids: Optional list of instruct token tensors (same length as input_ids).
        ref_ids: Optional list of reference token tensors for ICL (same length as input_ids).
        voice_clone_prompt: Optional dict (ref_spk_embedding, ref_code, x_vector_only_mode, icl_mode).
        languages: List of language strings (e.g. "chinese", "english", "auto").
        speakers: List of speaker names or None; length must match input_ids.
        non_streaming_mode: If True, use non-streaming input layout.
        get_text_embeddings: Callable() -> embedding layer; layer(token_ids) -> text embeddings (e.g. talker.get_text_embeddings).
        get_input_embeddings: Callable() -> embedding layer; layer(token_ids) -> codec/speaker embeddings (e.g. talker.get_input_embeddings).
        text_projection: Callable(embeddings) -> projected embeddings (e.g. talker.text_projection).
        device: torch.device or device string for creating tensors.
        voice_clone_spk_embeds: Precomputed speaker embeddings when using voice_clone_prompt.
        generate_speaker_prompt_fn: Callable(voice_clone_prompt) -> list of speaker embeds (if no precomputed).
        generate_icl_prompt_fn: Callable for ICL; (text_id, ref_id, ref_code, tts_pad_embed, tts_eos_embed, non_streaming_mode) -> (icl_embed, trailing_hidden).

    Returns:
        Tuple of (talker_input_embeds, trailing_text_hiddens, tts_pad_embed, talker_attention_mask).
    """
    if isinstance(device, str):
        device = torch.device(device)

    talker_input_embeds = [[] for _ in range(len(input_ids))]

    if voice_clone_prompt is not None and voice_clone_spk_embeds is None and generate_speaker_prompt_fn is not None:
        voice_clone_spk_embeds = generate_speaker_prompt_fn(voice_clone_prompt)
    elif voice_clone_prompt is not None and voice_clone_spk_embeds is None:
        raise ValueError(
            "voice_clone_prompt is set but neither voice_clone_spk_embeds nor generate_speaker_prompt_fn was provided"
        )

    if instruct_ids is not None:
        for index, instruct_id in enumerate(instruct_ids):
            if instruct_id is not None:
                talker_input_embeds[index].append(
                    text_projection(text_embedding(instruct_id))
                )

    trailing_text_hiddens = []
    if speakers is None:
        speakers = [None] * len(input_ids)
    if languages is None:
        languages = ["auto"] * len(input_ids)

    tts_pad_embed = None
    for index, (input_id, language, speaker) in enumerate(
        zip(input_ids, languages, speakers)
    ):
        # [DEBUG vLLM] tokenization / input_id
        if voice_clone_spk_embeds is None:
            if speaker == "" or speaker is None:
                speaker_embed = None
            else:
                if speaker.lower() not in config.talker_config.spk_id:
                    raise NotImplementedError(
                        f"Speaker {speaker} not implemented"
                    )
                spk_id = config.talker_config.spk_id[speaker.lower()]
                speaker_embed = input_embedding(
                    torch.tensor(
                        spk_id,
                        device=device,
                        dtype=input_id.dtype,
                    )
                )
        else:
            if voice_clone_prompt["x_vector_only_mode"][index] or voice_clone_prompt[
                "icl_mode"
            ][index]:
                speaker_embed = voice_clone_spk_embeds[index]
            else:
                speaker_embed = None

        assert language is not None
        if language.lower() == "auto":
            language_id = None
        else:
            if language.lower() not in config.talker_config.codec_language_id:
                raise NotImplementedError(
                    f"Language {language} not implemented"
                )
            language_id = config.talker_config.codec_language_id[
                language.lower()
            ]

        if (
            language.lower() in ["chinese", "auto"]
            and speaker != ""
            and speaker is not None
            and config.talker_config.spk_is_dialect.get(
                speaker.lower(), False
            ) is not False
        ):
            dialect = config.talker_config.spk_is_dialect[speaker.lower()]
            language_id = config.talker_config.codec_language_id[dialect]

        tts_bos_embed, tts_eos_embed, tts_pad_embed_i = text_projection(
            text_embedding(
                torch.tensor(
                    [
                        [
                            config.tts_bos_token_id,
                            config.tts_eos_token_id,
                            config.tts_pad_token_id,
                        ]
                    ],
                    device=device,
                    dtype=input_id.dtype,
                )
            )
        ).chunk(3, dim=1)
        tts_pad_embed = tts_pad_embed_i

        if language_id is None:
            codec_prefill_list = [
                [
                    config.talker_config.codec_nothink_id,
                    config.talker_config.codec_think_bos_id,
                    config.talker_config.codec_think_eos_id,
                ]
            ]
        else:
            codec_prefill_list = [
                [
                    config.talker_config.codec_think_id,
                    config.talker_config.codec_think_bos_id,
                    language_id,
                    config.talker_config.codec_think_eos_id,
                ]
            ]

        codec_input_embedding_0 = input_embedding(
            torch.tensor(
                codec_prefill_list,
                device=device,
                dtype=input_id.dtype,
            )
        )
        codec_input_embedding_1 = input_embedding(
            torch.tensor(
                [
                    [
                        config.talker_config.codec_pad_id,
                        config.talker_config.codec_bos_id,
                    ]
                ],
                device=device,
                dtype=input_id.dtype,
            )
        )

        if speaker_embed is None:
            codec_input_embedding = torch.cat(
                [codec_input_embedding_0, codec_input_embedding_1], dim=1
            )
        else:
            codec_input_embedding = torch.cat(
                [
                    codec_input_embedding_0,
                    speaker_embed.view(1, 1, -1),
                    codec_input_embedding_1,
                ],
                dim=1,
            )

        _talker_input_embed_role = text_projection(
            text_embedding(input_id[:, :3])
        )

        _talker_input_embed = torch.cat(
            (
                tts_pad_embed_i.expand(
                    -1, codec_input_embedding.shape[1] - 2, -1
                ),
                tts_bos_embed,
            ),
            dim=1,
        ) + codec_input_embedding[:, :-1]

        talker_input_embed = torch.cat(
            (_talker_input_embed_role, _talker_input_embed), dim=1
        )

        if (
            voice_clone_prompt is not None
            and voice_clone_prompt.get("ref_code") is not None
            and voice_clone_prompt["icl_mode"][index]
        ):
            if generate_icl_prompt_fn is None:
                raise ValueError(
                    "ICL mode requested but generate_icl_prompt_fn was not provided"
                )
            if ref_ids is None or ref_ids[index] is None:
                raise ValueError(
                    f"ICL mode requires ref_ids, but ref_ids[{index}] is None. "
                    "Please provide ref_text when creating voice_clone_prompt or when calling generate_voice_clone."
                )
            icl_input_embed, trailing_text_hidden = generate_icl_prompt_fn(
                text_id=input_id[:, 3:-5],
                ref_id=ref_ids[index][:, 3:-2],
                ref_code=voice_clone_prompt["ref_code"][index].to(device),
                tts_pad_embed=tts_pad_embed_i,
                tts_eos_embed=tts_eos_embed,
                non_streaming_mode=non_streaming_mode,
            )
            talker_input_embed = torch.cat(
                [talker_input_embed, icl_input_embed], dim=1
            )
        else:
            talker_input_embed = torch.cat(
                [
                    talker_input_embed,
                    text_projection(text_embedding(input_id[:, 3:4]))
                    + codec_input_embedding[:, -1:],
                ],
                dim=1,
            )
            if non_streaming_mode:
                talker_input_embed = talker_input_embed[:, :-1]
                talker_input_embed = torch.cat(
                    [
                        talker_input_embed,
                        torch.cat(
                            (
                                text_projection(
                                    text_embedding(input_id[:, 3:-5])
                                ),
                                tts_eos_embed,
                            ),
                            dim=1,
                        )
                        + input_embedding(
                            torch.tensor(
                                [
                                    [
                                        config.talker_config.codec_pad_id,
                                    ]
                                    * (input_id[:, 3:-5].shape[1] + 1)
                                ],
                                device=device,
                                dtype=input_id.dtype,
                            )
                        ),
                        tts_pad_embed_i
                        + input_embedding(
                            torch.tensor(
                                [[config.talker_config.codec_bos_id]],
                                device=device,
                                dtype=input_id.dtype,
                            )
                        ),
                    ],
                    dim=1,
                )
                trailing_text_hidden = tts_pad_embed_i
            else:
                trailing_text_hidden = torch.cat(
                    (
                        text_projection(
                            text_embedding(input_id[:, 4:-5])
                        ),
                        tts_eos_embed,
                    ),
                    dim=1,
                )

        talker_input_embeds[index].append(talker_input_embed)
        trailing_text_hiddens.append(trailing_text_hidden)

    for index in range(len(talker_input_embeds)):
        talker_input_embeds[index] = torch.cat(
            [item for item in talker_input_embeds[index] if item is not None],
            dim=1,
        )

    original_lengths = torch.tensor(
        [t.shape[1] for t in talker_input_embeds],
        device=talker_input_embeds[0].device,
    )
    sequences = [t.squeeze(0) for t in talker_input_embeds]
    sequences_reversed = [t.flip(dims=[0]) for t in sequences]
    padded_reversed = torch.nn.utils.rnn.pad_sequence(
        sequences_reversed,
        batch_first=True,
        padding_value=0.0,
    )
    talker_input_embeds_batched = padded_reversed.flip(dims=[1])

    batch_size, max_len = (
        talker_input_embeds_batched.shape[0],
        talker_input_embeds_batched.shape[1],
    )
    indices = torch.arange(
        max_len,
        device=talker_input_embeds_batched.device,
    ).expand(batch_size, -1)
    num_pads = max_len - original_lengths.to(talker_input_embeds_batched.device)
    talker_attention_mask = (
        (indices >= num_pads.unsqueeze(1))
        .long()
        .to(talker_input_embeds_batched.device)
    )

    pad_embedding_vector = tts_pad_embed.squeeze()
    sequences_to_pad = [t.squeeze(0) for t in trailing_text_hiddens]
    trailing_text_original_lengths = [s.shape[0] for s in sequences_to_pad]
    padded_hiddens = torch.nn.utils.rnn.pad_sequence(
        sequences_to_pad,
        batch_first=True,
        padding_value=0.0,
    )
    arange_tensor = torch.arange(
        max(trailing_text_original_lengths),
        device=padded_hiddens.device,
    ).expand(len(trailing_text_original_lengths), -1)
    lengths_tensor = torch.tensor(
        trailing_text_original_lengths,
        device=padded_hiddens.device,
    ).unsqueeze(1)
    padding_mask = arange_tensor >= lengths_tensor
    padded_hiddens[padding_mask] = pad_embedding_vector
    trailing_text_hiddens_batched = padded_hiddens

    return (
        talker_input_embeds_batched,
        trailing_text_hiddens_batched,
        tts_pad_embed,
        talker_attention_mask,
    )


def generate_speaker_prompt(
    voice_clone_prompt: dict,
    device: Union[torch.device, str],
    dtype: Optional[torch.dtype] = None,
) -> List[torch.Tensor]:
    """Generate speaker embeddings from voice_clone_prompt.
    
    Extracts ref_spk_embedding from voice_clone_prompt and moves to device/dtype.
    
    Args:
        voice_clone_prompt: Dict with 'ref_spk_embedding' key containing list of tensors.
        device: Target device for embeddings.
        dtype: Target dtype for embeddings (optional, uses original dtype if None).
    
    Returns:
        List of speaker embedding tensors, one per batch item.
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    voice_clone_spk_embeds = []
    for index in range(len(voice_clone_prompt['ref_spk_embedding'])):
        ref_spk_embedding = voice_clone_prompt["ref_spk_embedding"][index].to(device)
        if dtype is not None:
            ref_spk_embedding = ref_spk_embedding.to(dtype)
        voice_clone_spk_embeds.append(ref_spk_embedding)
    
    return voice_clone_spk_embeds


def generate_icl_prompt(
    text_id: torch.Tensor,
    ref_id: torch.Tensor,
    ref_code: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    tts_eos_embed: torch.Tensor,
    non_streaming_mode: bool,
    config,
    text_embedding: Callable,
    input_embedding: Callable,
    text_projection: Callable,
    code_predictor_embeddings: List[Callable],
    device: Union[torch.device, str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ICL (in-context learning) prompt embeddings.
    
    Creates ICL prompt by combining reference text + target text embeddings with
    reference codec embeddings.
    
    Args:
        text_id: Target text token IDs [1, T_text].
        ref_id: Reference text token IDs [1, T_ref].
        ref_code: Reference codec tokens [1, T_code, num_code_groups] or [1, T_code].
        tts_pad_embed: TTS pad embedding [1, 1, D].
        tts_eos_embed: TTS EOS embedding [1, 1, D].
        non_streaming_mode: Whether to use non-streaming mode.
        config: Config with talker_config containing codec token IDs.
        text_embedding: Callable that takes token IDs and returns text embeddings.
        input_embedding: Callable that takes token IDs and returns codec embeddings (for codebook 0).
        text_projection: Callable that projects text embeddings.
        code_predictor_embeddings: List of callables for codebooks 1-15.
        device: Device for creating tensors.
    
    Returns:
        Tuple of (icl_input_embed, trailing_text_hidden).
        - icl_input_embed: [1, T_icl, D] ICL input embeddings.
        - trailing_text_hidden: [1, T_trailing, D] or [1, 1, D] trailing hidden state.
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    num_code_groups = config.talker_config.num_code_groups
    
    # Text embed (ref id + text id + eos) [1, T1, D]
    text_embed = text_projection(
        text_embedding(torch.cat([ref_id, text_id], dim=-1))
    )
    text_embed = torch.cat([text_embed, tts_eos_embed], dim=1)
    
    # Codec embed (codec bos + codec) [1, T2, D]
    # ref_code is [1, num_code_groups] format (first timestep only for ICL prompt)
    # Match original implementation: ref_code[:, :1] for codebook 0, ref_code[:, i:i+1] for codebook i
    codec_embed = []
    for i in range(num_code_groups):
        if i == 0:
            # Use input_embedding for codebook 0
            # ref_code[:, :1] takes codebook 0 -> [1, 1]
            codec_embed.append(input_embedding(ref_code[:, :1]))
        else:
            # Use code_predictor_embeddings for codebooks 1-15
            # ref_code[:, i:i+1] takes codebook i -> [1, 1]
            if i - 1 < len(code_predictor_embeddings):
                codec_embed.append(code_predictor_embeddings[i - 1](ref_code[:, i:i+1]))
            else:
                # Fallback: use input_embedding if not enough code_predictor_embeddings
                codec_embed.append(input_embedding(ref_code[:, i:i+1]))
    
    # Each codec_embed[i] is [1, 1, D] (one codebook embedding)
    # Cat along dim=1 gives [1, num_code_groups, D], sum(1) gives [1, D], unsqueeze(0) gives [1, 1, D]
    codec_embed = torch.cat(codec_embed, dim=1).sum(1).unsqueeze(0)  # [1, 1, D]
    
    # Prepend codec_bos_id
    codec_bos_embed = input_embedding(
        torch.tensor(
            [[config.talker_config.codec_bos_id]],
            device=device,
            dtype=text_id.dtype,
        )
    )
    codec_embed = torch.cat([codec_bos_embed, codec_embed], dim=1)
    
    # Compute lengths
    text_lens = text_embed.shape[1]
    codec_lens = codec_embed.shape[1]
    
    if non_streaming_mode:
        # Add codec_pad_id to text_embed
        icl_input_embed = text_embed + input_embedding(
            torch.tensor(
                [[config.talker_config.codec_pad_id] * text_lens],
                device=device,
                dtype=text_id.dtype,
            )
        )
        # Concatenate with codec_embed + tts_pad_embed
        icl_input_embed = torch.cat([icl_input_embed, codec_embed + tts_pad_embed], dim=1)
        return icl_input_embed, tts_pad_embed
    else:
        # Streaming mode: align text and codec lengths
        if text_lens > codec_lens:
            return text_embed[:, :codec_lens] + codec_embed, text_embed[:, codec_lens:]
        else:
            text_embed = torch.cat([text_embed] + [tts_pad_embed] * (codec_lens - text_lens), dim=1)
            return text_embed + codec_embed, tts_pad_embed
