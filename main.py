#!/usr/bin/env -S uv run --script

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
from pydub import AudioSegment
import argparse

out_dir = 'out'

def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file.")
    parser.add_argument("source_path", type=str, help="Path to the source audio file")
    args = parser.parse_args()

    source_path = args.source_path
    if not os.path.exists(source_path):
        print(f"Source file '{source_path}' does not exist; skipping")
        return

    label, ext = os.path.splitext(os.path.basename(source_path))
    target_transcription_path = os.path.join(out_dir, label + ".txt")

    if os.path.exists(target_transcription_path):
        print(f"Transcription file '{target_transcription_path}' already exists; skipping")
        return

    if ext in ['.mp3', '.wav']:
        target_audio_path = source_path
    else:
        target_audio_path = os.path.join(out_dir, label + ".mp3")
        if not os.path.exists(target_audio_path):
            print(f"Target file '{target_audio_path}' doesn't exist; converting to mp3 and storing locally")
            audio = AudioSegment.from_file(source_path)
            audio.export(target_audio_path, format="mp3")

    device = torch.accelerator.current_accelerator()
    print(f"Using device: {device}")

    torch_dtype = torch.float16 if torch.accelerator.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    generate_kwargs = {
        "max_new_tokens": 445,
        "num_beams": 1,
        "condition_on_prev_tokens": False,
        "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "return_timestamps": True,
        "language": "english"
    }

    result = pipe(target_audio_path, return_timestamps=True, generate_kwargs=generate_kwargs)

    with open(target_transcription_path, 'w', encoding='utf-8') as f:
        f.write(result.get('text', ''))
    print(f"Transcription written to '{target_transcription_path}'")

if __name__ == "__main__":
    main()