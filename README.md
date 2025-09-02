You'll need the [uv](https://docs.astral.sh/uv/) Python package manager installed:

```sh
brew install uv
```

For `m4a` to `mp3` conversion with `pydub`, you'll need to install `ffmpeg`:

```sh
brew install ffmpeg
```

To run:

```sh
./main.py /path/to/audio.mp3
```

If the source file is `mp3` or `wav`, it will be used directly. If the source file is `m4a`, it will be converted to `mp3` and the copy will be stored in the `./out` directory. The transcription plain text file will be written to the `./out` directory with the same file name as the source file, but with the `.txt` extension.
