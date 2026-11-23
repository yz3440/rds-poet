# RDS Poet

An FM receiver plays a short musical phrase while its RadioText display carries one line of a poem. A local model writes the next line and a Strudel pattern together. You can give it a theme, a short sample from a local FM station, or a continuously tuned second receiver. The transmitter and RDS encoder live in [radio/](radio/README.md). The [art precedent review](docs/radio-art-precedents.md) traces related work in radio listening, language models and writing for receiver displays.

## Hear a performance

For a 40-second test with a read-along player, a Strudel score and a model sized for the ThinkPad, start with [Featherweight](examples/featherweight/README.md).

On the Linux machine with GNU Radio, run these from the repository root. The built-in model learns word, rhythm and pitch sequences from the [original verse and note pairs](examples/air-poems.tsv) at startup. It runs in NumPy with a small vocabulary. This command plays through the system speakers.

```sh
/usr/bin/python3 perform.py --model pocket --theme "the last spoon in the kitchen"
```

To send the music and poem to the iPod using the tested gain setting, run

```sh
/usr/bin/python3 perform.py --model pocket --fm 87.5 --tx-gain-db 40 --line-seconds 10
```

Tune the iPod to 87.5 MHz. The terminal prints each new line and its playable Strudel source. Each cycle lasts ten seconds by default. If a composition takes longer, the current music repeats and its text stays on screen. New phrases take effect on a cycle boundary. Receiver buffering can delay the display. Ctrl+C ends the performance.

For a local audio recording and the matching score transcript, run

```sh
/usr/bin/python3 perform.py --model pocket --wav-out poem.wav --bars 6 > poem.txt
```

## Use the pretrained language model

The default backend calls Ollama at `127.0.0.1:11434` with a schema for the line and score. Install Ollama on the host and start its service (or run `ollama serve` in a separate terminal). Download the compact Qwen model once. Then the performance runs locally. [Ollama for Linux](https://docs.ollama.com/linux) · [Qwen2.5 1.5B model](https://ollama.com/library/qwen2.5:1.5b) · [Structured output API](https://docs.ollama.com/capabilities/structured-outputs)

```sh
ollama pull qwen2.5:1.5b
/usr/bin/python3 perform.py --model qwen2.5:1.5b --theme "what the water heard" --fm 87.5 --tx-gain-db 40
```

## Let the air suggest the next line

HackRF One is half-duplex. With one device, `--sample-mhz` listens for two seconds, closes the receiver, then opens the transmitter. Every line can draw on that first impression and the history of the poem. Replace 99.5 with a public FM station you can hear where you are. [HackRF One specifications](https://hackrf.readthedocs.io/en/latest/hackrf_one.html)

```sh
/usr/bin/python3 perform.py --model pocket --sample-mhz 99.5 --fm 87.5 --tx-gain-db 40
```

With a second device, an RTL-SDR tuned to the source station can keep supplying new observations while the HackRF transmits. Each model request gets the latest one-second audio window. This mode needs the `rtl_fm` utility. Both modes describe acoustic features such as brightness, level and pulse. They use these measurements as prompts. Speech transcription would be a separate input.

```sh
/usr/bin/python3 perform.py --model qwen2.5:1.5b --rtl-mhz 99.5 --fm 87.5 --tx-gain-db 40
```

## Score and runtime

Both generators emit a line of up to 64 printable ASCII characters and a Strudel expression like `note("d3 ~ a3 c4").s("triangle")`. The audio renderer implements these monophonic, equal-duration notes and rests in Python, with sine and triangle oscillators at 48 kHz. You can also play the printed source in Strudel with the printed `setcps(...)` tempo. Strudel’s own synth envelopes can sound different. [Strudel synthesizers](https://strudel.cc/learn/synths/)

The performance requires Python 3, NumPy and GNU Radio 3.10. The FM output also uses gr-osmosdr with HackRF support and the HackRF tools. Radio input additionally uses SciPy. Rehearse the same three phrases repeatedly with `/usr/bin/python3 perform.py --score examples/rehearsal.jsonl`. For the original manual-title and MP3 transmitter commands, see the [radio guide](radio/README.md).
