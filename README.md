# RDS Poet

An FM receiver plays a short musical phrase while its RadioText display carries one line of a poem. A local model writes the next line and a Strudel pattern together. You can give it a theme, a short sample from a local FM station, or a continuously tuned second receiver. The transmitter and RDS encoder live in [radio/](radio/README.md). The [art precedent review](docs/radio-art-precedents.md) traces related work in radio listening, language models and writing for receiver displays.

## Hear a performance

For a 40-second test with a read-along player, a Strudel score and a model sized for the ThinkPad, start with [Featherweight](examples/featherweight/README.md).

On the Linux machine with GNU Radio, run these from the repository root. The built-in model learns word, rhythm and pitch sequences from the [original verse and note pairs](examples/air-poems.tsv) at startup. It runs in NumPy with a small vocabulary. This command plays through the system speakers.

```sh
/usr/bin/python3 perform.py --model pocket --theme "the last spoon in the kitchen"
```

To send a new Qwen poem to the iPod, pick two local stations you can hear. This example listens to 105.7 and 88.1 MHz before starting the music and RadioText on 100.0 MHz. The local Ollama service must be running.

```sh
/usr/bin/python3 perform.py --sample-mhz 105.7 88.1 --fm --tx-gain-db 40
```

Tune the iPod to 100.0 MHz. On a fifth-generation nano, if the dial skips from 99.9 to 100.1, choose Radio → Radio Regions → Europe for 0.1 MHz steps. Use `--fm 87.5` or another frequency to override the transmit default. [Apple’s tuning table, page 66](https://cdsassets.apple.com/live/6GJYWVAV/user/ma1194_ipod_nano_5th_gen_userguide.pdf#page=66)

The terminal prints each new line and its playable Strudel source. Each cycle lasts ten seconds by default. If a composition takes longer, the current music repeats and its text stays on screen. New phrases take effect on a cycle boundary. Receiver buffering can delay the display. Ctrl+C ends the performance.

For a local audio recording and the matching score transcript, run

```sh
/usr/bin/python3 perform.py --model pocket --wav-out poem.wav --bars 6 > poem.txt
```

## Use the pretrained language model

The default backend calls Ollama at `127.0.0.1:11434` with a schema for the line and score. Install Ollama on the host and start its service (or run `ollama serve` in a separate terminal). Download the compact Qwen model once. Then the performance runs locally. [Ollama for Linux](https://docs.ollama.com/linux) · [Qwen2.5 1.5B model](https://ollama.com/library/qwen2.5:1.5b) · [Structured output API](https://docs.ollama.com/capabilities/structured-outputs)

```sh
ollama pull qwen2.5:1.5b
/usr/bin/python3 perform.py --model qwen2.5:1.5b --theme "what the water heard" --fm --tx-gain-db 40
```

## Let the air suggest the next line

To see what the HackRF hears, sample each station once and print its frequency, tonal balance, pulse and level. This command uses just the receiver.

```sh
/usr/bin/python3 -m radio.listen 105.7 88.1
```

With no receiver option, the live generator uses the theme and an imagined room tone. With `--sample-mhz` and speaker or WAV output, the HackRF takes a fresh two-second sample for each composition, visiting your stations in order. With FM output, it samples every selected frequency once before transmission, then alternates those stored observations as it writes new lines. The terminal prints the cue for each composition. [HackRF One half-duplex specifications](https://hackrf.readthedocs.io/en/latest/hackrf_one.html)

```sh
/usr/bin/python3 perform.py --sample-mhz 105.7 88.1
```

With a second device, an RTL-SDR tuned to the source station can keep supplying new observations while the HackRF transmits. Each model request gets the latest one-second audio window. This mode needs the `rtl_fm` utility. Both modes describe acoustic features such as brightness, level and pulse. They use these measurements as prompts. Speech transcription would be a separate input.

```sh
/usr/bin/python3 perform.py --model qwen2.5:1.5b --rtl-mhz 99.5 --fm --tx-gain-db 40
```

## Score and runtime

Each phrase pairs up to 64 printable ASCII characters with playable Strudel, for example `note("d3 ~ a3 c4").s("triangle").clip(0.35).fast(2)`. The model writes the pitches and rests. The performer shuffles four voices across successive lines, giving sine notes room to ring, clipping square notes into taps, and varying the articulation and speed of the triangle and sawtooth figures. The speaker, WAV and FM paths synthesize the printed score in Python at 48 kHz. You can play the same expression in Strudel with the printed `setcps(...)` tempo. [Strudel synthesizers](https://strudel.cc/learn/synths/) · [Strudel timing controls](https://strudel.cc/learn/time-modifiers/)

The performance requires Python 3, NumPy and GNU Radio 3.10. The FM output also uses gr-osmosdr with HackRF support and the HackRF tools. Radio input additionally uses SciPy. A saved score plays once. Add `--loop-score` for a repeating rehearsal, for example `/usr/bin/python3 perform.py --score examples/rehearsal.jsonl --loop-score`. For the original manual-title and MP3 transmitter commands, see the [radio guide](radio/README.md).
