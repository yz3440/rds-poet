# RDS Poet

An FM receiver plays a short musical phrase while its RadioText display carries one line of a poem. A local model writes the next line and a Strudel pattern together. You can give it a theme, a short sample from a local FM station, or a continuously tuned second receiver. The transmitter and RDS encoder live in [radio/](radio/README.md). The [art precedent review](docs/radio-art-precedents.md) traces related work in radio listening, language models and writing for receiver displays.

## Hear a performance

For a 40-second test with a read-along player, a Strudel score and a model sized for the ThinkPad, start with [Featherweight](examples/featherweight/README.md).

On the Linux machine with GNU Radio, run these from the repository root. The built-in model learns word, rhythm and pitch sequences from the [original verse and note pairs](examples/air-poems.tsv) at startup. It runs in NumPy with a small vocabulary. This command plays through the system speakers.

```sh
/usr/bin/python3 perform.py --no-scan --model pocket --theme "the last spoon in the kitchen"
```

To send a new Qwen poem to the iPod, run `./setup-ollama.sh` once. The performer surveys the local FM dial, samples stations near 100.0 MHz, then starts the music and RadioText.

```sh
/usr/bin/python3 perform.py --fm --tx-gain-db 40
```

Tune the iPod to 100.0 MHz. On a fifth-generation nano, if the dial skips from 99.9 to 100.1, choose Radio → Radio Regions → Europe for 0.1 MHz steps. Use `--fm 87.5` or another frequency to override the transmit default. [Apple’s tuning table, page 66](https://cdsassets.apple.com/live/6GJYWVAV/user/ma1194_ipod_nano_5th_gen_userguide.pdf#page=66)

The terminal prints each new line and its playable Strudel source. Each cycle lasts ten seconds by default. If a composition takes longer, the current music repeats and its text stays on screen. New phrases take effect on a cycle boundary. Receiver buffering can delay the display. Ctrl+C ends the performance.

For a local audio recording and the matching score transcript, run

```sh
/usr/bin/python3 perform.py --no-scan --model pocket --wav-out poem.wav --bars 6 > poem.txt
```

## Use the pretrained language model

From the repository root on Linux, run the setup script below. It uses Ollama’s official installer when the CLI is missing and downloads the project’s Qwen model when needed. A fresh installation can prompt for sudo. The performer connects to `127.0.0.1:11434`. On a system without an Ollama service, run `ollama serve` in another terminal and rerun setup. [Ollama for Linux](https://docs.ollama.com/linux) · [Qwen2.5 1.5B model](https://ollama.com/library/qwen2.5:1.5b)

```sh
./setup-ollama.sh
/usr/bin/python3 perform.py --model qwen2.5:1.5b --theme "what the water heard" --fm --tx-gain-db 40
```

## Let the air suggest the next line

A live performance starts with a HackRF survey. To see the station list by itself, run

```sh
/usr/bin/python3 -m radio.scan
```

The receiver checks 206 dial positions from 87.5 through 108.0 MHz, then repeats the sweep with shifted tuner centers. A candidate must rise above the measured RF floor in both passes and show FM modulation. The scanner checks tuning offset, high-frequency hiss and the 19 kHz stereo pilot. Mono stations can qualify through modulation and low hiss. The inventory reports frequencies and relative strength at your antenna. Weak signals may come and go between scans.

The performer picks up to two clear source stations closest to the output frequency (100.0 MHz for speakers). For each source it samples the nearest distinct station with clear reception. The printed offset shows how far apart they are. When only weak stations are available it can use those too. With `--sample-mhz`, your choices take priority as the sources. To hear a particular pair of frequencies as *sources* with no survey or neighbor samples, add `--no-scan`. For a theme-only performance, use `--no-scan` alone.

To sample the very next dial position, use `--neighbor-offset-khz 100` or `200`. Negative offsets tune below the source. For example, `--sample-mhz 105.7 --neighbor-offset-khz 200` listens to 105.7 and 105.9 MHz even if the second channel is mostly hiss.

Each observation includes loudness, spectral centroid in Hz and an estimate of attacks per second from rises in the short-time spectrum. Qwen gets both observations. In the default bold arrangement, the source helps select the layer to edit. The neighbor’s brightness and attack density help choose the register and rhythm of the new section. The generated Strudel supplies the audible music.

```sh
/usr/bin/python3 perform.py --sample-mhz 105.7 --transcribe
```

With speaker or WAV output the HackRF takes fresh samples for each composition. With FM output it captures the sources and neighbors once at startup, releases the device, then rotates the stored observations. The terminal prints the observations used for each line. To listen to specific frequencies once and print their measurements, run `/usr/bin/python3 -m radio.listen 105.7 88.1`. [HackRF One half-duplex specifications](https://hackrf.readthedocs.io/en/latest/hackrf_one.html)

With a second device, `--rtl-mhz` uses a continuously tuned RTL-SDR as the source and skips the HackRF survey. It can keep supplying new observations while the HackRF transmits. Each model request gets the latest one-second audio window. This mode needs the `rtl_fm` utility. Both modes describe acoustic features such as brightness, level and pulse. Qwen uses these measurements as a writing prompt. Add `--transcribe` to include any English words Whisper recognizes.

```sh
/usr/bin/python3 perform.py --model qwen2.5:1.5b --rtl-mhz 99.5 --fm --tx-gain-db 40
```

### Use spoken words from the radio

Add `--transcribe` to take six seconds of FM audio per source station, run the local English Whisper model and send its transcript and the acoustic cue to Qwen. To see what the recognizer heard before starting a performance, run:

```sh
/usr/bin/python3 -m radio.listen --transcribe 105.7 88.1
```

For a performance on the iPod with a ten-second reading interval:

```sh
/usr/bin/python3 perform.py --sample-mhz 105.7 88.1 --transcribe --fm 100.0 --tx-gain-db 40 --save-sketch last-set.strudel
```

With a single HackRF the performer captures and transcribes each station **before** switching to FM output, then reuses the observations in turn. For new words throughout the performance, use speaker output with the HackRF or use `--rtl-mhz MHZ --transcribe` with a second receiver during FM transmission. A speech preview on speakers may hold the current phrase while capture and inference finish. Whisper can mishear music and noisy speech. The terminal shows the recognizer’s words alongside the poem so you can compare them. Set `WHISPER_MODEL` to use an English Whisper model installed outside the location below.

### Install whisper.cpp for offline transcription

On the ThinkPad, Git, CMake and a C++ compiler are already installed. To set up another copy on Linux, build the CPU command-line tool and download the English `tiny.en` model. [whisper.cpp release](https://github.com/ggml-org/whisper.cpp/releases/tag/v1.9.5)

```sh
mkdir -p ~/Developer ~/.local/bin
git clone --depth 1 --branch v1.9.5 https://github.com/ggml-org/whisper.cpp.git ~/Developer/whisper.cpp
cd ~/Developer/whisper.cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DWHISPER_BUILD_TESTS=OFF
cmake --build build --target whisper-cli --parallel 3
install -m 755 build/bin/whisper-cli ~/.local/bin/whisper-cli
sh models/download-ggml-model.sh tiny.en
```

Transcribe the included speech sample with the command below. To use your own recording, replace the path after `-f` with a 16 kHz mono PCM WAV file.

```sh
~/.local/bin/whisper-cli -m ~/Developer/whisper.cpp/models/ggml-tiny.en.bin -f ~/Developer/whisper.cpp/samples/jfk.wav -t 3
```

## Edit a Strudel sketch as it plays

The [starter sketch](examples/airwaves.strudel) opens with a bass, a seven-step pulse and a melody. With the default `--edit-size bold`, each new poem line starts a different musical section. Qwen writes a complete lead motif for the current sketch. Two or three sustained bass notes can give way to a clipped square-wave figure repeating four times in the same ten seconds. Brightness and pulse help select the lead. Open and busy sections alternate, including when the station cue repeats.

With three sounding layers, a second voice answers using pitches from its previous motif and a third carries straight through the transition. With one or two sounding layers, only the lead is rewritten. For gradual evolution, use `--edit-size subtle` to change up to three notes or rests in one layer, preserving the instrument and rhythm grid. Both modes apply to live shows and generated Featherweight takes. [Strudel’s stack function](https://strudel.cc/learn/factories/#stack)

To hand-edit alongside Qwen, copy the starter once, then run the performer with your working file:

```sh
cp -n examples/airwaves.strudel my-sketch.strudel
/usr/bin/python3 perform.py --sketch my-sketch.strudel --sample-mhz 105.7 88.1 --transcribe --save-sketch last-set.strudel
```

Edit and save a layer in `my-sketch.strudel` during the set. At the next composition the performer merges the changed layer with the other evolving parts. Adding or removing layers loads the new arrangement. Ctrl+C saves the last played version to `last-set.strudel`. Use that file as the next `--sketch` to continue the piece.

The FM and WAV renderer supports a single pitched pattern or a flat `stack(...)` of up to eight. Each layer takes 1–16 space-separated notes or rests (`~`), a sine, triangle, square or sawtooth synth, and optional `.clip(0–1)`, `.fast(1–4)` and `.gain(0–1)` in that order. The performer sets the cycle duration with `--line-seconds` (default 10). The exported source also plays in Strudel. [Strudel synthesizers](https://strudel.cc/learn/synths/) · [Strudel timing controls](https://strudel.cc/learn/time-modifiers/)

The performance requires Python 3, NumPy and GNU Radio 3.10. The FM output also uses gr-osmosdr with HackRF support and the HackRF tools. Radio input additionally uses SciPy. A saved score plays once. Add `--loop-score` for a repeating rehearsal, for example `/usr/bin/python3 perform.py --score examples/rehearsal.jsonl --loop-score`. For the original manual-title and MP3 transmitter commands, see the [radio guide](radio/README.md).
