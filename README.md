# RDS Poet

## HackRF live title test at 87.5 MHz

On the machine with the HackRF One attached, run the following from this
directory. Use a receiver that can tune 87.5 MHz and show RDS RadioText.

```sh
/usr/bin/python3 live_tx.py --freq-mhz 87.5 --title "First song"
```

Listen for the 440 Hz test tone and look for the station name `RDS POET`.
Type a new song title in the same terminal and press Enter. Allow a few
seconds for the receiver to assemble it. Type another title to repeat.
Ctrl+C or end of input stops the transmitter.

Add `--wav path/to/song.wav` to loop a WAV during the test. Stereo files
are mixed to mono. This entry point takes up to 64 printable ASCII
characters per title and toggles the RadioText A/B flag when the transmitted
title changes. It needs GNU Radio 3.10, NumPy, gr-osmosdr with HackRF support
and the `hackrf_info` utility. It uses an 8.208 MS/s IQ stream, with the RF
amplifier off and 8 dB TX VGA gain.

## Earlier gr-rds transmitter and encoder probe

Song titles belong in RDS RadioText (group 2A). The Program Service name
(PS) is the eight-character station label.

The GNU Radio runtime can change RadioText while a flowgraph runs without
GNU Radio Companion or a Qt window. Run the in-memory verification with a
Python interpreter that can import GNU Radio 3.10 and the gr-rds encoder:

```sh
env -u DISPLAY -u WAYLAND_DISPLAY /usr/bin/python3 verify_live_radiotext.py
```

The probe posts two updates to the same encoder and waits for the decoder and
parser to recover each title from the RDS bitstream. It uses no SDR hardware.

The existing `rds_tx.py` transmitter still opens a Qt window. Its existing TCP
Socket PDU block on port 52001 is wired to the same encoder input tested by
the probe. With that transmitter already running, a local client can send the
encoder's RadioText command. Use one short command per connection:

```sh
printf 'text Artist - Song Title\n' | nc -N 127.0.0.1 52001
```

Use printable ASCII and at most 64 bytes for the title. The existing listener
binds to all interfaces. GNU Radio's TCP Socket PDU uses read chunks as PDUs,
so the command has no guaranteed TCP framing or application acknowledgment.
The probe tests the PDU handler directly.

The tested gr-rds encoder rebuilds its groups after a command, allocates new
group buffers without freeing the old ones, and keeps the group 2A RadioText
A/B flag fixed. The software parser can recover subsequent titles. Receiver
behavior and long-running operation still need validation.
