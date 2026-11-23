# RDS Poet

## HackRF live title test at 87.5 MHz

On the machine with the HackRF One attached, run the following from the repository
root. Use a receiver that can tune 87.5 MHz and show RDS RadioText.

```sh
/usr/bin/python3 -m radio.live_tx --freq-mhz 87.5 --title "First song"
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
amplifier off and a default TX VGA gain of 8 dB. For a close-range reception
comparison, add `--tx-gain-db 14` and keep the iPod and headphone cord in
the same position. The startup line reports the configured gain.
[HackRF TX gain controls](https://hackrf.readthedocs.io/en/latest/setting_gain.html)

### Play the local MP3 with less receiver hiss

Loop the bundled song with two-second RadioText rotation:

```sh
/usr/bin/python3 -m radio.live_tx --freq-mhz 87.5 --tx-gain-db 40 --mono \
  --audio-file assets/never-gonna-give-you-up.mp3 \
  --lyrics assets/rotation-demo.txt --line-seconds 2
```

`--mono` omits the 19 kHz stereo pilot while carrying the music and 57 kHz
RDS subcarrier. This can reduce hiss from a receiver's stereo decoder.
Compare with and without `--mono` at the same gain and distance. This
checkout's installed libsndfile reads the MP3 directly. `--wav` remains
an alias for `--audio-file`.

### Rotate lyrics

To loop the three original sample lines at two seconds per line, run

```sh
/usr/bin/python3 -m radio.live_tx --freq-mhz 87.5 --tx-gain-db 14 --lyrics assets/rotation-demo.txt --line-seconds 2
```

Pass your own text file to `--lyrics`, one line per display update, up to
64 printable ASCII characters each. Blank lines are skipped. Two seconds is
the default. Use e.g. `--line-seconds 5` to give the receiver more time.
Rotation uses a fixed interval and repeats until Ctrl+C. Add `--wav` to
loop audio alongside the text. For manual entry, omit both rotation options.

### If the iPod nano is silent

On the fifth-generation nano, plug wired headphones into the headphone jack,
open Radio and check a familiar FM station first. The headphone cord is the
FM antenna and radio audio plays through the headphones. Then manually tune
87.5 MHz. [Apple's fifth-generation nano guide, chapter 6](https://cdsassets.apple.com/live/6GJYWVAV/user/ma1194_ipod_nano_5th_gen_userguide.pdf)

Stop the WAV run with Ctrl+C and use the command above without `--wav` to
send a continuous 440 Hz tone. Listen for the tone and check for `RDS POET`.
A queued title in the terminal only confirms that the transmitter accepted
the input. The iPod display and audio are the checks for actual reception.

## Earlier gr-rds transmitter and encoder probe

Song titles belong in RDS RadioText (group 2A). The Program Service name
(PS) is the eight-character station label.

The GNU Radio runtime can change RadioText while a flowgraph runs without
GNU Radio Companion or a Qt window. Run the in-memory verification with a
Python interpreter that can import GNU Radio 3.10 and the gr-rds encoder:

```sh
env -u DISPLAY -u WAYLAND_DISPLAY /usr/bin/python3 -m radio.verify_live_radiotext
```

The probe posts two updates to the same encoder and waits for the decoder and
parser to recover each title from the RDS bitstream. It uses no SDR hardware.

The existing `radio/rds_tx.py` transmitter still opens a Qt window. Its existing TCP
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

## Precedents for reading over FM

- **Midnight Bill (2011).** Joey Orr and Chris Campbell collaborated on a
  WREK Atlanta broadcast that paired a 15-minute audio piece with timed
  text on radio displays. The script identified speakers and dates and
  supplied extra narrative details while people listened.
  [Artist's exhibition record](https://joeyorr.com/cv/) ·
  [Interview with Campbell about the broadcast](https://www.radioworld.com/news-and-business/wrek-beefs-up-rds-hd-text-display-for-special-program)
- **Lyrics over RadioText (filed 1993).** A South Korean utility model
  described sending the words of a song in the RDS RadioText field for
  display during musical accompaniment. The filing documents a proposed
  design. [KR960001675Y1](https://patents.google.com/patent/KR960001675Y1/en)
- **NPR Labs Gulf Coast field tests (2014).** Deaf and hard-of-hearing
  participants read emergency messages sent over FM RDS to a dedicated
  receiver and displayed on a tablet. Researchers compared blocks of prose
  with versions that inserted line breaks at changes of subject.
  [Field report, sections on message design and results](https://www.nrscstandards.org/standards-and-guidelines/documents/references/Accessible_Common_Alerting_Protocol_Radio_Data_System_Demonstration_-_Gulf_Coast_States.pdf)
