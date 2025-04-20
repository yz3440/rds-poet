#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import subprocess

# Constants
FREQUENCY = 106.9e6  # 106.9 MHz
SAMPLE_RATE = 2e6  # 2 MHz sample rate
AUDIO_SAMPLE_RATE = 44100  # Standard audio sample rate
MP3_PATH = "assets/never-gonna-give-you-up.mp3"
WAV_PATH = "assets/temp_audio.wav"
IQ_PATH = "assets/fm_samples.bin"  # File to store IQ samples


def mp3_to_wav(mp3_path, wav_path):
    """Convert MP3 to WAV file format"""
    print(f"Converting {mp3_path} to WAV format...")
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_frame_rate(int(AUDIO_SAMPLE_RATE))
    audio = audio.set_channels(1)  # Convert to mono
    audio.export(wav_path, format="wav")
    return wav_path


def prepare_audio_samples():
    """Convert audio to the right format for transmission"""
    # Convert MP3 to WAV if needed
    if not os.path.exists(WAV_PATH):
        mp3_to_wav(MP3_PATH, WAV_PATH)

    # Load WAV file
    print("Loading audio data...")
    sample_rate, audio_data = wavfile.read(WAV_PATH)

    # Convert to float in range [-1, 1]
    if audio_data.dtype != np.float32:
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0

    # If stereo, convert to mono
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Normalize to use full dynamic range
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Resample if needed
    if sample_rate != AUDIO_SAMPLE_RATE:
        print(f"Resampling audio from {sample_rate}Hz to {AUDIO_SAMPLE_RATE}Hz...")
        from scipy import signal

        new_length = int(len(audio_data) * AUDIO_SAMPLE_RATE / sample_rate)
        audio_data = signal.resample(audio_data, new_length)

    return audio_data


def fm_modulate(audio_data, max_deviation=75000):
    """
    FM modulate the audio data

    Args:
        audio_data: Audio data normalized to [-1, 1]
        max_deviation: Maximum frequency deviation in Hz (75kHz is standard for FM)

    Returns:
        Complex samples ready for transmission
    """
    print("Performing FM modulation...")
    # Calculate the modulation index
    modulation_index = max_deviation / AUDIO_SAMPLE_RATE * 2 * np.pi

    # Integrate audio to get phase
    phase = np.cumsum(audio_data) * modulation_index

    # Convert to complex samples
    samples = np.exp(1j * phase)

    # Upsample to match HackRF sample rate
    upsample_factor = int(SAMPLE_RATE / AUDIO_SAMPLE_RATE)

    # Simple zero-order hold upsampling
    fm_signal = np.repeat(samples, upsample_factor)

    # Normalize to range [-1, 1] for both I and Q
    max_val = np.max(np.abs(fm_signal))
    fm_signal = fm_signal / max_val * 0.95  # Add a little headroom

    return fm_signal


def save_iq_samples(samples, iq_path):
    """Save IQ samples to a binary file for HackRF transmission"""
    print(f"Saving IQ samples to {iq_path}...")

    # Convert complex samples to format expected by HackRF
    # HackRF expects 8-bit signed I/Q samples interleaved
    samples_i = np.real(samples) * 127
    samples_q = np.imag(samples) * 127

    # Interleave I and Q samples
    samples_iq = np.empty(samples.shape[0] * 2, dtype=np.int8)
    samples_iq[0::2] = samples_i.astype(np.int8)
    samples_iq[1::2] = samples_q.astype(np.int8)

    # Save to file
    samples_iq.tofile(iq_path)
    return iq_path


def transmit_with_hackrf_transfer(iq_path):
    """Use hackrf_transfer command-line tool to transmit IQ samples"""
    freq_mhz = FREQUENCY / 1e6
    sample_rate_mhz = SAMPLE_RATE / 1e6

    cmd = [
        "hackrf_transfer",
        "-t",
        iq_path,  # Transmit from file
        "-f",
        str(int(FREQUENCY)),  # Frequency in Hz
        "-s",
        str(int(SAMPLE_RATE)),  # Sample rate
        "-a",
        "1",  # Enable amp
        "-x",
        "40",  # TX VGA gain (0-47 dB)
        "-R",  # Repeat transmission (loop)
    ]

    print(f"Starting transmission at {freq_mhz} MHz...")
    print("Press Ctrl+C to stop.")

    try:
        process = subprocess.Popen(cmd)
        # Keep the script running until Ctrl+C
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping transmission...")
        process.terminate()
        process.wait()
        print("Transmission stopped.")


def main():
    print("Starting FM audio transmission process...")

    # Check if MP3 file exists
    if not os.path.exists(MP3_PATH):
        print(f"Error: MP3 file not found at {MP3_PATH}")
        sys.exit(1)

    # Prepare audio
    audio_data = prepare_audio_samples()

    # Modulate
    fm_signal = fm_modulate(audio_data)

    # Save IQ samples
    save_iq_samples(fm_signal, IQ_PATH)

    # Transmit using hackrf_transfer tool
    transmit_with_hackrf_transfer(IQ_PATH)


if __name__ == "__main__":
    main()
