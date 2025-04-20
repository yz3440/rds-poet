#!/usr/bin/env python3
"""
FM Transmitter with RDS encoding using HackRF One
Transmits a message "RDS Test" over FM radio at 106.9 MHz
"""

from python_hackrf import pyhackrf
import numpy as np
import time
import math
import struct
import sys

# FM and RDS Parameters
FM_FREQ = 106.9e6  # Broadcast frequency in Hz (106.9 MHz)
SAMPLE_RATE = 2e6  # Sample rate
TX_GAIN = 0  # Transmit gain (0-47)
AUDIO_FREQ = 1000  # 1 kHz tone
FM_DEV = 75e3  # FM deviation for main carrier
RDS_DEV = 2e3  # FM deviation for RDS subcarrier
RDS_FREQ = 57e3  # RDS subcarrier frequency (57 kHz)

# RDS Constants
PI_CODE = 0x1234  # Program Identification code
PTY = 0  # Program Type (0 = None)
PS_NAME = "RDS Test"  # Program Service name (max 8 chars)
RT_MESSAGE = "RDS Test"  # Radio Text (max 64 chars)


class RDSEncoder:
    """RDS (Radio Data System) encoder for FM broadcast"""

    def __init__(self, pi_code, ps_name, rt_message, pty=0):
        self.pi_code = pi_code
        self.pty = pty
        self.ps_name = ps_name.ljust(8)[:8]  # Pad/truncate to 8 chars
        self.rt_message = rt_message.ljust(64)[:64]  # Pad/truncate to 64 chars
        self.group_counter = 0

        # CRC polynomial for RDS: x^16 + x^12 + x^5 + 1
        self.crc_poly = 0x1021
        self.offset_words = [0x0FC, 0x198, 0x168, 0x1B4, 0x350, 0x3AC]

    def calculate_crc(self, data):
        """Calculate the CRC for RDS block"""
        crc = 0
        for i in range(16):
            bit = (data >> (15 - i)) & 1
            msb = (crc >> 15) & 1
            crc = ((crc << 1) & 0xFFFF) | bit
            if msb:
                crc ^= self.crc_poly
        return crc

    def make_block(self, data, block_num):
        """Create a block with error correction"""
        # Add the offset word depending on the block position
        check_word = self.calculate_crc(data) ^ self.offset_words[block_num]
        return (data << 10) | check_word

    def get_group_0A(self):
        """Generate Group 0A - Basic tuning and switching information"""
        blocks = [0] * 4

        # Block 1: PI code
        blocks[0] = self.make_block(self.pi_code, 0)

        # Block 2: Group type (0A) + TP flag + PTY + TA flag + MS flag + DI bit + PS segment address
        ps_segment = self.group_counter % 4
        blocks[1] = self.make_block(
            (0 << 15)
            | (0 << 14)
            | (self.pty << 5)
            | (0 << 4)
            | (1 << 3)
            | (0 << 2)
            | ps_segment,
            1,
        )

        # Block 3: Alternative Frequencies (not used here)
        blocks[2] = self.make_block(0, 2)

        # Block 4: Two PS name characters
        char_pos = ps_segment * 2
        char1 = ord(self.ps_name[char_pos]) if char_pos < len(self.ps_name) else 32
        char2 = (
            ord(self.ps_name[char_pos + 1]) if char_pos + 1 < len(self.ps_name) else 32
        )
        blocks[3] = self.make_block((char1 << 8) | char2, 3)

        self.group_counter += 1
        return blocks

    def get_group_2A(self):
        """Generate Group 2A - Radio Text"""
        blocks = [0] * 4

        # Block 1: PI code
        blocks[0] = self.make_block(self.pi_code, 0)

        # Block 2: Group type (2A) + TP flag + PTY + RT segment address
        rt_segment = (self.group_counter // 4) % 16
        blocks[1] = self.make_block(
            (2 << 15) | (0 << 14) | (self.pty << 5) | (0 << 4) | rt_segment, 1
        )

        # Block 3 & 4: Four RT characters
        char_pos = rt_segment * 4
        char1 = (
            ord(self.rt_message[char_pos]) if char_pos < len(self.rt_message) else 32
        )
        char2 = (
            ord(self.rt_message[char_pos + 1])
            if char_pos + 1 < len(self.rt_message)
            else 32
        )
        char3 = (
            ord(self.rt_message[char_pos + 2])
            if char_pos + 2 < len(self.rt_message)
            else 32
        )
        char4 = (
            ord(self.rt_message[char_pos + 3])
            if char_pos + 3 < len(self.rt_message)
            else 32
        )

        blocks[2] = self.make_block((char1 << 8) | char2, 2)
        blocks[3] = self.make_block((char3 << 8) | char4, 3)

        self.group_counter += 1
        return blocks

    def differential_encode(self, bits):
        """Apply differential encoding to the bit stream"""
        out_bits = []
        prev_bit = 0
        for bit in bits:
            out_bits.append(bit ^ prev_bit)
            prev_bit = bit
        return out_bits

    def biphase_encode(self, bits):
        """Apply biphase (Manchester) encoding"""
        biphase = []
        for bit in bits:
            if bit:
                biphase.extend([1, 0])  # '1' becomes '10'
            else:
                biphase.extend([0, 1])  # '0' becomes '01'
        return biphase

    def get_next_group_bits(self):
        """Get the next group as bits with all encoding applied"""
        # Alternate between 0A (PS) and 2A (RT) groups
        if self.group_counter % 2 == 0:
            blocks = self.get_group_0A()
        else:
            blocks = self.get_group_2A()

        # Convert blocks to bits
        bits = []
        for block in blocks:
            # Each block is 26 bits
            for i in range(25, -1, -1):
                bits.append((block >> i) & 1)

        # Apply differential and biphase encoding
        diff_bits = self.differential_encode(bits)
        biphase_bits = self.biphase_encode(diff_bits)

        return biphase_bits


def generate_samples(duration, sample_rate):
    """Generate samples for the specified duration"""
    num_samples = int(duration * sample_rate)
    samples = np.zeros(num_samples, dtype=np.complex64)

    # Generate 1 kHz audio tone
    audio_phase = 0
    audio_samples = np.zeros(num_samples)
    for i in range(num_samples):
        t = i / sample_rate
        audio_samples[i] = 0.5 * np.sin(2 * np.pi * AUDIO_FREQ * t)

    # Create RDS encoder and generate bit stream
    rds_encoder = RDSEncoder(PI_CODE, PS_NAME, RT_MESSAGE, PTY)
    rds_phase = 0
    rds_bit_phase = 0

    # RDS bit rate is 1187.5 bits per second
    rds_bit_period = sample_rate / 1187.5
    rds_bits = []
    current_bit = 0

    # FM modulation (carrier + audio + RDS)
    fm_phase = 0

    for i in range(num_samples):
        t = i / sample_rate

        # Generate new RDS bit if needed
        if rds_bit_phase >= rds_bit_period:
            rds_bit_phase -= rds_bit_period
            if not rds_bits:
                rds_bits = rds_encoder.get_next_group_bits()
            current_bit = rds_bits.pop(0) if rds_bits else 0
        rds_bit_phase += 1

        # RDS subcarrier is a 57 kHz signal
        rds_subcarrier = current_bit * np.sin(2 * np.pi * RDS_FREQ * t + rds_phase)

        # Combine audio and RDS
        modulation = audio_samples[i] + 0.1 * rds_subcarrier

        # Apply FM modulation
        fm_phase += 2 * np.pi * (FM_DEV * modulation) / sample_rate
        samples[i] = np.exp(1j * fm_phase)

    return samples


def tx_callback(device, buffer, buffer_length, valid_length):
    """Callback function for HackRF transmission"""
    global tx_buffer, tx_idx, tx_buffer_size

    # Copy samples from our buffer to the device buffer
    bytes_to_copy = min(buffer_length, (tx_buffer_size - tx_idx) * 2)
    samples_to_copy = bytes_to_copy // 2

    if samples_to_copy <= 0:
        return -1  # End of transmission

    # Prepare interleaved I/Q
    for i in range(samples_to_copy):
        buffer[2 * i] = int(np.real(tx_buffer[tx_idx + i]) * 127)
        buffer[2 * i + 1] = int(np.imag(tx_buffer[tx_idx + i]) * 127)

    tx_idx += samples_to_copy

    # If we reached the end of our buffer, start from beginning
    if tx_idx >= tx_buffer_size:
        tx_idx = 0

    return 0


def main():
    print("Starting FM RDS Transmitter...")
    print(f"Frequency: {FM_FREQ/1e6} MHz")
    print(f"RDS Message: {PS_NAME} / {RT_MESSAGE}")

    # Initialize HackRF
    pyhackrf.pyhackrf_init()
    try:
        sdr = pyhackrf.pyhackrf_open()
        if not sdr:
            print("Failed to open HackRF device")
            return

        # Configure HackRF
        sdr.pyhackrf_set_freq(FM_FREQ)
        sdr.pyhackrf_set_sample_rate(SAMPLE_RATE)
        filter_bw = pyhackrf.pyhackrf_compute_baseband_filter_bw_round_down_lt(
            SAMPLE_RATE
        )
        sdr.pyhackrf_set_baseband_filter_bandwidth(filter_bw)

        # Set transmit gain (0-47)
        sdr.pyhackrf_set_txvga_gain(TX_GAIN)

        # Set up callback
        global tx_buffer, tx_idx, tx_buffer_size

        # Generate 5 seconds of samples
        duration = 5.0
        tx_buffer = generate_samples(duration, SAMPLE_RATE)
        tx_buffer_size = len(tx_buffer)
        tx_idx = 0

        # Start transmission
        sdr.set_tx_callback(tx_callback)
        sdr.pyhackrf_start_tx()

        print("Transmitting... Press Ctrl+C to stop")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping transmission.")

        # Stop and clean up
        sdr.pyhackrf_stop_tx()
        sdr.pyhackrf_close()

    except Exception as e:
        print(f"Error: {e}")

    finally:
        pyhackrf.pyhackrf_exit()


if __name__ == "__main__":
    main()
