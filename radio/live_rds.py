"""RDS station name (0B) and live RadioText (2A) for the headless transmitter."""

import numpy as np
from gnuradio import gr


DEFAULT_LINE_SECONDS = 10


def validate_title(title):
    if not (1 <= len(title) <= 64 and title.strip() and title.isascii() and title.isprintable()):
        raise ValueError("Use 1 to 64 printable ASCII characters for the title")
    return title


def group_bits(words, offsets):
    """Append each block's 10-bit checkword and serialize MSB first."""
    result = []
    for word, offset in zip(words, offsets):
        remainder = word << 10
        for bit in range(25, 9, -1):
            if remainder & (1 << bit):
                remainder ^= 0x5B9 << (bit - 10)
        block = (word << 10) | (remainder ^ offset)
        result.extend((block >> bit) & 1 for bit in range(25, -1, -1))
    return np.array(result, dtype=np.uint8)


class LiveRds(gr.sync_block):
    """Emit one differentially encoded biphase group per scheduler call."""

    def __init__(self, title):
        super().__init__("Live RDS", in_sig=None, out_sig=[np.float32])
        self.pending_title = validate_title(title)
        self.active_title = None
        self.text_ab = 0
        self.rt_segment = 0
        self.ps_segment = 0
        self.group_slot = 0
        self.last_bit = 0
        self.set_output_multiple(208)
        self.set_max_output_buffer(4096)

    def set_title(self, title):
        self.pending_title = validate_title(title)

    def work(self, input_items, output_items):
        pi = 0xD393
        if self.group_slot == 0:
            segment = self.ps_segment
            name = b"RDS POET"[2 * segment:2 * segment + 2]
            # Group 0B has the PI in block C' and no alternate-frequency list.
            words = (pi, 0x0808 | segment, pi, int.from_bytes(name, "big"))
            bits = group_bits(words, (0x0FC, 0x198, 0x350, 0x1B4))
            self.ps_segment = (segment + 1) % 4
        else:
            title = self.pending_title
            if title != self.active_title:
                if self.active_title is not None:
                    self.text_ab ^= 1
                self.active_title = title
                self.rt_segment = 0
                self.text = (title.encode("ascii") + b"\r")[:64]
            start = 4 * self.rt_segment
            chunk = self.text[start:start + 4].ljust(4, b" ")
            header = 0x2000 | (self.text_ab << 4) | self.rt_segment
            words = (pi, header, int.from_bytes(chunk[:2], "big"), int.from_bytes(chunk[2:], "big"))
            bits = group_bits(words, (0x0FC, 0x198, 0x168, 0x1B4))
            self.rt_segment += 1
            if self.rt_segment * 4 >= len(self.text):
                self.rt_segment = 0
        self.group_slot = (self.group_slot + 1) % 4

        encoded = np.bitwise_xor.accumulate(bits) ^ self.last_bit
        self.last_bit = int(encoded[-1])
        first_chip = 2.0 * encoded - 1.0
        out = output_items[0][:208]
        out[::2] = first_chip
        out[1::2] = -first_chip
        return len(out)
