#!/usr/bin/env python3
"""Verify live RadioText changes in memory, with no Qt or SDR device."""

import time

from gnuradio import blocks, gr
import pmt
import rds


def wait_for_title(messages, title, start_index=0):
    deadline = time.monotonic() + 8
    index = start_index
    while time.monotonic() < deadline:
        while index < messages.num_messages():
            message = messages.get_message(index)
            index += 1
            if pmt.to_long(pmt.tuple_ref(message, 0)) == 4:
                decoded = pmt.symbol_to_string(pmt.tuple_ref(message, 1)).rstrip()
                if decoded == title:
                    print(f"Decoded RadioText: {title}", flush=True)
                    return index
        time.sleep(0.02)
    raise RuntimeError(f"Timed out waiting for RadioText: {title}")


def main():
    initial_title = "The first song"
    encoder = rds.encoder(
        0, 14, True, "RDS POET", 89.8e6, True, False, 13, 3, 147, initial_title
    )
    # Pace the probe faster than real time while bounding queued old groups.
    throttle = blocks.throttle(gr.sizeof_char, 11875)
    encoder.set_max_output_buffer(4096)
    throttle.set_max_output_buffer(4096)
    decoder = rds.decoder(False, False)
    parser = rds.parser(False, False, 0)
    messages = blocks.message_debug()

    graph = gr.top_block()
    graph.connect(encoder, throttle, decoder)
    graph.msg_connect((decoder, "out"), (parser, "in"))
    graph.msg_connect((parser, "out"), (messages, "store"))

    graph.start(1040)
    try:
        index = wait_for_title(messages, initial_title)
        for title in ("A different song", "Final track"):
            command = f"text {title}\n".encode("ascii")
            payload = pmt.init_u8vector(len(command), list(command))
            encoder.to_basic_block()._post(
                pmt.intern("rds in"), pmt.cons(pmt.PMT_NIL, payload)
            )
            index = wait_for_title(messages, title, index)
    finally:
        graph.stop()
        graph.wait()


if __name__ == "__main__":
    main()
