"""Compatibility entry point for the headless transmitter."""

from radio.live_tx import Transmitter, hackrf_sink, main

if __name__ == "__main__":
    main()
