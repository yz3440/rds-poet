#!/bin/sh
# Set up the local language model used by perform.py and Featherweight.
set -eu

model=qwen2.5:1.5b
export OLLAMA_HOST=127.0.0.1:11434

if ! command -v ollama >/dev/null 2>&1; then
    installer=$(mktemp)
    trap 'rm -f "$installer"' EXIT
    echo "Downloading the official Ollama installer..."
    curl --fail --silent --show-error --location --retry 2 --output "$installer" https://ollama.com/install.sh
    sh "$installer"
fi

if ! ollama list >/dev/null 2>&1; then
    if [ "$(systemctl show ollama.service --property=LoadState --value 2>/dev/null || true)" != loaded ]; then
        echo "Start 'ollama serve' in another terminal, then run this script again." >&2
        exit 1
    fi
    if ! systemctl is-active --quiet ollama.service; then
        systemctl start ollama.service || sudo systemctl start ollama.service
    fi
    attempts=0
    until ollama list >/dev/null 2>&1; do
        attempts=$((attempts + 1))
        if [ "$attempts" -ge 15 ]; then
            echo "Ollama did not respond on $OLLAMA_HOST. Check 'systemctl status ollama.service'." >&2
            exit 1
        fi
        sleep 1
    done
fi

if ! ollama show "$model" >/dev/null 2>&1; then
    ollama pull "$model"
fi
echo "Ollama is ready at $OLLAMA_HOST with $model."
