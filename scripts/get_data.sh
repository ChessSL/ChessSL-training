#!/bin/bash

# ── Configuration ─────────────────────────────────────────────
DOWNLOAD_DIR="temp_chess_data"
EXTRACT_DIR="/home/alvinng/ChessSL_data/test80"
BASE_URL="https://data.lczero.org/files/training_data/test80"
BASE_NAME="training-run1-test80-20250801"
# ──────────────────────────────────────────────────────────────

mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$EXTRACT_DIR"

echo "==> Downloading files to '$DOWNLOAD_DIR'..."

for i in $(seq -w 0 23); do
    FILENAME="${BASE_NAME}-$(printf "%02d" "$i")17.tar"
    URL="${BASE_URL}/${FILENAME}"
    DEST="${DOWNLOAD_DIR}/${FILENAME}"

    echo ">>> Downloading: $FILENAME"
    curl -L --progress-bar -o "$DEST" "$URL"

    if [ $? -ne 0 ]; then
        echo "!!! Failed to download $FILENAME, skipping..."
    else
        echo "    Saved to $DEST"
    fi
done

echo ""
echo "==> All downloads done. Extracting to '$EXTRACT_DIR'..."

for TAR_FILE in "$DOWNLOAD_DIR"/*.tar; do
    [ -f "$TAR_FILE" ] || continue
    echo ">>> Extracting: $TAR_FILE"
    tar -xf "$TAR_FILE" -C "$EXTRACT_DIR"

    if [ $? -ne 0 ]; then
        echo "!!! Failed to extract $TAR_FILE"
    else
        echo "    Done."
    fi
done

echo ""
echo "==> All done! Files extracted to '$EXTRACT_DIR'."
