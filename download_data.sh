#!/usr/bin/env bash
# data/download_data.sh
# Downloads RockYou and LinkedIn password datasets.
# RockYou is available from SecLists (no login needed).
# LinkedIn reversed hashes require manual download from HIBP - instructions below.

set -e

RAW_DIR="$(dirname "$0")/raw"
mkdir -p "$RAW_DIR"

echo "=== Downloading RockYou dataset ==="
# Source: Daniel Miessler's SecLists (MIT licensed for research)
ROCKYOU_URL="https://github.com/danielmiessler/SecLists/raw/master/Passwords/Leaked-Databases/rockyou.txt.tar.gz"

if [ ! -f "$RAW_DIR/rockyou.txt" ]; then
    echo "Fetching rockyou.txt.tar.gz ..."
    curl -L "$ROCKYOU_URL" -o "$RAW_DIR/rockyou.txt.tar.gz"
    echo "Extracting ..."
    tar -xzf "$RAW_DIR/rockyou.txt.tar.gz" -C "$RAW_DIR"
    rm "$RAW_DIR/rockyou.txt.tar.gz"
    echo "RockYou saved to $RAW_DIR/rockyou.txt"
else
    echo "rockyou.txt already exists, skipping."
fi

echo ""
echo "=== LinkedIn Dataset (manual step required) ==="
echo "The LinkedIn cleartext subset is available via Have I Been Pwned:"
echo "  1. Go to: https://haveibeenpwned.com/Passwords"
echo "  2. Download the SHA-1 ordered by prevalence list (Pwned Passwords v8+)"
echo "  3. Many reversed cleartext versions are available on academic torrent sites"
echo "     under 'linkedin_passwords_cleartext.txt'"
echo "  4. Save it to: $RAW_DIR/linkedin.txt"
echo ""
echo "If you skip LinkedIn, preprocess.py will use RockYou only (still ~14M passwords)."
echo ""
echo "=== Done ==="
