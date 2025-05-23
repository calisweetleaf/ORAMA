#!/bin/bash
# This script is intended for automated execution by systems like Jules
# to set up the necessary environment for the ORAMA project.

set -eux # Exit on error, print commands

echo "--- Starting ORAMA Project Dependency Setup for Jules ---"

# 0. Identify Python command
PYTHON_CMD="python"
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif ! command -v python &>/dev/null; then
    echo "Error: Neither python3 nor python command found. Jules environment should provide Python."
    exit 1
fi
echo "Using Python command: $PYTHON_CMD"

# 1. Ensure pip is available and upgrade it
if ! $PYTHON_CMD -m pip --version &>/dev/null; then
    echo "Error: pip for $PYTHON_CMD is not available. Attempting to ensure pip..."
    $PYTHON_CMD -m ensurepip --upgrade || {
        echo "Failed to ensure pip. Jules environment should provide pip."
        exit 1
    }
fi

echo "Upgrading pip..."
$PYTHON_CMD -m pip install --upgrade pip

# 2. Install Python packages from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing Python dependencies from requirements.txt..."
    $PYTHON_CMD -m pip install -r requirements.txt
else
    echo "Error: requirements.txt not found in the current directory."
    echo "Please ensure requirements.txt is present in the root of the ORAMA project."
    exit 1
fi

# 3. Install NLTK data required by ORAMA
echo "Downloading NLTK 'punkt' tokenizer data..."
$PYTHON_CMD -c "import nltk; nltk.download('punkt', quiet=True)"

# 4. Install Playwright browser binaries
echo "Installing Playwright browser binaries (with system dependencies if on Linux)..."
# The --with-deps flag helps install necessary OS libraries on Linux for the browsers.
# It might be ignored or not applicable on other OSes Jules might use.
$PYTHON_CMD -m playwright install --with-deps

# 5. System-level dependencies (e.g., Tesseract OCR)
# Jules's environment might handle these differently or have them pre-installed.
# This section provides common commands for Debian/Ubuntu-based systems.
# If Jules runs on a different base OS, these might need adjustment or may not be needed.

echo "Checking for and attempting to install system-level dependencies (e.g., Tesseract OCR)..."
# Check if apt-get is available (common in Debian/Ubuntu based VMs)
if command -v apt-get &>/dev/null; then
    echo "apt-get found. Attempting to install Tesseract OCR..."
    # Run apt-get update only if needed, and non-interactively
    # Check if sudo is available and necessary
    SUDO_CMD=""
    if command -v sudo &>/dev/null && [ "$(id -u)" != "0" ]; then
        SUDO_CMD="sudo"
    fi

    echo "Updating package list (apt-get update)..."
    $SUDO_CMD apt-get update -y || echo "apt-get update failed, continuing..."

    echo "Installing Tesseract OCR (tesseract-ocr)..."
    $SUDO_CMD apt-get install -y tesseract-ocr || echo "Failed to install tesseract-ocr via apt-get. It might already be present or require manual installation/configuration in Jules's VM."
else
    echo "apt-get not found. Skipping automatic installation of Tesseract OCR."
    echo "Please ensure Tesseract OCR and any other required system libraries (like those for PyAutoGUI if GUI operations are needed) are available in Jules's execution environment."
fi

echo "--- ORAMA Project Dependency Setup for Jules Complete ---"