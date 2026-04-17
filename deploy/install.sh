#!/usr/bin/env bash
# Bootstrap script for Phoenyx on Ubuntu 22.04 (GCP e2-small).
# Run as the non-root user that will own the bot (e.g. `phoenyx`).
# Usage:  bash deploy/install.sh

set -euo pipefail

PHOENYX_USER="${PHOENYX_USER:-$USER}"
PHOENYX_HOME="/home/${PHOENYX_USER}"
PHOENYX_DIR="${PHOENYX_HOME}/Phoenyx"
IBC_DIR="${PHOENYX_HOME}/ibc"
IB_GATEWAY_VERSION="10.19.2g"
IBC_VERSION="3.20.0"

echo "==> Installing system packages"
sudo apt-get update
sudo apt-get install -y \
    python3 python3-venv python3-pip \
    git curl unzip \
    xvfb \
    openjdk-17-jre-headless \
    libxtst6 libxrender1 libxi6 \
    socat

echo "==> Creating directories"
mkdir -p "${IBC_DIR}" "${PHOENYX_HOME}/downloads"
cd "${PHOENYX_HOME}/downloads"

echo "==> Downloading IB Gateway ${IB_GATEWAY_VERSION}"
if [ ! -f "ibgateway-stable-standalone-linux-x64.sh" ]; then
    curl -fSLo ibgateway-stable-standalone-linux-x64.sh \
        "https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh"
    chmod +x ibgateway-stable-standalone-linux-x64.sh
fi

echo "==> Installing IB Gateway (accept defaults; install to ~/Jts/ibgateway/<version>)"
echo "    If prompted, install to: ${PHOENYX_HOME}/Jts/ibgateway/${IB_GATEWAY_VERSION}"
./ibgateway-stable-standalone-linux-x64.sh || true

echo "==> Downloading IBC ${IBC_VERSION}"
if [ ! -f "IBCLinux-${IBC_VERSION}.zip" ]; then
    curl -fSLo "IBCLinux-${IBC_VERSION}.zip" \
        "https://github.com/IbcAlpha/IBC/releases/download/${IBC_VERSION}/IBCLinux-${IBC_VERSION}.zip"
fi
unzip -o "IBCLinux-${IBC_VERSION}.zip" -d "${IBC_DIR}"
chmod +x "${IBC_DIR}"/*.sh

echo "==> Setting up Phoenyx Python environment"
if [ ! -d "${PHOENYX_DIR}" ]; then
    echo "!! ${PHOENYX_DIR} not found. Clone your repo there before running this script:"
    echo "   git clone <your-repo-url> ${PHOENYX_DIR}"
    exit 1
fi
cd "${PHOENYX_DIR}"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate

echo "==> Copying IBC config template (edit before starting the service)"
if [ ! -f "${IBC_DIR}/config.ini" ]; then
    cp "${PHOENYX_DIR}/deploy/ibc/config.ini.template" "${IBC_DIR}/config.ini"
    chmod 600 "${IBC_DIR}/config.ini"
    echo "    -> Edit ${IBC_DIR}/config.ini with your IB username and password"
fi

echo "==> Installing systemd units"
sudo cp "${PHOENYX_DIR}/deploy/systemd/phoenyx-ibgateway.service" /etc/systemd/system/
sudo cp "${PHOENYX_DIR}/deploy/systemd/phoenyx-daily.service" /etc/systemd/system/
sudo cp "${PHOENYX_DIR}/deploy/systemd/phoenyx-daily.timer" /etc/systemd/system/
sudo sed -i "s|__USER__|${PHOENYX_USER}|g" /etc/systemd/system/phoenyx-ibgateway.service
sudo sed -i "s|__USER__|${PHOENYX_USER}|g" /etc/systemd/system/phoenyx-daily.service
sudo sed -i "s|__IBGW_VERSION__|${IB_GATEWAY_VERSION}|g" /etc/systemd/system/phoenyx-ibgateway.service
sudo systemctl daemon-reload

echo ""
echo "================================================================"
echo "Install complete. Next steps:"
echo "  1. Edit ${IBC_DIR}/config.ini (IB credentials, trading mode)"
echo "  2. Edit ${PHOENYX_DIR}/config/settings.yaml and .env"
echo "  3. Start Gateway:       sudo systemctl enable --now phoenyx-ibgateway"
echo "  4. Check Gateway login: journalctl -u phoenyx-ibgateway -f"
echo "  5. Arm 4h timer:        sudo systemctl enable --now phoenyx-daily.timer"
echo "  6. Inspect schedule:    systemctl list-timers phoenyx-daily"
echo "  7. Manual test run:     sudo systemctl start phoenyx-daily.service"
echo "  8. Check run logs:      journalctl -u phoenyx-daily -f"
echo "================================================================"
