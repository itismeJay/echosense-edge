#!/bin/sh
# Install (but do not enable) the EchoSense Wi-Fi setup service on the Pi.
# Usage: sudo deploy/install-wifi-setup.sh [EDGE_DIR]
# EDGE_DIR defaults to $ECHOSENSE_EDGE_DIR, then /home/echosense/echosense-edge.
set -eu
EDGE_DIR="${1:-${ECHOSENSE_EDGE_DIR:-/home/echosense/echosense-edge}}"
case "$EDGE_DIR" in /*) ;; *) echo "EDGE_DIR must be absolute" >&2; exit 2 ;; esac
[ -x "$EDGE_DIR/echosense-env/bin/python3" ] || { echo "missing $EDGE_DIR/echosense-env" >&2; exit 2; }
SRC="$(cd "$(dirname "$0")" && pwd)"
sed "s|/home/echosense/echosense-edge|$EDGE_DIR|g" "$SRC/echosense-wifi-setup.service" \
  > /etc/systemd/system/echosense-wifi-setup.service
install -d -m 0755 /etc/NetworkManager/dnsmasq-shared.d
install -m 0644 "$SRC/dnsmasq-shared.d/echosense-captive.conf" \
  /etc/NetworkManager/dnsmasq-shared.d/echosense-captive.conf
systemctl daemon-reload
echo "Installed for $EDGE_DIR. Enable with: sudo systemctl enable --now echosense-wifi-setup"
