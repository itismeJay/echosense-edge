# Raspberry Pi 5 Wi-Fi Provisioning

## Safety status

The repository contains the local provisioning service, portal, unit tests, and
an **uninstalled** systemd unit. No command in this guide has been run against a
Raspberry Pi by this development change.

Do not install, enable, or start the service until the read-only preflight has
been run on the target Pi, its output has been reviewed, and the rollback steps
below have been approved. Do not test on a production classroom network first.

## Architecture

The Wi-Fi setup service is separate from `echosense.service` and never imports
or modifies detection, model, device-identity, classroom-assignment, or outbox
state. NetworkManager remains the only owner of authorized client profiles.

On startup and after a connection loss:

1. An existing client connection is kept; the service never switches a usable
   connection merely because another profile has higher priority.
2. Otherwise, the service scans in client mode and makes two bounded attempts
   to activate saved, autoconnect-enabled profiles in NetworkManager priority
   order. Visible profiles that are absent from a successful scan are skipped;
   saved hidden profiles remain eligible because they cannot appear in a scan.
3. If none connects, the service caches the client-mode scan results.
4. It then starts the device-specific WPA setup network
   `EchoSense-Setup-XXXX` at `10.42.0.1`.
5. The authenticated portal is served at `http://10.42.0.1` while the setup
   hotspot is active. While client Wi-Fi is connected, the same portal binds
   only to the Pi's current LAN address and accepts clients only from that
   interface's IPv4 subnet.
6. A submitted connection requires an explicit confirmation and gets one
   bounded attempt. The setup hotspot must stop
   during the attempt because the Pi has one built-in Wi-Fi interface.
7. Success enables NetworkManager autoconnect for the newly authorized profile.
   Failure removes the candidate, tries a saved authorized connection once,
   and restores the setup hotspot if no saved connection works. It never loops
   on a bad password.

On the client LAN, `Refresh network list` scans without changing modes. On the
setup hotspot it is an explicit single-radio transition: the phone temporarily
disconnects, the Pi scans, and the same setup hotspot returns. Hidden SSIDs can
always be entered manually.

## Dashboard integration contract

The authenticated device heartbeat preserves its existing device credentials,
`ip`, `ssid`, `mac`, and `hostname` fields and adds:

- `wifi_status`: `connected` or `disconnected`;
- `wifi_change_method`: `local_portal`;
- `wifi_local_url`: the current LAN portal URL, or `http://10.42.0.1/` when
  disconnected;
- `wifi_remote_scan_supported`: `true` when NetworkManager is available (Pi),
  `false` on the Mac.

The dashboard command `{"action": "scan"}` makes the box run `nmcli device wifi
rescan` and list nearby networks without leaving its current network. The
result is `{"status": "applied", "networks": [{"ssid", "signal" (0-100),
"security" ("open" or e.g. "WPA2"), "known"}], "current_ssid"}`: at most 50,
one entry per SSID, strongest first, hidden SSIDs omitted. The Mac reports
`{"status": "failed", "error": "not_supported_on_this_device"}`.

The dashboard should show the last reported SSID and online state. When the
browser can reach `wifi_local_url`, **Change Wi-Fi** opens the authenticated
local portal, where live scan results, hidden-network entry, confirmed switching,
password-free switching to saved profiles, and confirmed forgetting are
available. If the device is offline, the dashboard
must not show cloud scan results or imply that a remote switch is possible. It
should tell the administrator to join the device-specific setup hotspot from
the setup card and open `http://10.42.0.1/`.

This repository contains the edge heartbeat and local portal only. Persisting
these fields and rendering the Wi-Fi Settings controls must be implemented and
tested in the separate backend/dashboard repository.

Client Wi-Fi passwords are supplied to NetworkManager through its documented
owner-only `passwd-file` activation input. The volatile file lives under
`/run/echosense-wifi`, is never included in process arguments or logs, and is
unlinked immediately. NetworkManager persists successful profiles in its own
root-controlled system connection store. EchoSense does not put client Wi-Fi
passwords in `edge.env`, application JSON, SQLite, or logs.

The setup hotspot PSK must be available after reboot, so it is held in
`/etc/echosense/wifi-setup.json` with mode `0600`. The portal administrator
password is stored only as a salted scrypt digest. Both one-time plaintext
values are shown only by the explicit bootstrap command and should be written
on the physical device setup card.

## Read-only hardware preflight (approval gate)

Run only this command first on the Pi:

```bash
cd /home/echosense/echosense-edge
echosense-env/bin/python3 -m provisioning.wifi_setup preflight --interface wlan0
```

It only reads `/proc/device-tree/model`, `/etc/os-release`, interface
capabilities, and NetworkManager status. It must report all of the following:

- a Raspberry Pi model (the deployment target is Raspberry Pi 5);
- 64-bit ARM architecture;
- Raspberry Pi OS based on Debian Bookworm or Trixie;
- NetworkManager running;
- `wlan0` present as Wi-Fi;
- access-point mode supported;
- `supported: true` and
  `radio_strategy: single_interface_safe_transitions`.

Also capture these read-only operator checks for the validation record:

```bash
uname -m
cat /etc/os-release
systemctl is-active NetworkManager
nmcli --version
nmcli -f GENERAL,WIFI-PROPERTIES device show wlan0
nmcli -f NAME,UUID,TYPE,AUTOCONNECT connection show
sudo systemctl cat echosense.service
```

Stop if the OS is 32-bit, NetworkManager is not the active manager, `wlan0` is
renamed, AP support is absent, or another hotspot/network service manages the
interface. Resolve and re-review the design rather than installing it.

## Rollback plan to approve before activation

Before changing networking, keep a keyboard/monitor or wired Ethernet console
available. Record the existing active profile and take a root-only backup:

```bash
sudo install -d -m 700 /root/echosense-network-backup
sudo cp -a /etc/NetworkManager/system-connections \
  /root/echosense-network-backup/
nmcli -f NAME,UUID,TYPE,AUTOCONNECT connection show | \
  sudo tee /root/echosense-network-backup/connections.txt >/dev/null
```

Rollback does not touch the EchoSense SQLite outbox:

```bash
sudo systemctl disable --now echosense-wifi-setup.service
sudo rm /etc/systemd/system/echosense-wifi-setup.service
sudo systemctl daemon-reload
sudo nmcli connection delete id "EchoSense Setup Hotspot" || true
sudo nmcli connection reload
sudo nmcli device up wlan0
```

If a newly added `EchoSense WiFi ...` profile must be removed, identify its
exact UUID with `nmcli connection show`, delete only that UUID, and bring up the
previously recorded profile. Do not delete or reset an outbox database.

## Installation after explicit hardware approval

These steps are documented for reproducibility but are intentionally not
performed by this change:

```bash
sudo install -d -m 700 /etc/echosense
sudo /home/echosense/echosense-edge/echosense-env/bin/python3 \
  -m provisioning.wifi_setup bootstrap \
  --interface wlan0 --output /etc/echosense/wifi-setup.json
```

Record the hotspot and administrator passwords on the physical setup card. Do
not copy the terminal output to tickets, chat, screenshots, or source control.

Then install, but do not yet enable, the reviewed unit:

```bash
sudo install -m 644 deploy/echosense-wifi-setup.service \
  /etc/systemd/system/echosense-wifi-setup.service
sudo systemctl daemon-reload
sudo systemctl enable echosense-wifi-setup.service
```

For the first controlled test, use a non-production hotspot, keep console
access open, and start the unit manually:

```bash
sudo systemctl start echosense-wifi-setup.service
journalctl -u echosense-wifi-setup.service -f
```

Logs contain state names and hashes of requested SSIDs, never SSIDs, portal
passwords, Wi-Fi passwords, backend device keys, or HTTP form data.

## Capstone checking: exact phone workflow

1. Before leaving, keep the physical setup card with the Pi. It contains that
   Pi's `EchoSense-Setup-XXXX` name, hotspot password, and separate administrator
   password.
2. Power the Pi. If a previously authorized Wi-Fi is reachable, the service
   reconnects automatically in configured priority order. The authenticated
   portal is reachable at the LAN IP reported by the dashboard.
3. To demonstrate recovery using only the phone, turn off or move out of range
   of the currently connected phone hotspot. Wait about 45–60 seconds.
4. On the phone, join `EchoSense-Setup-XXXX` using the setup hotspot password.
   Disable mobile-data auto-switching for this step if the phone abandons
   Wi-Fi networks without Internet.
5. Open `http://10.42.0.1` and enter the administrator password from the setup
   card.
6. Choose a scanned network, or type the exact SSID and select **Hidden
   network** if necessary. Enter its Wi-Fi password and tap **Connect**.
7. Confirm the switch. The setup hotspot disappears during the bounded attempt.
   Join the selected
   Wi-Fi on the phone. The EchoSense dashboard should return the same registered
   device/classroom to Online after its next authenticated heartbeat.
8. If connection fails, wait up to about 45 seconds, rejoin the same
   `EchoSense-Setup-XXXX`, reopen `http://10.42.0.1`, read the failure status,
   and retry or choose another network.
9. Use **Forget** only for an inactive saved profile that should no longer be
   authorized, and confirm the removal. The portal refuses to delete the active
   profile.

If the phone itself must provide the destination hotspot, one phone generally
cannot remain joined to the Pi setup AP while simultaneously hosting its own
hotspot. Enter the hotspot SSID manually, select **This phone will provide the
destination hotspot**, tap **Connect**, and immediately enable the phone
hotspot. The Pi waits 10 seconds before its bounded attempt. A second phone is
still the safer fallback. This timing case requires hardware validation before
relying on it for a live demonstration.

## Supported and unsupported scenarios

Designed and locally tested:

- WPA/WPA2 Personal networks using an 8–63 character PSK or 64-digit hex PSK;
- open networks when explicitly selected by the administrator;
- visible and hidden SSIDs;
- automatic reconnection to saved, authorized NetworkManager profiles;
- NetworkManager priority ordering without replacing a live usable connection;
- failed-password cleanup, saved-profile restoration, and setup-hotspot recovery;
- authenticated local-LAN Wi-Fi changes with explicit confirmation;
- password-free switching to an inactive saved profile;
- offline detection/outbox operation while networking is unavailable;
- changing networks after the current network becomes unavailable.

Not currently supported:

- WPA-Enterprise/802.1X credentials, captive-portal acceptance, or per-user
  certificates;
- guaranteed WPA3-only operation (must be checked with the target OS/firmware);
- simultaneous client + hotspot operation on built-in `wlan0`;
- remote Internet exposure of the setup portal.

Open networks are accepted only after an explicit checkbox and provide no
link-layer privacy. Prefer WPA2/WPA3 Personal for school and demonstration use.

## Hardware validation checklist

Before production use, verify on the actual Pi 5:

1. preflight output and rollback through local console;
2. WPA2 hotspot creation, DHCP, and `http://10.42.0.1` on Android and iOS;
3. cached scan accuracy before AP mode and refresh recovery afterward;
4. visible, hidden, wrong-password, unavailable-SSID, and open-network cases;
5. power loss during scan, connect, and hotspot restoration;
6. NetworkManager profile secret persistence and autoconnect after reboot;
7. authenticated portal subnet restriction while a client connection is active;
8. loss and restoration of Wi-Fi while `echosense.service` keeps detecting;
9. pending outbox retention offline and authenticated delivery after reconnect;
10. unchanged QR device identity, classroom assignment, and heartbeat headers;
11. phone-hotspot behavior, especially same-phone destination timing;
12. journal review confirming no submitted SSID or password appears.

Official references used for the implementation are the
[Raspberry Pi NetworkManager guidance](https://www.raspberrypi.com/documentation/configuration/computers/raspberry-pi.html)
and the
[NetworkManager `nmcli` reference](https://networkmanager.dev/docs/api/latest/nmcli).
