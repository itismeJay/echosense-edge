# EchoSense School Setup Guide

## Your Phone Hotspot Settings
  Hotspot Name:     <SCHOOL_HOTSPOT_NAME>
  Hotspot Password: <SCHOOL_HOTSPOT_PASSWORD>
  (Keep these settings saved on your phone)

## Every Time At School

### Step 1 — Turn on hotspot
  Android: Settings → Hotspot → Turn ON
  Make sure the configured school hotspot is selected

### Step 2 — Plug in Pi
  Connect Pi power cable to outlet

### Step 3 — Wait 60 seconds
  Watch the green LED on Pi:
  - Fast blinks = loading EchoSense
  - Slow heartbeat pulse = connected and listening ✅

### Step 4 — SSH into Pi (if needed)
  Connect your laptop to the school hotspot
  Then run ONE of these:

  Option A (easier):
    ssh echosense@raspberrypi.local

  Option B (if A does not work):
    Check phone hotspot for connected devices
    Find Raspberry Pi IP address
    ssh echosense@[that IP address]

### Step 5 — Check logs
  tail -f /home/echosense/echosense-edge/logs/echosense.log

  You should see:
    [WIFI] Connected!
    [WIFI] Network:  YOUR_WIFI_SSID
    [WIFI] IP:       192.168.x.x
    [MAIN] Listening...

### Step 6 — Open dashboard
  https://echosense-frontend.vercel.app
  Pi status should show: 🟢 Online

## If Pi Does Not Connect
  1. Make sure hotspot name matches: YOUR_WIFI_SSID
  2. Make sure the saved password matches: YOUR_WIFI_PASSWORD
  3. Turn hotspot OFF then ON again
  4. Wait 30 more seconds
  5. Check Pi LED — if OFF, Pi has no power

## Demo Commands
  Check service is running:
    sudo systemctl status echosense.service

  Watch live logs:
    tail -f /home/echosense/echosense-edge/logs/echosense.log

  Restart if needed:
    sudo systemctl restart echosense.service

  Check the durable outbox without sending an alert:
    echosense-env/bin/python3 -m sender.outbox_admin health

  Do not bypass the outbox with a direct curl request. Run a harmless
  controlled ingestion test only after backend deployment and explicit
  approval.

## Network Notes (this Pi)
  This Pi runs Raspberry Pi OS Bookworm and uses **NetworkManager**
  (not wpa_supplicant). WiFi networks are stored as NetworkManager
  connection profiles, not in /etc/wpa_supplicant/wpa_supplicant.conf.

  See/manage saved WiFi from the Pi:
    nmcli connection show
    nmcli device wifi list

  Configure the school hotspot with the intended autoconnect priority so the
  Pi prefers it whenever it is in range.

  Reload WiFi config without rebooting:
    sudo nmcli connection reload
    sudo nmcli device wifi rescan
