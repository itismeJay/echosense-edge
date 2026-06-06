# EchoSense School Setup Guide

## Your Phone Hotspot Settings
  Hotspot Name:     TECNOCAMON
  Hotspot Password: 123456789
  (Keep these settings saved on your phone)

## Every Time At School

### Step 1 — Turn on hotspot
  Android: Settings → Hotspot → Turn ON
  Make sure name is TECNOCAMON

### Step 2 — Plug in Pi
  Connect Pi power cable to outlet

### Step 3 — Wait 60 seconds
  Watch the green LED on Pi:
  - Fast blinks = loading EchoSense
  - Slow heartbeat pulse = connected and listening ✅

### Step 4 — SSH into Pi (if needed)
  Connect your laptop to TECNOCAMON hotspot
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
    [WIFI] Network:  TECNOCAMON
    [WIFI] IP:       192.168.x.x
    [MAIN] Listening...

### Step 6 — Open dashboard
  https://echosense-frontend.vercel.app
  Pi status should show: 🟢 Online

## If Pi Does Not Connect
  1. Make sure hotspot name is exactly: TECNOCAMON
  2. Make sure password is: 123456789
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

  Manual test alert:
    curl -s -X POST https://echosense-backend-75h3.onrender.com/alerts/ \
      -H "Content-Type: application/json" \
      -d '{"severity":"high","confidence":0.94,"duration":3.0,
      "location":"Grade 6 Classroom",
      "transcribed_text":"yawa ka bogo kaayo",
      "detected_words":["yawa","bogo kaayo"],
      "yamnet_class":"Screaming","yamnet_score":0.83,
      "emotion":"angry","rms":820.0,"energy_variance":6200.5,
      "zero_crossing_rate":0.14,"peak_to_average":4.2,
      "waveform_snapshot":[120,340,890,450,230],
      "categories":["academic_shaming"],
      "language":"ceb","hard_hits":["yawa"],"soft_hits":[]}'

## Network Notes (this Pi)
  This Pi runs Raspberry Pi OS Bookworm and uses **NetworkManager**
  (not wpa_supplicant). WiFi networks are stored as NetworkManager
  connection profiles, not in /etc/wpa_supplicant/wpa_supplicant.conf.

  See/manage saved WiFi from the Pi:
    nmcli connection show
    nmcli device wifi list

  TECNOCAMON is saved with autoconnect priority 10 (home WiFi is 0),
  so the Pi prefers the hotspot whenever it is in range.

  Reload WiFi config without rebooting:
    sudo nmcli connection reload
    sudo nmcli device wifi rescan
