# 🏌️ Golf Marshal ⛳
By artdev99 and XXX

## Update the Raspberry
```
sudo apt update && sudo apt upgrade
```

## Setup [WiFi](https://howtoraspberrypi.com/connect-wifi-raspberry-pi-3-others/)
Open a terminal on the raspberry. Update drivers if needed: <br> 

```
sudo apt update
sudo apt dist-upgrade
```
Setup the WiFi using the command line:
```
sudo nano /etc/wpa_supplicant/wpa_supplicant.conf
```

Add to the bottom of the file:
```
network={
    ssid="Your_SSID"
    psk="Your_Password"
    key_mgmt=WPA-PSK
}
```
Close the file:
```
CTRL+ O, then CTRL + X
```
Start the WiFi on the raspberry:
```
ifconfig wlan0
```

<br>

To find the raspberry IP address:

```
ifconfig
```

## Setup Camera v2 8MB

[Camera hardware documentation](https://www.raspberrypi.com/documentation/accessories/camera.html)<br> 
[Camera software documentation](https://www.raspberrypi.com/documentation/computers/camera_software.html)

No optical zoom, but it is possible to manually adjust the sharpness ("focus") by turning the lens.

[Youtube (short): Connecting camera to the raspberry](https://www.youtube.com/watch?v=GImeVqHQzsE) <br>
[Youtube: Connecting the camera to the raspberry + config](https://www.youtube.com/watch?v=bpzGN35oaJ4)

To become root:
```
sudo su
```

Enter the config:
```
sudo raspi-config
```
1. Select Interface Options (or Interfacing Options)
2. Enable camera interace
3. Finish
4. Reboot
```
sudo reboot
```

<br>

`libcamera` is replaced by `rpicam`. Test the camera (run from Desktop folder):
```
rpicam-hello
rpicam-jpeg --output test.jpg
rpicam-vid  -t 10s -o test.mp4
```

<br>

Install/Update picamera2 for python:
```
sudo apt install -y python3-picamera2
```

[picamera2 documentation](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)

Test Scripts
- Low level
```
from picamera2 import Picamera2, Preview
import time
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration()
picam2.configure(camera_config)
picam2.start_preview(Preview.QTGL)
picam2.start()
time.sleep(2)
picam2.capture_file("test.jpg")
```
- High-level
```
from picamera2 import Picamera2
picam2 = Picamera2()
picam2.start_and_record_video("test.mp4", duration=5)
```

## Setup Video Stream
```
sudo apt install python3-opencv
```

Create venv
```
python3 -m venv camera_stream_env
source camera_stream_env/bin/activate
pip install flask
python3 camera_stream.py
```

http://<raspberry_pi_ip>:5000/video_feed

### Remote access using ngrok / localtunnel / pinggy.io
[ngrok documentation](https://ngrok.com/docs/guides/device-gateway/raspberry-pi/)

Setup ngrok
```
 curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list && sudo apt update && sudo apt install ngrok

```

To use
```
ngrok http 5000
```

Change remote url to: <br>
http://<your_ngrok_url>/upload"

[localtunnel documentation](https://theboroer.github.io/localtunnel-www/)

Install npm node.js
```
sudo apt install nodejs npm -y
```

Install localtunnel
```
sudo npm install -g localtunnel
```

To use
```
lt --port 5000
```

[pinggy.io](https://pinggy.io/)

To use
```
ssh -p 443 -R0:localhost:5000 a.pinggy.io
```



## Raspberry Pi Specifications

SSH to the Pi 3b+
```
ssh -p 22 XXX@XXX.XXX.com
```

SSH to the Pi Zero
```
ssh -p 23 XXX@XXX.XXX.com
```

| Feature                  | Raspberry Pi 3b+                                                  | Raspberry Pi Zero                        |
|--------------------------|-------------------------------------------------------------------|------------------------------------------|
| **Model**                | Raspberry Pi 3 Model B Plus Rev 1.3                                | Raspberry Pi Zero 2 W Rev 1.0            |
| **OS Version**           | Debian GNU/Linux 12 (bookworm)                                     | Debian GNU/Linux 12 (bookworm)           |
| **Kernel Version**       | Linux golfmarshal1 6.6.31+rpt-rpi-v8 #1 SMP PREEMPT Debian 1:6.6.31-1+rpt1 (2024-05-29) aarch64 GNU/Linux | Linux golfmarshal2 6.6.31+rpt-rpi-v8 #1 SMP PREEMPT Debian 1:6.6.31-1+rpt1 (2024-05-29) aarch64 GNU/Linux |
| **CPU**                  | Quad-core ARM Cortex-A53 (Armv8-A)                                           | Quad-core ARM Cortex-A53 (Armv8-A)                |
| **CPU Cores**            | 4                                                                 | 4                                        |
| **CPU Speed**            | 1.4 GHz                                                           | 1 GHz                                    |
| **RAM**                  | 1 GB LPDDR2                                                       | 512MB LPDDR2                             |
| **Local IP**             | 192.168.1.1                                                       | 192.168.1.41                             |
| **Power Supply**         | 5V DC                                                             | 5V DC                                    |



https://www.pidramble.com/wiki/benchmarks/power-consumption <br>
3 B+ 	HDMI off, LEDs off, onboard WiFi 	400 mA (2.0 W) <br>
Zero 2 W 	HDMI off, LEDs off, onboard WiFi 	120 mA (0.7 W)

## Ultralytics (YOLO)
[documentation](https://docs.ultralytics.com/guides/raspberry-pi/)

Install with pip
```
sudo apt update
sudo apt install python3-pip -y
pip install -U pip
pip install ultralytics[export]
sudo reboot
```

### Docker
```
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

Install with Docker
```
t=ultralytics/ultralytics:latest-arm64 && sudo docker pull $t && sudo docker run -it --ipc=host $t
```

## Blue tint image
https://forums.raspberrypi.com/viewtopic.php?t=350241

https://github.com/Gordon999/RPiCamGUI

https://github.com/raspberrypi/libcamera/issues/87

## picamera2 github python3.9
https://github.com/raspberrypi/picamera2

https://github.com/raspberrypi/libcamera

https://github.com/raspberrypi/pylibcamera

