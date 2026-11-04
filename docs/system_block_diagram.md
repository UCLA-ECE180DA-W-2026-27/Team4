# System Block Diagram — Assistive Shopping Assistant

This document visualizes the complete end-to-end hardware and software architecture for **ECE 180DA Team 4**.

---

## 1. Complete System Block Diagram

```mermaid
flowchart TD
    %% Styling
    classDef hardware fill:#1f2335,stroke:#7aa2f7,stroke-width:2px,color:#c0caf5;
    classDef streamer fill:#24283b,stroke:#bb9af7,stroke-width:2px,color:#c0caf5;
    classDef compute fill:#1a1b26,stroke:#7dcfff,stroke-width:2px,color:#c0caf5;
    classDef cloud fill:#292e42,stroke:#ff9e64,stroke-width:2px,color:#c0caf5;
    classDef userio fill:#1f2335,stroke:#9ece6a,stroke-width:2px,color:#c0caf5;

    %% Subgraph 1: User / Head Wearable Rig
    subgraph HEAD_RIG ["1. User Wearable (Head / Hat Rig)"]
        CAM["Hat/Visor Camera<br/>(Video Capture)"]:::hardware
        IMU["BerryIMU v3<br/>(Accel / Gyro / Mag)"]:::hardware
        EARBUD["Bluetooth Earbud<br/>(Mic & Speaker)"]:::userio
    end

    %% Subgraph 2: Sensor Hub (Backpack / Belt)
    subgraph RPI_HUB ["2. Sensor Hub (Raspberry Pi 4)"]
        RPI_CAP["Camera Ingestion<br/>(Picamera2 / OpenCV)"]:::streamer
        RPI_IMU["BerryIMU Reader<br/>(I2C @ 50 Hz)"]:::streamer
        UDP_STREAM["Low-Latency Streamer<br/>(UDP / ZeroMQ)"]:::streamer
        
        CAM -->|CSI / USB| RPI_CAP
        IMU -->|I2C GPIO| RPI_IMU
        RPI_CAP --> UDP_STREAM
        RPI_IMU --> UDP_STREAM
    end

    %% Subgraph 3: Local Network Gateway
    subgraph NETWORK ["3. Local Wireless LAN & Internet"]
        HOTSPOT["Smartphone 5G Hotspot<br/>(Private Wi-Fi Subnet)"]:::cloud
    end

    UDP_STREAM -->|Wi-Fi UDP Stream| HOTSPOT

    %% Subgraph 4: Central Processing Engine (Backpack Laptop)
    subgraph LAPTOP_COMPUTE ["4. Central Compute Engine (Backpack Laptop)"]
        
        subgraph STREAM_RECV ["A. Streaming & Ingestion"]
            RECV["Stream Receiver<br/>(Decompresses Frames & IMU Packets)"]:::compute
        end

        subgraph VOICE_IO ["B. Voice & Audio Dialogue"]
            STT["Local STT Engine<br/>(Faster-Whisper / VAD)"]:::compute
            TTS["Local TTS Engine<br/>(Piper / Edge-TTS)"]:::compute
            DIALOGUE["Dialogue State Coordinator"]:::compute
        end

        subgraph NAV_SUBSYSTEM ["C. Indoor Store Navigation"]
            MAP_DB["Topological Waypoint Graph<br/>(CVS Aisle JSON Database)"]:::compute
            DIJKSTRA["Pathfinding Engine<br/>(Dijkstra Route Planner)"]:::compute
            PDR["Pedestrian Dead Reckoning<br/>(Step Counter + Gyro Heading)"]:::compute
            OCR_RESET["Landmark Drift Corrector<br/>(Overhead Aisle Sign OCR)"]:::compute
        end

        subgraph VISION_SUBSYSTEM ["D. Shelf Vision & Product Q&A"]
            SNAP["High-Res Frame Capture"]:::compute
            VLM_CLIENT["Gemini Client / Parser<br/>(Structured JSON Catalog)"]:::compute
            QA_ENGINE["Shelf Q&A Manager<br/>(Ingredients, Skin Types, Prices)"]:::compute
            TARGET_LOCK["Target Object Selector<br/>(Pixel Bounding Box)"]:::compute
        end

        subgraph SERVO_SUBSYSTEM ["E. Hand Grasp Guidance (Visual Servoing)"]
            HAND_TRACK["MediaPipe Hands<br/>(Fingertip & Wrist 3D Pose)"]:::compute
            SERVO_DELTA["Delta Calculator<br/>(Hand vs Product dx, dy)"]:::compute
            SERVO_CUES["Real-Time Audio Cues<br/>('Higher... Right... Grab!')"]:::compute
        end
    end

    %% Subgraph 5: Cloud AI
    subgraph CLOUD_SERVICES ["5. Cloud Multimodal AI"]
        GEMINI_API["Google Gemini 2.5 / 1.5 Flash<br/>(Multimodal VLM API)"]:::cloud
    end

    %% Inter-subsystem Connections
    HOTSPOT -->|Low-Latency Socket| RECV
    RECV -->|Video Frames| SNAP
    RECV -->|Video Frames| OCR_RESET
    RECV -->|Video Frames| HAND_TRACK
    RECV -->|IMU Heading & Steps| PDR

    %% Audio connections
    EARBUD <-->|Bluetooth Audio| LAPTOP_COMPUTE
    EARBUD -->|User Spoken Speech| STT
    STT -->|Transcribed Text| DIALOGUE
    TTS -->|Synthesized Spoken Guidance| EARBUD

    %% Navigation flow
    DIALOGUE -->|"Where is moisturizer?"| DIJKSTRA
    MAP_DB --> DIJKSTRA
    DIJKSTRA -->|"Turn-by-turn Plan"| DIALOGUE
    PDR --> OCR_RESET
    OCR_RESET -->|"Snap Position"| DIJKSTRA
    PDR -->|"Step Milestones"| TTS

    %% Shelf Q&A flow
    DIALOGUE -->|"Arrived at shelf: 'What is here?'"| SNAP
    SNAP -->|Image Snapshot| VLM_CLIENT
    VLM_CLIENT <-->|HTTPS REST / GenAI SDK| HOTSPOT
    HOTSPOT <--> GEMINI_API
    VLM_CLIENT --> QA_ENGINE
    QA_ENGINE -->|"Spoken Overview & Answers"| TTS
    DIALOGUE -->|"User: 'I want CeraVe'"| TARGET_LOCK
    QA_ENGINE --> TARGET_LOCK

    %% Visual servoing flow
    TARGET_LOCK -->|"Locked Box (x1, y1, x2, y2)"| SERVO_DELTA
    HAND_TRACK -->|"Hand Position (xh, yh)"| SERVO_DELTA
    SERVO_DELTA --> SERVO_CUES
    SERVO_CUES -->|"Direct Audio Alerts"| TTS
```

---

## 2. Subsystem Interface Summary

| Subsystem | Input Source | Processing Engine | Output / Destination |
|---|---|---|---|
| **Sensor Hub (RPi 4)** | Camera (CSI/USB), BerryIMU (I2C) | Lightweight OpenCV + SMBus | UDP/ZeroMQ packet stream to Laptop over Wi-Fi |
| **Voice Interface** | Bluetooth Earbud Microphone | Faster-Whisper (STT) + Piper (TTS) | Text commands to Dialogue Coordinator; Spoken audio to earbud |
| **Indoor Navigation** | Pre-mapped CVS JSON, IMU steps & gyro, camera frames | Dijkstra Shortest Path + OCR Landmark Resets | Turn-by-turn walking instructions (*"Walk 8 steps... turn left into Aisle 4"*) |
| **Shelf Multimodal Q&A** | 1280x800 camera snapshot, user voice queries | Google Gemini 2.5/1.5 Flash API | Spoken product summary, interactive answers, product bounding boxes `[ymin, xmin, ymax, xmax]` |
| **Hand Guidance (Servoing)**| Live camera feed + locked target bounding box | MediaPipe Hands (`hand_landmarker.task`) + offset delta | Rapid verbal guidance cues (*"Move right... down... reach forward... grab it!"*) |
