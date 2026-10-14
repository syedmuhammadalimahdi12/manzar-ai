#!/usr/bin/env python3
"""
MANZAR — Full Demo Test Script
================================
Tests the entire pipeline end-to-end:
  1. Creates a session
  2. Generates synthetic test images (overlapping scenes)
  3. Uploads them from 4 angles
  4. Triggers reconstruction
  5. Monitors progress

Usage:
  source venv/bin/activate
  python test_demo.py

Server must be running:
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import io
import json
import time
import sys
import random

import requests
from PIL import Image, ImageDraw

API = "http://localhost:8000"


def generate_test_images(count=4, size=(800, 600)):
    """Generate synthetic overlapping images simulating an accident scene."""
    images = []
    angles = ["front", "right", "rear", "left"]

    for i in range(count):
        img = Image.new("RGB", size, "#555555")
        draw = ImageDraw.Draw(img)

        # Road markings
        y_off = i * 40
        for ly in range(100 + y_off, 500 + y_off, 60):
            if ly < size[1]:
                draw.rectangle([380, ly, 420, ly + 30], fill="#ffffff")

        # Offset per camera angle for parallax
        ox = [0, 150, 300, -150][i]
        oy = [0, 50, 100, -50][i]

        # Car 1 (blue sedan)
        draw.rectangle([250 + ox, 250 + oy, 400 + ox, 350 + oy],
                       fill="#2255cc", outline="#1a3399", width=2)
        # Car 2 (red motorcycle)
        draw.rectangle([350 + ox, 280 + oy, 500 + ox, 380 + oy],
                       fill="#cc2222", outline="#991a1a", width=2)

        # Debris
        random.seed(42 + i)
        for _ in range(20):
            dx = random.randint(300, 500) + ox // 2
            dy = random.randint(300, 400) + oy // 2
            if 0 < dx < size[0] and 0 < dy < size[1]:
                r = random.randint(2, 6)
                draw.ellipse([dx - r, dy - r, dx + r, dy + r], fill="#888888")

        # Skid marks
        draw.line([(300 + ox, 400 + oy), (380 + ox, 300 + oy)],
                  fill="#333333", width=4)

        # Texture noise for COLMAP features
        random.seed(100 + i)
        pixels = img.load()
        for _ in range(5000):
            px, py = random.randint(0, size[0]-1), random.randint(0, size[1]-1)
            r, g, b = pixels[px, py]
            n = random.randint(-15, 15)
            pixels[px, py] = (max(0, min(255, r+n)), max(0, min(255, g+n)), max(0, min(255, b+n)))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        buf.name = f"cam{i+1}_{angles[i]}.jpg"
        images.append((buf, angles[i]))

    return images


def main():
    print("=" * 60)
    print("MANZAR — Full Pipeline Demo Test")
    print("=" * 60)

    # 1. Check server
    print("\n[1/6] Checking server...")
    try:
        r = requests.get(f"{API}/")
        print(f"  Server: {r.json()['status']}")
    except Exception as e:
        print(f"  Server not running: {e}")
        print("  Start it with: uvicorn app.main:app --reload --port 8000")
        sys.exit(1)

    # 2. Create session
    print("\n[2/6] Creating session...")
    r = requests.post(f"{API}/session/create")
    session = r.json()
    sid = session["session_id"]
    print(f"  Session created: {sid}")

    # 3. Generate test images
    print("\n[3/6] Generating 4 test images (4 angles)...")
    images = generate_test_images(4)
    print(f"  Generated {len(images)} images")

    # 4. Upload photos
    print("\n[4/6] Uploading photos...")
    for i, (buf, angle) in enumerate(images):
        buf.seek(0)
        r = requests.post(
            f"{API}/session/{sid}/upload",
            files={"file": (buf.name, buf, "image/jpeg")},
            data={"angle_hint": angle, "device_id": f"test-cam-{i+1}"},
        )
        result = r.json()
        coverage = result.get("coverage", {})
        print(f"  Photo {i+1} ({angle}): {result['status']} | "
              f"coverage: {coverage.get('sufficient', '?')} | "
              f"missing: {coverage.get('missing_angles', [])}")

    # 5. Session state
    print("\n[5/6] Session state before reconstruction:")
    r = requests.get(f"{API}/session/{sid}")
    state = r.json()
    print(f"  Photos: {len(state['photos'])}")
    print(f"  Status: {state['status']}")
    print(f"  Coverage: {state['coverage_status']}")
    print(f"  Missing angles: {state['missing_angles']}")

    # 6. Trigger reconstruction
    print("\n[6/6] Triggering 3D reconstruction...")
    print("  (This takes 2-10 minutes depending on image complexity)")
    r = requests.post(f"{API}/session/{sid}/reconstruct")
    if r.status_code == 200:
        print(f"  Reconstruction started: {r.json()['status']}")
    else:
        print(f"  {r.status_code}: {r.text}")

    # Monitor
    print("\nMonitoring progress...")
    start = time.time()
    while True:
        time.sleep(5)
        r = requests.get(f"{API}/session/{sid}/reconstruction/status")
        recon = r.json()
        status = recon.get("status", "unknown")
        recon_data = recon.get("reconstruction", {})
        stage = recon_data.get("stage", "unknown") if recon_data else "unknown"
        pct = recon_data.get("percent", 0) if recon_data else 0
        elapsed = round(time.time() - start, 0)

        if status == "reconstruction_done":
            print(f"\n  DONE in {recon_data.get('elapsed_seconds', '?')}s!")
            print(f"  Point cloud: {recon_data.get('pointcloud_path', 'N/A')}")
            print(f"  Mesh: {recon_data.get('mesh_path', 'N/A')}")
            print(f"\n  Open dashboard: {API}/MANZAR.html#dashboard")
            print(f"  Session ID: {sid}")
            break
        elif status == "reconstruction_failed":
            error = recon_data.get("error", "unknown") if recon_data else "unknown"
            print(f"\n  FAILED at stage '{stage}': {error[:200]}")
            break
        else:
            sys.stdout.write(f"\r  [{elapsed:>3.0f}s] {stage} ({pct}%)")
            sys.stdout.flush()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print(f"Open: {API}/MANZAR.html")
    print("=" * 60)


if __name__ == "__main__":
    main()
