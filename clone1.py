import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from collections import deque

# Pose setup
mp_pose = mp.solutions.pose
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS  # Using built-in connections

# Capture setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Matplotlib 3D setup
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(111, projection='3d')
plt.ion()

# Trail buffer
trail_buffer = deque(maxlen=10)

# FPS metrics
fps_times = deque(maxlen=10)
start_time = time.time()
frame_count = 0

def get_neon_color(time_stamp):
    hue = int(time_stamp * 60) % 180
    hsv = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    pulse = 0.8 + 0.2 * math.sin(time_stamp * 5)
    return tuple(int(c * pulse) for c in bgr)

def draw_pose_lines(img, landmarks_2d, color):
    for i, j in POSE_CONNECTIONS:
        pt1 = landmarks_2d[i]
        pt2 = landmarks_2d[j]
        if pt1 and pt2:
            cv2.line(img, pt1, pt2, color, 4)

def draw_neon_joints(img, landmarks_2d, color):
    for pt in landmarks_2d:
        if pt:
            cv2.circle(img, pt, 6, color, -1)
            cv2.circle(img, pt, 3, (255, 255, 255), -1)

# MediaPipe processing
with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                  smooth_landmarks=True, min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        neon_color = get_neon_color(time.time() - start_time)
        glow = np.zeros_like(frame)

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark
            lm_2d = [(int(lm[i].x * w), int(lm[i].y * h)) if lm[i].visibility > 0.5 else None for i in range(33)]
            lm_3d = [(lm[i].x - 0.5, -lm[i].y + 0.5, -lm[i].z) for i in range(33)]

            draw_pose_lines(glow, lm_2d, neon_color)
            draw_neon_joints(glow, lm_2d, neon_color)

            trail_buffer.append(lm_2d.copy())
            for idx, trail in enumerate(trail_buffer):
                fade = 0.6 * (1 - idx / len(trail_buffer))
                faded_color = tuple(int(c * fade) for c in neon_color)
                for pt in trail:
                    if pt:
                        cv2.circle(glow, pt, 3, faded_color, -1)

            frame = cv2.addWeighted(frame, 0.7, glow, 0.6, 0)

            # 3D Pose (update every 6 frames)
            if frame_count % 6 == 0:
                ax.clear()
                ax.set_xlim([-0.5, 0.5])
                ax.set_ylim([-0.5, 0.5])
                ax.set_zlim([-0.5, 0.5])
                ax.set_facecolor('black')
                ax.axis('off')
                color3d = tuple(c / 255 for c in neon_color)
                for i, j in POSE_CONNECTIONS:
                    x, y, z = zip(lm_3d[i], lm_3d[j])
                    ax.plot(x, y, z, color=color3d, lw=2, alpha=0.8)
                plt.draw()
                plt.pause(0.001)

        else:
            scan_color = get_neon_color((time.time() - start_time) * 2)
            msg = "SCANNING..."
            cv2.putText(frame, msg, (w//3, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, scan_color, 3)

        # FPS
        if frame_count % 5 == 0:
            now = time.time()
            fps = int(1 / (now - start_time + 1e-6))
            start_time = now
            fps_times.append(fps)

        avg_fps = int(sum(fps_times) / len(fps_times)) if fps_times else 0
        cv2.putText(frame, f"FPS: {avg_fps}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("NEON SKELETON TRACKER", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.close()
