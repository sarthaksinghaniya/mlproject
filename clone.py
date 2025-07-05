import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import time
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
                    (25, 27), (26, 28), (27, 31), (28, 32)]

# OpenCV Video Capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Matplotlib 3D plot setup
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')
plt.ion()

# Trail and FPS
trail_points = []
max_trail_length = 8
fps_times = []
frame_count = 0
start_time = time.time()

# Gradient color function
def get_gradient_color(t):
    hsv = np.uint8([[[int(t * 60) % 180, 255, 255]]])  # HSV hue in [0,180]
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in bgr)

# Drawing functions
def draw_smooth_line(img, pt1, pt2, color, thickness):
    cv2.line(img, pt1, pt2, color, thickness)
    cv2.line(img, pt1, pt2, tuple(c//2 for c in color), thickness+2)

def draw_smooth_circle(img, center, radius, color):
    cv2.circle(img, center, radius, color, -1)
    cv2.circle(img, center, radius//2, (255, 255, 255), -1)

# Use MediaPipe pose in context manager
with mp_pose.Pose(static_image_mode=False, model_complexity=0, smooth_landmarks=True, 
                  min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        current_time = time.time()
        wave_time = current_time - start_time

        # Color pulsing and trail
        gradient_color = get_gradient_color(wave_time)
        pulse = 0.8 + 0.2 * math.sin(wave_time * 6)
        neon_color = tuple(int(c * pulse) for c in gradient_color)

        height, width, _ = frame.shape
        glow_layer = np.zeros_like(frame, dtype=np.uint8)

        if results.pose_landmarks:
            clone_2d = [(int((1 - lm.x) * width), int(lm.y * height)) for lm in results.pose_landmarks.landmark]
            clone_3d = [(1 - lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]

            for i, j in POSE_CONNECTIONS:
                draw_smooth_line(glow_layer, clone_2d[i], clone_2d[j], neon_color, thickness=6)

            joint_points = []
            for x, y in clone_2d:
                if 0 <= x < width and 0 <= y < height:
                    joint_points.append((x, y))
                    draw_smooth_circle(glow_layer, (x, y), 8, neon_color)

            trail_points.append(joint_points.copy())
            if len(trail_points) > max_trail_length:
                trail_points.pop(0)

            for t_idx, trail in enumerate(trail_points[:-1]):
                fade = 0.5 * (1 - math.cos(t_idx / len(trail_points) * math.pi))
                trail_color = tuple(int(c * fade) for c in neon_color)
                for x, y in trail:
                    cv2.circle(glow_layer, (x, y), 3, trail_color, -1)

            frame = cv2.addWeighted(frame, 0.7, glow_layer, 0.6, 0)

            if frame_count % 6 == 0:
                ax.clear()
                ax.set_xlim([-1, 1])
                ax.set_ylim([-1, 1])
                ax.set_zlim([-1, 1])
                ax.set_facecolor('black')
                ax.set_title("3D Clone", color='cyan', fontsize=12)
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                rgb_color = tuple(c/255 for c in neon_color)
                for i, j in POSE_CONNECTIONS:
                    x_vals, y_vals, z_vals = zip(clone_3d[i], clone_3d[j])
                    ax.plot(x_vals, y_vals, z_vals, color=rgb_color, linewidth=3, alpha=0.8)
                for x, y, z in clone_3d:
                    ax.scatter(x, y, z, c=[rgb_color], s=50, alpha=0.8)
                plt.draw()
                plt.pause(0.001)

        else:
            scan_color = get_gradient_color(wave_time * 2)
            scan_text = "SCANNING..."
            text_size = cv2.getTextSize(scan_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2
            cv2.putText(frame, scan_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, scan_color, 2)

        # FPS Calculation
        fps = 0
        if frame_count % 10 == 0:
            elapsed = time.time() - current_time
            fps = int(1.0 / (elapsed + 1e-6))
            fps_times.append(fps)
            if len(fps_times) > 10:
                fps_times.pop(0)

        if fps_times:
            avg_fps = int(sum(fps_times) / len(fps_times))
            cv2.putText(frame, f"FPS: {avg_fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("CLONE TRACKER", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.close()