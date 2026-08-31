import os, time, json, re, ctypes
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# RealSense
# -----------------------------
import pyrealsense2 as rs

# -----------------------------
# CONFIG
# -----------------------------
ACTOR_VIDEO_PATH = r"input\actor.mp4" # UPDATED TO VIDEO

# ---- SLOW / STEADY CONTROL ----
SMOOTHING = 0.15          
MAX_DOT_SPEED_PX_S = 500  
DEADZONE_PX = 6           

# Fixation / saccade settings
VEL_THRESH_PX_PER_SEC = 250
MIN_FIX_DUR_SEC = 0.20

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# -----------------------------
# AOI labels 
# -----------------------------
AOI_LABELS = [
    "RIGHT_EYEBROW", "LEFT_EYEBROW",
    "RIGHT_EYE", "LEFT_EYE",
    "RIGHT_EAR", "LEFT_EAR",
    "NOSE", "LIPS",
    "FOREHEAD", "CHIN",
    "RIGHT_CHEEK", "LEFT_CHEEK",
    "HAIR",
    "NECK",
    "RIGHT_SHOULDER", "LEFT_SHOULDER",
    "FULL_FACE"
]

AOI_PRIORITY = [
    "RIGHT_EYE", "LEFT_EYE",
    "RIGHT_EYEBROW", "LEFT_EYEBROW",
    "NOSE", "LIPS",
    "RIGHT_EAR", "LEFT_EAR",
    "RIGHT_CHEEK", "LEFT_CHEEK",
    "FOREHEAD", "CHIN",
    "NECK",
    "RIGHT_SHOULDER", "LEFT_SHOULDER",
    "HAIR",
    "FULL_FACE"
]

OUTSIDE_ACTOR_XY_TO_ZERO = True

# -----------------------------
# Helpers
# -----------------------------
def safe_name(s: str) -> str:
    s = s.strip()
    if not s:
        return "UNKNOWN"
    s = re.sub(r'[^a-zA-Z0-9_\-]+', "_", s)
    return s[:60]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def clamp_point(x, y, w, h):
    return (max(0, min(int(x), w - 1)), max(0, min(int(y), h - 1)))

def centroid(points_np: np.ndarray):
    cx = int(np.mean(points_np[:, 0]))
    cy = int(np.mean(points_np[:, 1]))
    return cx, cy

def point_in_poly(px, py, poly_pts):
    if poly_pts is None or len(poly_pts) < 3:
        return False
    cnt = np.array(poly_pts, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(cnt, (float(px), float(py)), False) >= 0

def get_screen_size():
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

SCREEN_W, SCREEN_H = get_screen_size()

def show_fullscreen(window_name):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# -----------------------------
# AOI Annotation 
# -----------------------------
def draw_polygons(base_img, polygons, current_label=None, current_points=None):
    img = base_img.copy()
    h, w = img.shape[:2]

    for label, pts in polygons.items():
        if pts is None or len(pts) < 3:
            continue
        poly = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
        cx, cy = centroid(poly)
        cv2.putText(img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if current_label is not None:
        cv2.putText(img, f"Drawing: {current_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    if current_points is not None and len(current_points) > 0:
        for p in current_points:
            cv2.circle(img, tuple(p), 4, (0, 255, 255), -1)
        if len(current_points) >= 2:
            cv2.polylines(img, [np.array(current_points, dtype=np.int32)],
                          isClosed=False, color=(0, 255, 255), thickness=2)

    cv2.putText(
        img,
        "Left click:add | N:save+next | U:undo | R:reset | S:save | Q:quit",
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )
    return img

def save_master_aoi(master_json_path, overlay_path, base_img, polygons, actor_shape):
    data = {
        "video_path": ACTOR_VIDEO_PATH,
        "image_shape_hw": [int(actor_shape[0]), int(actor_shape[1])],
        "polygons": polygons
    }
    with open(master_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] Saved MASTER AOIs to: {master_json_path}")

    overlay = draw_polygons(base_img, polygons)
    cv2.imwrite(overlay_path, overlay)
    print(f"[OK] Saved MASTER overlay to: {overlay_path}")

def load_master_aoi(master_json_path):
    if not os.path.exists(master_json_path):
        return None
    with open(master_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    polys = data.get("polygons", {})
    out = {}
    for k, pts in polys.items():
        if pts is None:
            out[k] = None
        else:
            out[k] = [(int(p[0]), int(p[1])) for p in pts]
    return out

def annotate_master_aoi_once(base_img, master_json_path, overlay_path):
    h, w = base_img.shape[:2]
    polygons = {}

    idx = 0
    current_label = AOI_LABELS[idx]
    current_points = []

    win = "MASTER AOI Annotator (First Frame of Video)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        nonlocal current_points
        if event == cv2.EVENT_LBUTTONDOWN:
            x2, y2 = clamp_point(x, y, w, h)
            current_points.append([int(x2), int(y2)])

    cv2.setMouseCallback(win, on_mouse)

    print("\n=== MASTER AOI Annotation (ONE TIME) ===")
    while True:
        canvas = draw_polygons(base_img, polygons, current_label, current_points)
        cv2.imshow(win, canvas)

        key = cv2.waitKey(10) & 0xFF

        if key == ord('q'):
            save_master_aoi(master_json_path, overlay_path, base_img, polygons, base_img.shape)
            break
        elif key == ord('u'):
            if current_points:
                current_points.pop()
        elif key == ord('r'):
            current_points = []
        elif key == ord('s'):
            save_master_aoi(master_json_path, overlay_path, base_img, polygons, base_img.shape)
        elif key == ord('n'):
            if len(current_points) < 3:
                print("[WARN] Need at least 3 points.")
                continue
            polygons[current_label] = current_points.copy()
            current_points = []
            idx += 1
            if idx >= len(AOI_LABELS):
                save_master_aoi(master_json_path, overlay_path, base_img, polygons, base_img.shape)
                break
            current_label = AOI_LABELS[idx]

    cv2.destroyWindow(win)
    return polygons

# -----------------------------
# AOI classification
# -----------------------------
def classify_aoi_poly(ax, ay, aois_poly):
    for name in AOI_PRIORITY:
        if name in aois_poly and point_in_poly(ax, ay, aois_poly[name]):
            return name
    return "OUTSIDE"

def draw_polys(img, aois_poly, color=(0, 255, 0), thickness=2):
    out = img.copy()
    for name, pts in aois_poly.items():
        if pts is None or len(pts) < 3:
            continue
        cnt = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [cnt], isClosed=True, color=color, thickness=thickness)
    return out

# -----------------------------
# Eye feature & Calibration logic remains unchanged
# -----------------------------
def get_pupil_center(eye_gray):
    eye_blur = cv2.GaussianBlur(eye_gray, (7, 7), 0)
    _, th = cv2.threshold(eye_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 30: return None
    M = cv2.moments(c)
    if M["m00"] == 0: return None
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

def extract_eye_features(gray):
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(120, 120))
    if len(faces) == 0: return None
    fx, fy, fw, fh = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[0]
    face_roi = gray[fy:fy + fh, fx:fx + fw]
    eyes = EYE_CASCADE.detectMultiScale(face_roi, 1.1, 6, minSize=(30, 30))
    if len(eyes) == 0: return None
    eyes = sorted(eyes[:2], key=lambda e: e[0]) 
    feats = []
    for (ex, ey, ew, eh) in eyes:
        pupil = get_pupil_center(face_roi[ey:ey + eh, ex:ex + ew])
        if pupil is None: return None
        feats.extend([pupil[0] / float(ew), pupil[1] / float(eh)])
    if len(feats) != 4: return None
    return np.array(feats, dtype=np.float32)

def fit_affine(features, targets):
    X = np.hstack([features, np.ones((features.shape[0], 1), dtype=np.float32)]) 
    Y = targets.astype(np.float32) 
    W, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    return W.astype(np.float32)

def apply_affine(W, f):
    y = np.array([f[0], f[1], f[2], f[3], 1.0], dtype=np.float32) @ W
    return float(y[0]), float(y[1])

def run_calibration(pipeline, align, calib_bg_image, out_calib_path):
    win = "Actor Screen"
    show_fullscreen(win)
    margin = 0.12
    targets = [(int(x * SCREEN_W), int(y * SCREEN_H)) for y in [margin, 0.5, 1.0 - margin] for x in [margin, 0.5, 1.0 - margin]]
    samples_f, samples_t = [], []

    idx = 0
    while idx < len(targets):
        frames = align.process(pipeline.wait_for_frames())
        cf = frames.get_color_frame()
        if not cf: continue
        gray = cv2.cvtColor(np.asanyarray(cf.get_data()), cv2.COLOR_BGR2GRAY)
        
        display = calib_bg_image.copy()
        tx, ty = targets[idx]
        cv2.circle(display, (tx, ty), 16, (0, 0, 255), -1)
        cv2.putText(display, f"Calibration {idx+1}/{len(targets)}  SPACE=Capture  BACKSPACE=Undo",
                    (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.imshow(win, display)

        k = cv2.waitKey(1) & 0xFF
        if k == 27: return None
        if k == 8 and samples_f:
            samples_f.pop(); samples_t.pop(); idx = max(0, idx - 1)
        if k == 32:
            f = extract_eye_features(gray)
            if f is not None:
                samples_f.append(f); samples_t.append([tx, ty]); idx += 1

    W = fit_affine(np.array(samples_f, dtype=np.float32), np.array(samples_t, dtype=np.float32))
    calib = {"W": W.tolist(), "screen_w": SCREEN_W, "screen_h": SCREEN_H}
    with open(out_calib_path, "w", encoding="utf-8") as f: json.dump(calib, f, indent=2)
    return W

def load_calibration(path):
    if not os.path.exists(path): return None
    with open(path, "r", encoding="utf-8") as f: return np.array(json.load(f)["W"], dtype=np.float32)

def slow_follow(prev_x, prev_y, target_x, target_y, dt):
    if prev_x is None: return int(target_x), int(target_y)
    dist = ((target_x - prev_x)**2 + (target_y - prev_y)**2)**0.5
    if dist < DEADZONE_PX: target_x, target_y = prev_x, prev_y
    sx = prev_x + SMOOTHING * (target_x - prev_x)
    sy = prev_y + SMOOTHING * (target_y - prev_y)
    
    max_step = max(1.0, MAX_DOT_SPEED_PX_S * max(dt, 1e-3))
    step = ((sx - prev_x)**2 + (sy - prev_y)**2)**0.5
    if step > max_step:
        sx = prev_x + (sx - prev_x) * (max_step / step)
        sy = prev_y + (sy - prev_y) * (max_step / step)
    return int(sx), int(sy)

# -----------------------------
# MAIN
# -----------------------------
def main():
    patient_id = safe_name(input("Enter Patient ID (ex: P001): "))
    session_name = safe_name(input("Enter Session Name (ex: session_001): ") or "session_001")

    patient_dir = ensure_dir(os.path.join("output", patient_id))
    master_dir  = ensure_dir(os.path.join(patient_dir, "_MASTER"))
    session_dir = ensure_dir(os.path.join(patient_dir, session_name))

    # --- VIDEO SETUP ---
    cap = cv2.VideoCapture(ACTOR_VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found or cannot be opened: {ACTOR_VIDEO_PATH}")
    
    # Read the very first frame to establish scale and allow AOI drawing
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Video is empty.")
    
    ah, aw = first_frame.shape[:2]
    first_frame_full = cv2.resize(first_frame, (SCREEN_W, SCREEN_H), interpolation=cv2.INTER_LINEAR)

    # MASTER AOI paths 
    master_aoi_json = os.path.join(master_dir, "aois_polygons_master.json")
    master_overlay  = os.path.join(master_dir, "aoi_overlay_master.png")
    calib_json_path = os.path.join(session_dir, "calibration.json")

    # 1) Load or create MASTER AOIs (Using the first frame of the video)
    aois_poly = load_master_aoi(master_aoi_json)
    if aois_poly is None:
        print("Starting ONE-TIME manual AOI annotation on the first frame of the video...\n")
        annotate_master_aoi_once(first_frame, master_aoi_json, master_overlay)
        aois_poly = load_master_aoi(master_aoi_json)

    # 2) Load session calibration
    W = load_calibration(calib_json_path)

    # RealSense setup
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    win = "Actor Screen"
    show_fullscreen(win)

    rows = []
    prev_t, prev_sx, prev_sy = None, None, None
    fixation_id = 0
    fix_start_t = None
    in_fixation = False

    start_time = time.time()
    tracking_frame_index = 0

    try:
        while True:
            # --- READ VIDEO FRAME ---
            ret, vid_frame = cap.read()
            if not ret:
                # Video ended, loop back to the beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, vid_frame = cap.read()
            
            # Get current video frame index for data logging
            current_vid_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Resize the video frame to fill the screen
            display = cv2.resize(vid_frame, (SCREEN_W, SCREEN_H), interpolation=cv2.INTER_LINEAR)

            # --- READ WEBCAM FRAME ---
            frames = align.process(pipeline.wait_for_frames())
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

            t = time.time()
            if prev_t is None: prev_t = t
            dt = max(1e-3, t - prev_t)

            # draw AOIs (scaled) on top of the playing video
            if aois_poly is not None:
                sx_scale = SCREEN_W / float(aw)
                sy_scale = SCREEN_H / float(ah)
                aois_screen = {k: (None if pts is None else [(int(p[0]*sx_scale), int(p[1]*sy_scale)) for p in pts]) 
                               for k, pts in aois_poly.items()}
                display = draw_polys(display, aois_screen, color=(0, 255, 0), thickness=2)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k == 27: break
            if k == ord('c'):
                # Pass the static first frame for calibration so the user isn't distracted by a moving video
                W2 = run_calibration(pipeline, align, first_frame_full, calib_json_path)
                if W2 is not None: W = W2
                continue

            if W is None:
                cv2.putText(display, "Press 'c' to calibrate.", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
                cv2.imshow(win, display)
                prev_t = t
                continue

            f = extract_eye_features(gray)
            if f is None:
                cv2.putText(display, "Face/Eyes not detected", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
                cv2.imshow(win, display)
                prev_t = t
                continue

            # Gaze mapping & Smoothing
            sx_pred, sy_pred = apply_affine(W, f)
            sx_pred = clamp(int(sx_pred), 0, SCREEN_W - 1)
            sy_pred = clamp(int(sy_pred), 0, SCREEN_H - 1)
            sx_slow, sy_slow = slow_follow(prev_sx, prev_sy, sx_pred, sy_pred, dt)

            # depth
            depth_m = float(depth_frame.get_distance(320, 240))

            # screen -> actor coords (original video resolution)
            ax = clamp(int((sx_slow / float(SCREEN_W)) * aw), 0, aw - 1)
            ay = clamp(int((sy_slow / float(SCREEN_H)) * ah), 0, ah - 1)

            # Classify AOI
            aoi = "OUTSIDE"
            if aois_poly is not None:
                aoi = classify_aoi_poly(ax, ay, aois_poly)

            out_actor_x, out_actor_y = ax, ay
            if OUTSIDE_ACTOR_XY_TO_ZERO and aoi == "OUTSIDE":
                out_actor_x, out_actor_y = 0, 0

            # fixation / saccade math
            vel, event = np.nan, "NA"
            if prev_sx is not None and dt > 0:
                vel = (((sx_slow - prev_sx)**2 + (sy_slow - prev_sy)**2)**0.5) / dt
                if vel < VEL_THRESH_PX_PER_SEC:
                    if not in_fixation:
                        in_fixation, fix_start_t, fixation_id, event = True, t, fixation_id + 1, "FIX_START"
                    else:
                        event = "FIX" if (t - fix_start_t) >= MIN_FIX_DUR_SEC else "FIX_BUILD"
                else:
                    if in_fixation: event = "FIX_END"
                    in_fixation, fix_start_t, event = False, None, "SACCADE"

            rows.append({
                "patient_id": patient_id,
                "session_name": session_name,
                "tracking_frame": tracking_frame_index,
                "video_frame": current_vid_frame_idx,
                "timestamp": t - start_time,
                "screen_gaze_x": sx_slow,
                "screen_gaze_y": sy_slow,
                "actor_x": out_actor_x,
                "actor_y": out_actor_y,
                "depth_m": depth_m,
                "aoi": aoi,
                "velocity_px_s": vel,
                "event": event,
                "fixation_id": fixation_id if in_fixation else (fixation_id if event == "FIX_END" else np.nan)
            })

            cv2.circle(display, (sx_slow, sy_slow), 14, (0, 0, 255), -1)
            cv2.putText(display, f"AOI: {aoi} | Depth: {depth_m:.2f}m", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
            cv2.imshow(win, display)

            prev_sx, prev_sy, prev_t = sx_slow, sy_slow, t
            tracking_frame_index += 1

    finally:
        pipeline.stop()
        cap.release()
        cv2.destroyAllWindows()

    if len(rows) == 0: return

    df = pd.DataFrame(rows)
    session_dir = ensure_dir(os.path.join("output", patient_id, session_name))
    
    # Save Excel
    df.to_excel(os.path.join(session_dir, "data.xlsx"), index=False)

    # Heatmap (Mapped onto the First Frame of the video)
    heat = np.zeros((ah, aw), dtype=np.float32)
    d_ok = df[(df["actor_x"] != 0) & (df["actor_y"] != 0)]
    for _, r in d_ok.iterrows():
        x, y = int(r["actor_x"]), int(r["actor_y"])
        if 0 <= x < aw and 0 <= y < ah: heat[y, x] += 1

    heat_blur = cv2.GaussianBlur(heat, (0, 0), 15)
    heat_norm = cv2.normalize(heat_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
    
    # Overlay the heatmap on the very first frame of the video
    overlay = cv2.addWeighted(first_frame, 0.6, heat_color, 0.4, 0)
    cv2.imwrite(os.path.join(session_dir, "heatmap.png"), overlay)

    print("\nDONE ✅ Data saved in output directory.")

if __name__ == "__main__":
    main()
