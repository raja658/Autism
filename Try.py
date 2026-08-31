import os, time, json, re, ctypes
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import pyrealsense2 as rs

# -----------------------------
# CONFIG
# -----------------------------
ACTOR_VIDEO_PATH = r"input\actor.mp4"

# Gaze Smoothing
SMOOTHING = 0.15
MAX_DOT_SPEED_PX_S = 500
DEADZONE_PX = 6

# Fixation Settings
VEL_THRESH_PX_PER_SEC = 250
MIN_FIX_DUR_SEC = 0.20

# Haar Cascades for USER eye tracking (RealSense)
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# MediaPipe Setup for ACTOR tracking (Video)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5
)

# EXACT 17 AOIs AS REQUESTED
AOI_LABELS = [
    "RIGHT_EYEBROW", "LEFT_EYEBROW",
    "RIGHT_EYE", "LEFT_EYE",
    "RIGHT_EAR", "LEFT_EAR",
    "NOSE", "LIPS",
    "FOREHEAD", "CHIN",
    "RIGHT_CHEEK", "LEFT_CHEEK",
    "HAIR", "NECK",
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

# MediaPipe Index Mapping for precise internal facial features
# Note: "Right" and "Left" in AI refers to the image axis (Actor's Right = Screen Left)
MP_INDICES = {
    "RIGHT_EYEBROW": [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
    "LEFT_EYEBROW": [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
    "RIGHT_EYE": [33, 160, 158, 133, 153, 144],
    "LEFT_EYE": [362, 385, 387, 263, 373, 380],
    "NOSE": [168, 197, 5, 4, 19, 94, 2, 278, 344, 440, 275, 220, 45, 274],
    "LIPS": [61, 39, 0, 269, 291, 405, 17, 181],
    "FOREHEAD": [103, 67, 109, 10, 338, 297, 332, 285, 295, 282, 283, 276, 168, 46, 53, 52, 65, 55],
    "CHIN": [150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 435, 367, 364, 394, 395, 369, 396, 175, 200, 201, 208, 171, 140, 170, 169, 210, 212, 214, 192, 213, 147, 123, 117, 118, 101, 50, 36, 205, 206, 207, 216],
    "RIGHT_CHEEK": [137, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 200, 201, 208, 171, 140, 170, 169, 210, 212, 214, 192, 213, 147, 123, 117, 118, 101, 50, 36, 205, 206, 207, 216],
    "LEFT_CHEEK": [366, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
    "FULL_FACE": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
}

# -----------------------------
# Helpers
# -----------------------------
def safe_name(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\-]+', "_", s.strip())[:60] if s.strip() else "UNKNOWN"

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def get_screen_size():
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

SCREEN_W, SCREEN_H = get_screen_size()

def show_fullscreen(window_name):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def point_in_poly(px, py, poly_pts):
    if not poly_pts: return False
    cnt = np.array(poly_pts, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(cnt, (float(px), float(py)), False) >= 0

def classify_aoi_poly(ax, ay, aois_poly):
    for name in AOI_PRIORITY:
        if name in aois_poly and point_in_poly(ax, ay, aois_poly[name]):
            return name
    return "OUTSIDE"

def draw_polys(img, aois_poly, color=(0, 255, 0), thickness=2):
    out = img.copy()
    for name, pts in aois_poly.items():
        if not pts: continue
        cnt = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [cnt], isClosed=True, color=color, thickness=thickness)
    return out

# -----------------------------
# DYNAMIC AOI GENERATOR (17 AOIs)
# -----------------------------
def extract_dynamic_aois(frame_bgr):
    """Processes frame via MediaPipe. Uses exact indices for facial features, 
       and anchors geometric boxes for Hair, Neck, Shoulders, and Ears."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    aois = {label: [] for label in AOI_LABELS}
    if not results.multi_face_landmarks:
        return aois

    h, w = frame_bgr.shape[:2]
    landmarks = results.multi_face_landmarks[0].landmark
    
    # 1. Map internal facial features using precise indices
    for aoi_name, indices in MP_INDICES.items():
        pts = []
        for idx in indices:
            # Prevent points from crossing frame boundaries
            lx = clamp(int(landmarks[idx].x * w), 0, w-1)
            ly = clamp(int(landmarks[idx].y * h), 0, h-1)
            pts.append((lx, ly))
        aois[aoi_name] = pts

    # 2. Extrapolate external features based on the FULL_FACE bounding box
    if aois["FULL_FACE"]:
        face_pts = np.array(aois["FULL_FACE"])
        x_min, y_min = np.min(face_pts, axis=0)
        x_max, y_max = np.max(face_pts, axis=0)
        fw, fh = x_max - x_min, y_max - y_min

        # HAIR: Box directly above the top of the face mesh
        aois["HAIR"] = [
            (x_min, y_min - int(fh*0.35)), (x_max, y_min - int(fh*0.35)),
            (x_max, y_min), (x_min, y_min)
        ]
        
        # NECK: Box directly below the chin
        aois["NECK"] = [
            (x_min + int(fw*0.2), y_max), (x_max - int(fw*0.2), y_max),
            (x_max - int(fw*0.2), y_max + int(fh*0.35)), (x_min + int(fw*0.2), y_max + int(fh*0.35))
        ]

        # SHOULDERS: Boxes expanding outward and downward from the neck
        aois["RIGHT_SHOULDER"] = [
            (max(0, x_min - int(fw*0.5)), y_max + int(fh*0.1)), 
            (x_min + int(fw*0.2), y_max + int(fh*0.1)),
            (x_min + int(fw*0.2), min(h-1, y_max + fh)), 
            (max(0, x_min - int(fw*0.5)), min(h-1, y_max + fh))
        ]
        
        aois["LEFT_SHOULDER"] = [
            (x_max - int(fw*0.2), y_max + int(fh*0.1)), 
            (min(w-1, x_max + int(fw*0.5)), y_max + int(fh*0.1)),
            (min(w-1, x_max + int(fw*0.5)), min(h-1, y_max + fh)), 
            (x_max - int(fw*0.2), min(h-1, y_max + fh))
        ]

        # EARS: Small boxes attached to the extreme left/right of the face mesh
        aois["RIGHT_EAR"] = [
            (max(0, x_min - int(fw*0.15)), y_min + int(fh*0.35)), 
            (x_min, y_min + int(fh*0.35)),
            (x_min, y_min + int(fh*0.65)), 
            (max(0, x_min - int(fw*0.15)), y_min + int(fh*0.65))
        ]
        
        aois["LEFT_EAR"] = [
            (x_max, y_min + int(fh*0.35)), 
            (min(w-1, x_max + int(fw*0.15)), y_min + int(fh*0.35)),
            (min(w-1, x_max + int(fw*0.15)), y_min + int(fh*0.65)), 
            (x_max, y_min + int(fh*0.65))
        ]

    return aois

# -----------------------------
# USER EYE TRACKING & CALIBRATION (RealSense)
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
    with open(out_calib_path, "w", encoding="utf-8") as f: 
        json.dump({"W": W.tolist(), "screen_w": SCREEN_W, "screen_h": SCREEN_H}, f, indent=2)
    return W

def load_calibration(path):
    if not os.path.exists(path): return None
    with open(path, "r", encoding="utf-8") as f: 
        return np.array(json.load(f)["W"], dtype=np.float32)

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
# MAIN LOOP
# -----------------------------
def main():
    patient_id = safe_name(input("Enter Patient ID (ex: P001): "))
    session_name = safe_name(input("Enter Session Name (ex: session_001): ") or "session_001")

    session_dir = ensure_dir(os.path.join("output", patient_id, session_name))
    calib_json_path = os.path.join(session_dir, "calibration.json")

    # --- VIDEO SETUP ---
    cap = cv2.VideoCapture(ACTOR_VIDEO_PATH)
    if not cap.isOpened(): raise FileNotFoundError("Video not found.")
    ret, first_frame = cap.read()
    ah, aw = first_frame.shape[:2]
    first_frame_full = cv2.resize(first_frame, (SCREEN_W, SCREEN_H))

    W = load_calibration(calib_json_path)

    # --- REALSENSE SETUP ---
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
    fixation_id, in_fixation = 0, False
    start_time = time.time()
    tracking_frame_index = 0

    try:
        while True:
            # 1. READ VIDEO
            ret, vid_frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, vid_frame = cap.read()
            current_vid_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # 2. DYNAMICALLY GENERATE 17 AOIs
            dynamic_aois = extract_dynamic_aois(vid_frame)
            display = cv2.resize(vid_frame, (SCREEN_W, SCREEN_H))

            # 3. READ REALSENSE
            frames = align.process(pipeline.wait_for_frames())
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame: continue

            gray = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2GRAY)
            t = time.time()
            dt = max(1e-3, t - (prev_t if prev_t else t))

            # Scale and draw the dynamically generated AOIs to the screen resolution
            if dynamic_aois:
                sx_scale, sy_scale = SCREEN_W / float(aw), SCREEN_H / float(ah)
                aois_screen = {k: [(int(p[0]*sx_scale), int(p[1]*sy_scale)) for p in pts] for k, pts in dynamic_aois.items()}
                display = draw_polys(display, aois_screen, color=(0, 255, 0), thickness=1)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k == 27: break
            if k == ord('c'):
                W = run_calibration(pipeline, align, first_frame_full, calib_json_path) or W
                continue

            if W is None:
                cv2.putText(display, "Press 'c' to calibrate.", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
                cv2.imshow(win, display)
                prev_t = t
                continue

            # 4. PREDICT USER GAZE
            f = extract_eye_features(gray)
            if f is None:
                cv2.imshow(win, display)
                prev_t = t
                continue

            sx_pred, sy_pred = apply_affine(W, f)
            sx_slow, sy_slow = slow_follow(prev_sx, prev_sy, clamp(int(sx_pred), 0, SCREEN_W - 1), clamp(int(sy_pred), 0, SCREEN_H - 1), dt)
            
            # 5. CLASSIFY GAZE AGAINST DYNAMIC AOIS
            ax, ay = clamp(int((sx_slow / float(SCREEN_W)) * aw), 0, aw - 1), clamp(int((sy_slow / float(SCREEN_H)) * ah), 0, ah - 1)
            aoi = classify_aoi_poly(ax, ay, dynamic_aois) if dynamic_aois else "OUTSIDE"
            
            out_actor_x, out_actor_y = (0, 0) if (OUTSIDE_ACTOR_XY_TO_ZERO and aoi == "OUTSIDE") else (ax, ay)
            depth_m = float(depth_frame.get_distance(320, 240))

            # 6. FIXATION LOGIC
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

            # 7. LOGGING & DISPLAY
            rows.append({
                "patient_id": patient_id,
                "session_name": session_name,
                "tracking_frame": tracking_frame_index,
                "video_frame": current_vid_frame_idx,
                "timestamp": t - start_time,
                "screen_gaze_x": sx_slow, "screen_gaze_y": sy_slow,
                "actor_x": out_actor_x, "actor_y": out_actor_y,
                "depth_m": depth_m, "aoi": aoi,
                "velocity_px_s": vel, "event": event,
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

    if rows:
        df = pd.DataFrame(rows)
        df.to_excel(os.path.join(session_dir, "data.xlsx"), index=False)
        print("\nDONE ✅ Data saved in output directory.")

if __name__ == "__main__":
    main()