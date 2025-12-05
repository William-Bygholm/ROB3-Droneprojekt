# save_features.py
import cv2, joblib, numpy as np, os, time

VIDEO_IN = r"C:\Users\alexa\Documents\GitHub\ROB3-Droneprojekt\ProjektVideoer\2 militær med blå bånd .MP4"
MODEL_FILE = "Person_Detector_Json+YOLO.pkl"
OUTPUT_FILE = "hog_features_all.npy"

SCALES = [1.2, 1.0, 0.8, 0.64]
STEP_SIZES = {1.2: 36, 1.0: 32, 0.8: 28, 0.64: 24}
WINDOW_SIZE = (128, 256)
FRAME_SKIP = 2

# ---------------- LOAD MODEL ----------------
model_data = joblib.load(MODEL_FILE)
clf = model_data["classifier"] if isinstance(model_data, dict) else model_data
hog = cv2.HOGDescriptor(WINDOW_SIZE,(32,32),(16,16),(8,8),9)

def sliding_windows(img, step, win_size):
    w,h = win_size
    for y in range(0,img.shape[0]-h+1,step):
        for x in range(0,img.shape[1]-w+1,step):
            yield x,y,img[y:y+h,x:x+w]

# ---------------- PROCESS VIDEO ----------------
cap = cv2.VideoCapture(VIDEO_IN)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_id = 0
start_time = time.time()

all_data = []  # gemmer dicts med features og bokse per frame

while True:
    ret,frame = cap.read()
    if not ret: break
    frame_id += 1
    if frame_id % FRAME_SKIP != 0: continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h0,w0 = gray.shape[:2]

    feats,boxes = [],[]
    for scale in SCALES:
        resized = cv2.resize(gray,None,fx=scale,fy=scale)
        step = STEP_SIZES[scale]
        sx,sy = w0/resized.shape[1], h0/resized.shape[0]
        for x,y,win in sliding_windows(resized,step,WINDOW_SIZE):
            if win.shape!=(WINDOW_SIZE[1],WINDOW_SIZE[0]): continue
            feat = hog.compute(win).ravel()
            feats.append(feat)
            x1,y1 = int(x*sx), int(y*sy)
            x2,y2 = int((x+WINDOW_SIZE[0])*sx), int((y+WINDOW_SIZE[1])*sy)
            boxes.append([x1,y1,x2,y2])

    all_data.append({
    "frame": frame_id,
    "features": [f.astype(np.float32) for f in feats],  # gem som liste
    "boxes": boxes
})


    # Progress info
    elapsed = time.time()-start_time
    progress = frame_id/total_frames
    if progress>0:
        est_total = elapsed/progress
        remaining = est_total-elapsed
        print(f"Frame {frame_id}/{total_frames} "
              f"({progress*100:.2f}%), "
              f"Elapsed: {elapsed:.1f}s, "
              f"Remaining: {remaining:.1f}s", end="\r")

cap.release()

# ---------------- SAVE ALL ----------------
np.save(OUTPUT_FILE, np.array(all_data, dtype=object))
print(f"\n[INFO] Saved all features and boxes to {OUTPUT_FILE}")
