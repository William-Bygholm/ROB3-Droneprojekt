# save_features_chunks.py
import cv2, joblib, numpy as np, os, time

VIDEO_IN = r"C:\Users\alexa\Documents\GitHub\ROB3-Droneprojekt\ProjektVideoer\2 militær med blå bånd .MP4"
MODEL_FILE = "Person_Detector_Json+YOLO.pkl"
OUTPUT_DIR = "hog_chunks"

SCALES = [1.2, 1.0, 0.8, 0.64]
STEP_SIZES = {1.2: 36, 1.0: 32, 0.8: 28, 0.64: 24}
WINDOW_SIZE = (128, 256)
FRAME_SKIP = 2
CHUNK_SIZE = 50

# Load model
model_data = joblib.load(MODEL_FILE)
clf = model_data["classifier"] if isinstance(model_data, dict) else model_data
hog = cv2.HOGDescriptor(WINDOW_SIZE,(32,32),(16,16),(8,8),9)

def sliding_windows(img, step, win_size):
    w,h = win_size
    for y in range(0,img.shape[0]-h+1,step):
        for x in range(0,img.shape[1]-w+1,step):
            yield x,y,img[y:y+h,x:x+w]

os.makedirs(OUTPUT_DIR, exist_ok=True)
cap = cv2.VideoCapture(VIDEO_IN)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_id = 0
chunk_id = 0
start_time = time.time()
all_data = []

while True:
    ret,frame = cap.read()
    if not ret: break
    frame_id += 1
    if frame_id % FRAME_SKIP != 0: continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h0,w0 = gray.shape[:2]

    feats,boxes = [],[]
    for scale in SCALES:
        resized = cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        step = STEP_SIZES[scale]
        sx,sy = w0/resized.shape[1], h0/resized.shape[0]
        for x,y,win in sliding_windows(resized,step,WINDOW_SIZE):
            if win.shape!=(WINDOW_SIZE[1],WINDOW_SIZE[0]): continue
            feat = hog.compute(win).astype(np.float32).ravel()
            feats.append(feat)
            x1,y1 = int(x*sx), int(y*sy)
            x2,y2 = int((x+WINDOW_SIZE[0])*sx), int((y+WINDOW_SIZE[1])*sy)
            boxes.append([x1,y1,x2,y2])

    all_data.append({"frame":frame_id,"features":feats,"boxes":boxes})

    # Progress
    elapsed = time.time()-start_time
    progress = frame_id/total_frames
    est_total = elapsed/progress if progress>0 else 0
    remaining = est_total-elapsed
    print(f"Frame {frame_id}/{total_frames} "
          f"({progress*100:.2f}%), "
          f"Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s", end="\r")

    # Gem chunk
    if len(all_data) >= CHUNK_SIZE//FRAME_SKIP:
        np.save(os.path.join(OUTPUT_DIR,f"chunk_{chunk_id}.npy"),
                np.array(all_data,dtype=object))
        print(f"\n[INFO] Saved chunk {chunk_id} with {len(all_data)} frames")
        all_data = []
        chunk_id += 1

cap.release()
# Gem sidste chunk
if all_data:
    np.save(os.path.join(OUTPUT_DIR,f"chunk_{chunk_id}.npy"),
            np.array(all_data,dtype=object))
    print(f"\n[INFO] Saved final chunk {chunk_id}")
