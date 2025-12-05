# save_features.py
import cv2, joblib, numpy as np, os

VIDEO_IN = r"C:\Users\alexa\Documents\GitHub\ROB3-Droneprojekt\ProjektVideoer\2 militær med blå bånd .MP4"
MODEL_FILE = "Person_Detector_Json+YOLO.pkl"
FEATURE_DIR = "hog_features"

SCALES = [1.2, 1.0, 0.8, 0.64]
STEP_SIZES = {1.2: 36, 1.0: 32, 0.8: 28, 0.64: 24}
WINDOW_SIZE = (128, 256)
FRAME_SKIP = 2

model_data = joblib.load(MODEL_FILE)
clf = model_data["classifier"] if isinstance(model_data, dict) else model_data
hog = cv2.HOGDescriptor(WINDOW_SIZE,(32,32),(16,16),(8,8),9)

def sliding_windows(img, step, win_size):
    w,h = win_size
    for y in range(0,img.shape[0]-h+1,step):
        for x in range(0,img.shape[1]-w+1,step):
            yield x,y,img[y:y+h,x:x+w]

os.makedirs(FEATURE_DIR, exist_ok=True)
cap = cv2.VideoCapture(VIDEO_IN)
frame_id=0

while True:
    ret,frame=cap.read()
    if not ret: break
    frame_id+=1
    if frame_id%FRAME_SKIP!=0: continue

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    h0,w0=gray.shape[:2]
    feats,boxes=[],[]

    for scale in SCALES:
        resized=cv2.resize(gray,None,fx=scale,fy=scale)
        step=STEP_SIZES[scale]
        sx,sy=w0/resized.shape[1],h0/resized.shape[0]
        for x,y,win in sliding_windows(resized,step,WINDOW_SIZE):
            if win.shape!=(WINDOW_SIZE[1],WINDOW_SIZE[0]): continue
            feat=hog.compute(win).ravel()
            feats.append(feat)
            x1,y1=int(x*sx),int(y*sy)
            x2,y2=int((x+WINDOW_SIZE[0])*sx),int((y+WINDOW_SIZE[1])*sy)
            boxes.append([x1,y1,x2,y2])

    np.save(os.path.join(FEATURE_DIR,f"frame_{frame_id}.npy"),
            {"features":np.array(feats),"boxes":np.array(boxes)})
