import cv2
path = "/workspace/output/pds_project_flame/train/ours_face_latest/gt/out.mp4"
cap = cv2.VideoCapture(path)
means = []
idx = 0
while idx < 20:
    ret, frame = cap.read()
    if not ret:
        break
    means.append(frame.mean())
    idx += 1
cap.release()
print("frames_read", idx)
print("mean_min", min(means) if means else None)
print("mean_max", max(means) if means else None)
print("means", means)