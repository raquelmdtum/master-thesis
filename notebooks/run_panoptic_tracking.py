import os
import cv2 as cv
import torch
import pandas as pd
import numpy as np
import argparse

from transformers import (Mask2FormerImageProcessor,Mask2FormerForUniversalSegmentation)
from ultralytics import YOLO

# ==================================================
# ARGUMENTS
# ==================================================
parser = argparse.ArgumentParser()

parser.add_argument("--participant",type=str,required=True)
parser.add_argument("--session",type=str,required=True)
args = parser.parse_args()

participant = args.participant
session_name = args.session

# ==================================================
# PATHS
# ==================================================
data_root = "/work3/s243636/data/raw_data"
aligned_root = "/work3/s243636/data/aligned_gaze"
output_root = "/work3/s243636/data/fused_output"

os.makedirs(output_root,exist_ok=True)

# ==================================================
# LOAD GAZE
# ==================================================
gaze_path = os.path.join(
    aligned_root,
    participant,
    session_name,
    "aligned_gaze.csv"
)

if not os.path.exists(gaze_path):
    raise FileNotFoundError(f"Gaze file not found:\n{gaze_path}")

# Converts timestamp column into datetime objects
aligned = pd.read_csv(gaze_path,parse_dates=["Seconds"])
aligned = aligned.rename(columns={"Seconds": "timestamp"})
aligned = aligned.sort_values(["frame", "timestamp"]).reset_index(drop=True)

# Groups gaze rows by video frame
gaze_groups = aligned.groupby("frame")

# ==================================================
# FIND VIDEO
# ==================================================
participant_path = os.path.join(data_root,participant)

if not os.path.exists(participant_path):
    raise FileNotFoundError(f"Missing participant path:\n{participant_path}")

session_path = None

for s in os.listdir(participant_path):

    if session_name in s:
        session_path = os.path.join(participant_path,s)
        break

if session_path is None:
    raise FileNotFoundError(f"Session not found:\n{session_name}")

video_path = os.path.join(session_path,"pupil_video.avi")

if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video not found:\n{video_path}")

# Video stream for frame-by-frame reading
cap = cv.VideoCapture(video_path)

# ==================================================
# DEVICE
# ==================================================
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"\nUsing device: {device}")

# ==================================================
# LOAD MASK2FORMER
# ==================================================
processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-panoptic")

# Loads pretrained Mask2Former model
seg_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-panoptic").to(device)
seg_model.eval()

# Maps class IDs into readable labels
id2label = seg_model.config.id2label

print("✓ Mask2Former loaded")

# ==================================================
# LOAD YOLO
# ==================================================
yolo_model = YOLO("yolov8m.pt")
print("✓ YOLO loaded")

# ==================================================
# PARAMETERS
# ==================================================
VEL_THRESHOLD = 100
FLOW_THRESHOLD = 25
PATCH_RADIUS = 100

# ==================================================
# STORAGE
# ==================================================
results = []

prev_frame_gray = None
prev_gaze = None
prev_timestamp = None
frame_idx = 0

# ==================================================
# MAIN LOOP
# ==================================================
# Processes the entire video sequentially
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Skip empty frames
    if frame_idx not in gaze_groups.groups:

        frame_idx += 1
        continue

    gaze_rows = gaze_groups.get_group(frame_idx)

    frame_rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)

    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    h, w = gray.shape

    print(f"Processing frame {frame_idx} "f"({len(gaze_rows)} gaze samples)")

    # ==================================================
    # PANOPTIC SEGMENTATION
    # ==================================================
    inputs = processor(
        images=frame_rgb,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        outputs = seg_model(**inputs)

    # Converts raw tensors into segmentation maps and object metadata
    panoptic = processor.post_process_panoptic_segmentation(
        outputs,
        target_sizes=[frame_rgb.shape[:2]],
        label_ids_to_fuse=set()
    )[0]

    # Segmentation map > each pixel contains a segment ID
    seg_map = panoptic["segmentation"].cpu().numpy()
    # example: pixel (300,400) -> segment 42 & segment 42 -> sidewalk

    # Lookup: segment_id -> label
    segment_dict = {
        s["id"]: s
        for s in panoptic["segments_info"]
    }

    # ==================================================
    # FULL FRAME CONTEXT
    # ==================================================
    # Computing what proportion of the scene is each class
    unique_ids, counts = np.unique(seg_map,return_counts=True)

    frame_context = {}

    total_pixels = counts.sum()

    for sid, count in zip(unique_ids, counts):

        segment = segment_dict.get(int(sid))

        if segment is None:
            continue

        label = id2label.get(
            segment["label_id"],
            "unknown"
        )

        percentage = round(
            100 * count / total_pixels,
            2
        )

        frame_context[label] = (
            frame_context.get(label, 0)
            + percentage
        )

    # ==================================================
    # YOLO TRACKING
    # ==================================================
    tracked_objects = []

    try:

        # Performs detection, tracking and identity assignment using BoT-SORT
        track_results = yolo_model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            verbose=False
        )

        if (
            track_results[0] is not None
            and track_results[0].boxes is not None
            and track_results[0].boxes.id is not None
        ):

            boxes = (
                track_results[0]
                .boxes
                .xyxy
                .cpu()
                .numpy()
            )

            ids = (
                track_results[0]
                .boxes
                .id
                .cpu()
                .numpy()
            )

            classes = (
                track_results[0]
                .boxes
                .cls
                .cpu()
                .numpy()
            )

            confidences = (
                track_results[0]
                .boxes
                .conf
                .cpu()
                .numpy()
            )

            for box, tid, cls, conf in zip(
                boxes,
                ids,
                classes,
                confidences
            ):

                x1, y1, x2, y2 = map(
                    int,
                    box
                )

                tracked_objects.append({

                    "track_id": int(tid),
                    "class_id": int(cls),
                    "class_name": yolo_model.names[int(cls)],
                    "confidence": float(conf),
                    "bbox": (x1,y1,x2,y2)
                })

    except Exception as e:
        print(f"Tracking failed at frame "f"{frame_idx}: {e}")

    # ==================================================
    # OPTICAL FLOW
    # ==================================================
    flow = None

    # Computes pixel-wise motion field between previous frame and current frame!
    # Which estimates camera movement and scene motion
    if prev_frame_gray is not None:

        flow = cv.calcOpticalFlowFarneback(
            prev_frame_gray,
            gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )

    # ==================================================
    # PROCESS GAZE
    # ==================================================
    # Going through each individual gaze point
    for _, row in gaze_rows.iterrows():

        # Skips missing gaze points
        if (
            pd.isna(row["gaze_x"])
            or pd.isna(row["gaze_y"])
        ):
            continue

        x = int(row["gaze_x"])
        y = int(row["gaze_y"])

        # Ensures gaze stays inside image bounds
        x = np.clip(x, 0, w - 1)
        y = np.clip(y, 0, h - 1)

        # ==================================================
        # SEGMENTATION LABEL
        # ==================================================
        # Gets the segment under gaze
        segment_id = int(seg_map[y, x])

        segment = segment_dict.get(segment_id,None)
        segmentation_label = "unknown"
        panoptic_segment_id = None

        if segment is not None:
            segmentation_label = id2label.get(segment["label_id"],"unknown")
            panoptic_segment_id = segment_id

        # ==================================================
        # TRACKED OBJECT LABEL
        # ==================================================
        tracked_object_label = None
        track_id = None
        tracking_confidence = None

        matching_objects = []

        for obj in tracked_objects:

            x1, y1, x2, y2 = obj["bbox"]

            # Looping through YOLO boxes to check whether gaze falls inside an object
            if (
                x1 <= x <= x2
                and y1 <= y <= y2
            ):  # if yes then give name, etc

                box_area = (x2 - x1) * (y2 - y1)

                matching_objects.append({

                    "label": obj["class_name"],
                    "track_id": obj["track_id"],
                    "confidence": obj["confidence"],
                    "bbox_area": box_area
                })

        # --------------------------------------------------
        # SELECT BEST MATCH
        # --------------------------------------------------
        if len(matching_objects) > 0:

            # choose smallest enclosing object and highest confidence breaks ties
            best_obj = min(
                matching_objects,
                key=lambda o: (
                    o["bbox_area"],
                    -o["confidence"]
                )
            )

            tracked_object_label = best_obj["label"]
            track_id = best_obj["track_id"]
            tracking_confidence = best_obj["confidence"]

        # ==================================================
        # LABEL FUSION
        # ==================================================
        # Default output label is semantic segmentation (in case tracked object is blank)
        source_priority = "segmentation"

        final_label = segmentation_label

        # Priotitizing tracked_object over segmentation bc object tracking is more specific
        if tracked_object_label is not None:

            source_priority = "tracked_object"
            final_label = tracked_object_label

        # ==================================================
        # AGREEMENT / CONFLICT
        # ==================================================
        # Checks whether Mask2Former (segmentation) and YOLO + BotSORT (tracking) predict same class
        models_agree = None

        # if not then what is the difference?
        conflict_type = None

        if tracked_object_label is not None:

            models_agree = (segmentation_label == tracked_object_label)

            if not models_agree:
                conflict_type = (f"{segmentation_label}"f"_vs_"f"{tracked_object_label}")

        # ==================================================
        # LOCAL GAZE CONTEXT
        # ==================================================
        # Defines patch around gaze
        gx1 = max(0, x - PATCH_RADIUS)
        gx2 = min(w, x + PATCH_RADIUS)
        gy1 = max(0, y - PATCH_RADIUS)
        gy2 = min(h, y + PATCH_RADIUS)

        # Extracts segmentation patch around gaze
        local_patch = seg_map[gy1:gy2,gx1:gx2]

        unique_local, counts_local = np.unique(local_patch,return_counts=True)

        gaze_context = {}

        total_local = counts_local.sum()

        for sid, count in zip(
            unique_local,
            counts_local
        ):

            segment = segment_dict.get(int(sid))

            if segment is None:
                continue

            label = id2label.get(segment["label_id"],"unknown")
            percentage = round(100 * count / total_local,2)

            # Stores local semantic composition
            gaze_context[label] = (
                gaze_context.get(label, 0)
                + percentage
            )

        # ==================================================
        # TEMPORAL FEATURES
        # ==================================================
        gaze_velocity = None
        relative_motion = None
        is_static_fixation = None
        is_smooth_pursuit = None
        is_saccade = None

        if (
            prev_gaze is not None
            and prev_timestamp is not None
        ):
            # Computes elapsed time between gaze samples
            dt = (
                row["timestamp"] - prev_timestamp
            ).total_seconds()

            if dt > 0:

                gaze_dx = x - prev_gaze[0]
                gaze_dy = y - prev_gaze[1]

                # Computes Euclidean gaze displacement magnitude
                gaze_distance = np.sqrt(gaze_dx**2 + gaze_dy**2)

                # Converts displacement into pixels per second
                gaze_velocity = (gaze_distance / dt)

                # ------------------------------------------
                # MOTION-COMPENSATED VELOCITY
                # ------------------------------------------
                if flow is not None:
                    
                    # Extracts local optical-flow region around gaze
                    patch = flow[
                        max(0, y - 2):y + 3,
                        max(0, x - 2):x + 3
                    ]

                    # Computes average scene motion near gaze
                    flow_dx = np.mean(patch[..., 0])
                    flow_dy = np.mean(patch[..., 1])

                    # Computes gaze movement relative to scene movement
                    relative_distance = np.sqrt(
                        (gaze_dx - flow_dx) ** 2 +
                        (gaze_dy - flow_dy) ** 2
                    )

                    # Measures stabilized gaze behavior
                    relative_motion = (relative_distance / dt)

                # ------------------------------------------
                # FIXATION
                # ------------------------------------------
                if (
                    gaze_velocity is not None
                    and relative_motion is not None
                ):

                    # Represents stable environmental fixation
                    is_static_fixation = (
                        gaze_velocity < VEL_THRESHOLD # gaze movement low
                        and relative_motion < FLOW_THRESHOLD # relative motion low
                    )

                    # Represents tracking moving targets
                    is_smooth_pursuit = (
                        gaze_velocity >= VEL_THRESHOLD # gaze velocity high
                        and relative_motion < FLOW_THRESHOLD # relative motion low
                    )

                    # Represents rapid attentional shift
                    is_saccade = (
                        gaze_velocity >= VEL_THRESHOLD
                        and relative_motion >= FLOW_THRESHOLD
                    )

        # ==================================================
        # STORE RESULTS
        # ==================================================
        results.append({

            # ----------------------------------------------
            # IDENTITY
            # ----------------------------------------------
            "participant": participant,
            "session": session_name,

            # ----------------------------------------------
            # TIME
            # ----------------------------------------------
            "frame": frame_idx,
            "timestamp": row["timestamp"],

            # ----------------------------------------------
            # GAZE
            # ----------------------------------------------
            "x": x,
            "y": y,

            # ----------------------------------------------
            # LABELS
            # ----------------------------------------------
            "final_label": final_label,
            "segmentation_label": segmentation_label,
            "tracked_object_label": tracked_object_label,

            # ----------------------------------------------
            # AGREEMENT
            # ----------------------------------------------
            "models_agree": models_agree,
            "conflict_type": conflict_type,
            "source_priority": source_priority,

            # ----------------------------------------------
            # TRACKING
            # ----------------------------------------------
            "track_id": track_id,
            "tracking_confidence": tracking_confidence,

            # ----------------------------------------------
            # PANOPTIC
            # ----------------------------------------------
            "panoptic_segment_id": panoptic_segment_id,

            # ----------------------------------------------
            # DYNAMICS
            # ----------------------------------------------
            "gaze_velocity_px_s": gaze_velocity,
            "relative_motion_px_s": relative_motion,
            "is_static_fixation": is_static_fixation, # fixation
            "is_smooth_pursuit": is_smooth_pursuit, # active pursuit of moving targets
            "is_saccade": is_saccade, # HIGH relative motion

            # ----------------------------------------------
            # CONTEXT
            # ----------------------------------------------
            "frame_context": frame_context,
            "gaze_context": gaze_context,

            # ----------------------------------------------
            # REPRODUCIBILITY
            # ----------------------------------------------
            "patch_radius": PATCH_RADIUS
        })

        # ==================================================
        # UPDATE TEMPORAL STATE
        # ==================================================
        # Stores current gaze for next iteration
        prev_gaze = (x, y)

        # Stores current timestamp for velocity calculations
        prev_timestamp = row["timestamp"]

    # ==================================================
    # UPDATE FRAME STATE
    # ==================================================
    prev_frame_gray = gray
    # To next video frame
    frame_idx += 1

# Cleanup: this closes video stream
cap.release()

# ==================================================
# SAVE
# ==================================================
save_path = os.path.join(
    output_root,
    f"{participant}_{session_name}_fused.csv"
)

pd.DataFrame(results).to_csv(save_path,index=False)
print(f"\nSaved → {save_path}\n")