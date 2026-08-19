"""
tactical_logic.py
=================
Tactical Decision Logic & Spatial Non-Maximum Suppression (NMS).

Applies spatial filtering and NMS thresholding over C++ anomaly heatmaps to discard
transient false positives (falling debris, atmospheric smoke bursts, camera shake)
and isolate high-confidence battlefield threat detections.
"""

import cv2
import numpy as np


class TacticalDecisionLogic:
    """
    Filters spatial MSE anomaly heatmaps, runs contour clustering, and applies NMS.
    """
    def __init__(self, mse_threshold=0.06, min_area=150, max_area=15000, nms_iou_threshold=0.3):
        self.mse_threshold = mse_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.nms_iou_threshold = nms_iou_threshold

    def set_mse_threshold(self, th):
        self.mse_threshold = th

    def compute_iou(self, boxA, boxB):
        """Calculates Intersection over Union (IoU) between two bounding boxes [x, y, w, h]."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def apply_nms(self, boxes, scores):
        """Performs Non-Maximum Suppression (NMS) over bounding boxes."""
        if len(boxes) == 0:
            return [], []

        boxes_arr = np.array(boxes)
        scores_arr = np.array(scores)

        # Sort boxes by anomaly score descending
        idxs = np.argsort(scores_arr)[::-1]
        keep = []

        while len(idxs) > 0:
            current = idxs[0]
            keep.append(current)

            if len(idxs) == 1:
                break

            rest = idxs[1:]
            filtered_idxs = []
            for i in rest:
                iou = self.compute_iou(boxes_arr[current], boxes_arr[i])
                if iou < self.nms_iou_threshold:
                    filtered_idxs.append(i)
            idxs = np.array(filtered_idxs)

        keep_boxes = [boxes[i] for i in keep]
        keep_scores = [scores[i] for i in keep]
        return keep_boxes, keep_scores

    def detect_tactical_threats(self, fused_heatmap, motion_map=None, thermal_map=None):
        """
        Processes multi-modal fused anomaly matrix and returns consolidated threat alerts.
        
        Args:
            fused_heatmap (np.ndarray): Pixel-wise fused anomaly matrix (float32, 0.0 - 1.0)
            motion_map (np.ndarray, optional): Normalized motion vector heatmap
            thermal_map (np.ndarray, optional): Normalized thermal intensity heatmap
            
        Returns:
            threats (list): List of dicts [{'id': int, 'bbox': [x, y, w, h], 'score': float, 'type': str, 'confidence': float}]
            cleaned_mask (np.ndarray): Binary thresholded anomaly mask
        """
        if fused_heatmap is None or fused_heatmap.size == 0:
            return [], None

        # 1. Adaptive / Absolute Thresholding
        binary_mask = (fused_heatmap > self.mse_threshold).astype(np.uint8) * 255

        # 2. Morphological filtering to discard transient visual noise (smoke/dust bursts)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        # 3. Contour Detection & Candidate Extraction
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidate_boxes = []
        candidate_scores = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Compute localized mean anomaly intensity score
                roi_fused = fused_heatmap[y:y+h, x:x+w]
                score = float(np.mean(roi_fused)) if roi_fused.size > 0 else float(self.mse_threshold)
                
                candidate_boxes.append([x, y, w, h])
                candidate_scores.append(score)

        # 4. Apply Spatial Non-Maximum Suppression (NMS)
        nms_boxes, nms_scores = self.apply_nms(candidate_boxes, candidate_scores)

        # 5. Multi-Modal Threat Classification & Alert Formatting
        threats = []
        for i, (box, score) in enumerate(zip(nms_boxes, nms_scores)):
            x, y, w, h = box
            area = w * h
            
            # Extract localized Motion & Thermal metrics if available
            motion_score = 0.0
            if motion_map is not None and motion_map.size > 0:
                roi_m = motion_map[y:y+h, x:x+w]
                if roi_m.size > 0:
                    motion_score = float(np.mean(roi_m))

            thermal_score = 0.0
            if thermal_map is not None and thermal_map.size > 0:
                roi_t = thermal_map[y:y+h, x:x+w]
                if roi_t.size > 0:
                    thermal_score = float(np.mean(roi_t))

            # Multi-Modal Decision Matrix
            has_high_motion = motion_score > 0.15
            has_high_thermal = thermal_score > 0.25

            if area > 1000 and has_high_motion and has_high_thermal:
                classification = "HOSTILE ARMORED VEHICLE (MOVING THERMAL)"
            elif area > 1000 and has_high_thermal:
                classification = "STATIC ARMORED ASSET / THERMAL ENGINE"
            elif has_high_motion and has_high_thermal:
                classification = "MOVING INFANTRY / PERSONNEL (THERMAL)"
            elif has_high_thermal and not has_high_motion:
                classification = "STATIC THERMAL HOTSPOT (EXHAUST / HAZARD)"
            elif has_high_motion:
                classification = "CAMOUFLAGED MOTION ANOMALY"
            elif area > 800:
                classification = "LARGE TERRAIN DISRUPTION / STRUCTURE"
            else:
                classification = "DISTURBED GROUND / IMMINENT LANDMINE"

            threats.append({
                'id': i + 1,
                'bbox': [x, y, w, h],
                'score': score,
                'type': classification,
                'motion_score': motion_score,
                'thermal_score': thermal_score,
                'confidence': min(1.0, score / (self.mse_threshold * 2.5))
            })

        return threats, cleaned_mask
