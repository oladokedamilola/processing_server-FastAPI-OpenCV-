"""
Advanced detection features: crowd detection, vehicle counting, etc.
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

from ..core.config import settings
from ..utils.logger import logger
from .detection import Detection, DetectionType

@dataclass
class CrowdConfig:
    """Configuration for crowd detection"""
    density_threshold: float = 0.3  # Minimum density to consider as crowd
    min_people_count: int = 3  # Minimum number of people to consider as crowd
    proximity_threshold: float = 50.0  # Max distance between people to consider as group
    group_time_window: float = 2.0  # Time window to maintain group tracking (seconds)

class CrowdDetector:
    """Crowd detection and people counting"""
    
    def __init__(self, config: CrowdConfig = None):
        self.config = config or CrowdConfig()
        self.tracked_groups = {}  # group_id -> group_data
        self.last_update_time = time.time()
        
    def detect_crowd(
        self, 
        person_detections: List[Detection],
        image_shape: Tuple[int, int]
    ) -> List[Detection]:
        """Detect crowds from person detections"""
        if len(person_detections) < self.config.min_people_count:
            return []
        
        try:
            # Calculate image area
            image_area = image_shape[0] * image_shape[1]
            
            # Group people by proximity
            groups = self._group_by_proximity(person_detections)
            
            crowd_detections = []
            
            for group_id, group_members in enumerate(groups):
                if len(group_members) >= self.config.min_people_count:
                    # Calculate group bounding box
                    group_bbox = self._calculate_group_bbox(group_members)
                    
                    # Calculate group area
                    group_area = (group_bbox[2] - group_bbox[0]) * (group_bbox[3] - group_bbox[1])
                    
                    # Calculate density
                    density = len(group_members) / (group_area / 1000)  # People per 1000 pixels
                    
                    # Calculate normalized density (0-1)
                    normalized_density = min(density / 10.0, 1.0)  # Assuming max 10 people per 1000 pixels
                    
                    # Calculate confidence based on group size and density
                    confidence = min(
                        0.3 + (len(group_members) / 10) * 0.3 + normalized_density * 0.4,
                        1.0
                    )
                    
                    if normalized_density >= self.config.density_threshold:
                        # Create crowd detection
                        crowd_detection = Detection(
                            label="crowd",
                            confidence=confidence,
                            bbox=group_bbox,
                            detection_type=DetectionType.CROWD,
                            attributes={
                                "people_count": len(group_members),
                                "density": round(normalized_density, 3),
                                "group_id": group_id,
                                "group_area": group_area,
                                "members": [
                                    {
                                        "bbox": member.bbox,
                                        "confidence": member.confidence
                                    }
                                    for member in group_members
                                ]
                            }
                        )
                        
                        crowd_detections.append(crowd_detection)
                        
                        logger.debug(f"Crowd detected: {len(group_members)} people, density: {normalized_density:.3f}")
            
            # Update group tracking
            self._update_group_tracking(crowd_detections)
            
            return crowd_detections
            
        except Exception as e:
            logger.error(f"Error in crowd detection: {str(e)}")
            return []
    
    def _group_by_proximity(self, detections: List[Detection]) -> List[List[Detection]]:
        """Group detections by spatial proximity"""
        if not detections:
            return []
        
        groups = []
        assigned = set()
        
        for i, det1 in enumerate(detections):
            if i in assigned:
                continue
            
            # Start a new group
            group = [det1]
            assigned.add(i)
            
            # Find nearby detections
            for j, det2 in enumerate(detections):
                if j in assigned or i == j:
                    continue
                
                # Calculate distance between centers
                center1 = det1.attributes.get("center", {"x": 0, "y": 0})
                center2 = det2.attributes.get("center", {"x": 0, "y": 0})
                
                distance = np.sqrt(
                    (center1["x"] - center2["x"]) ** 2 + 
                    (center1["y"] - center2["y"]) ** 2
                )
                
                if distance <= self.config.proximity_threshold:
                    group.append(det2)
                    assigned.add(j)
            
            groups.append(group)
        
        return groups
    
    def _calculate_group_bbox(self, detections: List[Detection]) -> List[int]:
        """Calculate bounding box that contains all detections in group"""
        if not detections:
            return [0, 0, 0, 0]
        
        # Initialize with first detection
        first_bbox = detections[0].bbox
        min_x, min_y, max_x, max_y = first_bbox
        
        # Expand to include all detections
        for detection in detections[1:]:
            x1, y1, x2, y2 = detection.bbox
            min_x = min(min_x, x1)
            min_y = min(min_y, y1)
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)
        
        # Add padding
        padding = 20
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = max_x + padding
        max_y = max_y + padding
        
        return [min_x, min_y, max_x, max_y]
    
    def _update_group_tracking(self, crowd_detections: List[Detection]):
        """Update tracking of crowd groups over time"""
        current_time = time.time()
        time_diff = current_time - self.last_update_time
        
        # Remove old groups
        groups_to_remove = []
        for group_id, group_data in self.tracked_groups.items():
            if current_time - group_data["last_seen"] > self.config.group_time_window:
                groups_to_remove.append(group_id)
        
        for group_id in groups_to_remove:
            del self.tracked_groups[group_id]
        
        # Update or add new groups
        for crowd in crowd_detections:
            group_id = crowd.attributes.get("group_id")
            
            # Try to match with existing group
            matched = False
            for existing_id, existing_data in self.tracked_groups.items():
                # Calculate overlap between current and previous bbox
                bbox1 = crowd.bbox
                bbox2 = existing_data["last_bbox"]
                
                overlap = self._calculate_iou(bbox1, bbox2)
                
                if overlap > 0.3:  # 30% overlap to consider as same group
                    # Update existing group
                    self.tracked_groups[existing_id] = {
                        "last_bbox": crowd.bbox,
                        "last_seen": current_time,
                        "people_count": crowd.attributes.get("people_count", 0),
                        "persistence": existing_data.get("persistence", 0) + time_diff
                    }
                    matched = True
                    break
            
            if not matched:
                # Add new group
                self.tracked_groups[group_id] = {
                    "last_bbox": crowd.bbox,
                    "last_seen": current_time,
                    "people_count": crowd.attributes.get("people_count", 0),
                    "persistence": 0.0
                }
        
        self.last_update_time = current_time
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def get_crowd_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected crowds"""
        current_time = time.time()
        active_crowds = []
        
        for group_id, group_data in self.tracked_groups.items():
            if current_time - group_data["last_seen"] <= self.config.group_time_window:
                active_crowds.append({
                    "group_id": group_id,
                    "people_count": group_data["people_count"],
                    "persistence": group_data.get("persistence", 0),
                    "time_since_last_seen": current_time - group_data["last_seen"]
                })
        
        total_people = sum(crowd["people_count"] for crowd in active_crowds)
        
        return {
            "active_crowds": len(active_crowds),
            "total_people_in_crowds": total_people,
            "average_crowd_size": total_people / len(active_crowds) if active_crowds else 0,
            "crowd_details": active_crowds
        }

@dataclass
class VehicleCounterConfig:
    """Configuration for vehicle counting"""
    counting_line_position: float = 0.5  # Normalized position of counting line (0-1)
    direction_threshold: float = 10.0  # Minimum movement to count as crossing
    track_history_length: int = 10  # Number of frames to track for each vehicle

class VehicleCounter:
    """Count vehicles crossing a virtual line"""
    
    def __init__(self, config: VehicleCounterConfig = None):
        self.config = config or VehicleCounterConfig()
        self.vehicle_tracks = {}  # track_id -> track_history
        self.next_track_id = 0
        self.counts = {
            "left_to_right": 0,
            "right_to_left": 0,
            "total": 0
        }
        
    def count_vehicles(
        self,
        vehicle_detections: List[Detection],
        frame_number: int,
        image_width: int
    ) -> Dict[str, Any]:
        """Count vehicles crossing the counting line"""
        try:
            # Calculate counting line position
            line_x = int(image_width * self.config.counting_line_position)
            
            # Update tracks
            self._update_tracks(vehicle_detections, frame_number)
            
            # Check for crossings
            new_crossings = self._check_crossings(line_x, image_width)
            
            # Update counts
            for crossing in new_crossings:
                direction = crossing["direction"]
                self.counts[direction] += 1
                self.counts["total"] += 1
                
                logger.debug(f"Vehicle counted: {direction} (track_id: {crossing['track_id']})")
            
            # Clean up old tracks
            self._cleanup_old_tracks(frame_number)
            
            return {
                "counts": self.counts.copy(),
                "current_vehicles": len(self.vehicle_tracks),
                "new_crossings": new_crossings,
                "counting_line": line_x
            }
            
        except Exception as e:
            logger.error(f"Error in vehicle counting: {str(e)}")
            return {
                "counts": self.counts.copy(),
                "current_vehicles": 0,
                "new_crossings": [],
                "counting_line": 0
            }
    
    def _update_tracks(self, detections: List[Detection], frame_number: int):
        """Update vehicle tracks with new detections"""
        # First, try to match detections with existing tracks
        matched_detections = set()
        matched_tracks = set()
        
        for track_id, track_data in self.vehicle_tracks.items():
            if not track_data["positions"]:
                continue
            
            last_position = track_data["positions"][-1]
            best_match = None
            best_distance = float('inf')
            
            for i, detection in enumerate(detections):
                if i in matched_detections:
                    continue
                
                center = detection.attributes.get("center", {"x": 0, "y": 0})
                distance = abs(center["x"] - last_position["x"])
                
                if distance < best_distance and distance < 100:  # Max matching distance
                    best_distance = distance
                    best_match = i
            
            if best_match is not None:
                # Update track
                detection = detections[best_match]
                center = detection.attributes.get("center", {"x": 0, "y": 0})
                
                self.vehicle_tracks[track_id]["positions"].append({
                    "x": center["x"],
                    "y": center["y"],
                    "frame": frame_number,
                    "bbox": detection.bbox
                })
                
                # Keep only recent history
                if len(self.vehicle_tracks[track_id]["positions"]) > self.config.track_history_length:
                    self.vehicle_tracks[track_id]["positions"] = self.vehicle_tracks[track_id]["positions"][-self.config.track_history_length:]
                
                self.vehicle_tracks[track_id]["last_update"] = frame_number
                self.vehicle_tracks[track_id]["detection_count"] += 1
                
                matched_detections.add(best_match)
                matched_tracks.add(track_id)
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detections:
                center = detection.attributes.get("center", {"x": 0, "y": 0})
                
                self.vehicle_tracks[self.next_track_id] = {
                    "positions": [{
                        "x": center["x"],
                        "y": center["y"],
                        "frame": frame_number,
                        "bbox": detection.bbox
                    }],
                    "last_update": frame_number,
                    "detection_count": 1,
                    "counted": False
                }
                
                self.next_track_id += 1
    
    def _check_crossings(self, line_x: int, image_width: int) -> List[Dict[str, Any]]:
        """Check for vehicles crossing the counting line"""
        crossings = []
        
        for track_id, track_data in self.vehicle_tracks.items():
            if track_data.get("counted", False) or len(track_data["positions"]) < 2:
                continue
            
            positions = track_data["positions"]
            
            # Check if vehicle crossed the line
            first_pos = positions[0]["x"]
            last_pos = positions[-1]["x"]
            
            # Determine direction
            if first_pos < line_x < last_pos:
                # Left to right
                direction = "left_to_right"
                crossed = True
            elif last_pos < line_x < first_pos:
                # Right to left
                direction = "right_to_left"
                crossed = True
            else:
                crossed = False
            
            if crossed and abs(last_pos - first_pos) >= self.config.direction_threshold:
                # Mark as counted
                self.vehicle_tracks[track_id]["counted"] = True
                
                crossings.append({
                    "track_id": track_id,
                    "direction": direction,
                    "start_position": first_pos,
                    "end_position": last_pos,
                    "frames_tracked": len(positions)
                })
        
        return crossings
    
    def _cleanup_old_tracks(self, current_frame: int, max_age: int = 30):
        """Remove tracks that haven't been updated recently"""
        tracks_to_remove = []
        
        for track_id, track_data in self.vehicle_tracks.items():
            if current_frame - track_data["last_update"] > max_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.vehicle_tracks[track_id]
    
    def reset_counts(self):
        """Reset all counts"""
        self.counts = {
            "left_to_right": 0,
            "right_to_left": 0,
            "total": 0
        }
        logger.info("Vehicle counts reset")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vehicle counting statistics"""
        return {
            "counts": self.counts,
            "active_tracks": len(self.vehicle_tracks),
            "next_track_id": self.next_track_id
        }