import streamlit as st
import cv2
import numpy as np
import os
import mediapipe as mp
import onnxruntime as ort
from collections import deque
import time

# --- Subvector Configuration ---
SUBVECTOR_CONFIG = {
    'location_l_hand': 63,
    'location_r_hand': 63,
    'location_pose': 48,      # Upper body only
    'handshape_l': 210,
    'handshape_r': 210,
    'palm_orientation': 200,
    'movement': 126
}
TOTAL_FEATURES = sum(SUBVECTOR_CONFIG.values())  # 920 features

# --- App Information and Setup ---
st.set_page_config(page_title="MSL Recognition App", layout="wide")

st.title("Real-Time Malaysian Sign Language (MSL) Recognition")
st.write("""
This application uses a deep learning Transformer model to recognize dynamic Malaysian Sign Language gestures in real-time. 
Enable your webcam below and start signing!
""")

# --- Sidebar Controls ---
st.sidebar.header("Controls")
show_face_landmarks = st.sidebar.checkbox("Show Face Landmarks", value=True)
show_pose_landmarks = st.sidebar.checkbox("Show Pose/Body Landmarks", value=True)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.4, 1.0, 0.9, 0.05)
# NEW: Hands loss timeout configuration
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔄 Hands Loss Settings")
hands_lost_timeout = st.sidebar.slider("Hands Loss Timeout (frames)", 10, 120, 15, 5)
st.sidebar.markdown(f"*Clears predictions after {hands_lost_timeout/30:.1f}s of no hands*")

# --- Helper Functions for Model Loading ---
def get_available_providers():
    """Get available ONNX Runtime providers with GPU detection."""
    available_providers = ort.get_available_providers()
    
    # Prefer GPU if available
    if 'CUDAExecutionProvider' in available_providers:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        st.success("✅ GPU acceleration enabled for ONNX Runtime")
    else:
        providers = ['CPUExecutionProvider']
        st.info("ℹ️ Using CPU for ONNX Runtime")
    
    return providers

def validate_model(session, expected_inputs=7, expected_outputs=1):
    """Validate model specifications match expectations."""
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    
    if len(inputs) != expected_inputs:
        raise ValueError(f"Expected {expected_inputs} inputs, got {len(inputs)}")
    
    if len(outputs) != expected_outputs:
        raise ValueError(f"Expected {expected_outputs} outputs, got {len(outputs)}")
    
    # Validate input shapes (expect sequence length 30)
    for i, inp in enumerate(inputs):
        if len(inp.shape) != 3:  # [batch_size, sequence_length, features]
            raise ValueError(f"Input {i} has invalid shape: {inp.shape}")
        if inp.shape[1] != 30:  # sequence length should be 30
            raise ValueError(f"Input {i} has invalid sequence length: {inp.shape[1]}")
    
    return True

# --- ONNX Model Loading ---
@st.cache_resource
def load_onnx_model(model_path):
    """Load and validate ONNX model with proper error handling."""
    # Check file existence first
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.error("Please ensure the model file exists in the correct location.")
        return None, None, None
    
    try:
        # Get optimal providers
        providers = get_available_providers()
        
        # Create inference session
        session = ort.InferenceSession(model_path, providers=providers)
        
        # Validate model structure
        validate_model(session, expected_inputs=7, expected_outputs=1)
        
        # Extract input/output info
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        input_names = [inp.name for inp in inputs]
        output_name = outputs[0].name
        
        st.success(f"✅ ONNX Model loaded successfully from {model_path}")
        st.info(f"📊 Model: {len(inputs)} inputs, {len(outputs)} outputs")
        
        return session, input_names, output_name
        
    except FileNotFoundError:
        st.error(f"❌ Model file not found: {model_path}")
        return None, None, None
    except ValueError as ve:
        st.error(f"❌ Model validation failed: {ve}")
        st.error("The model structure doesn't match expected specifications.")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error loading ONNX model: {e}")
        st.error("Please check the model file and try again.")
        return None, None, None

# --- Load Model and Actions ---
#MODEL_PATH = 'best_model_multistream_30.onnx'
# Use absolute paths to avoid directory confusion
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'bigger.onnx')
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

model, input_names, output_name = load_onnx_model(MODEL_PATH)

if model:
    try:
        actions = np.array([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    except FileNotFoundError:
        st.error(f"Data directory '{DATA_PATH}' not found. Cannot determine action labels.")
        actions = np.array(['Gesture_1', 'Gesture_2']) # Placeholder

    # --- MediaPipe Setup ---
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    # --- Helper Functions ---
    def mediapipe_detection(image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_face_mesh_with_hidden_features(image, face_landmarks):
        """Draw face mesh with eyes, nose, and mouth hidden"""
        if not face_landmarks:
            return
            
        # Define which face landmarks to hide
        face_features_to_hide = {
            # Eyes
            33, 7, 163, 144, 362, 398, 384, 381, 382, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,
            # Nose
            1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 237, 238, 239, 240, 241, 242, 243, 190, 221, 55, 65, 52, 53, 46, 124, 35, 234, 127, 162, 21, 54, 103, 67, 109,
            # Mouth
            13, 14, 78, 80, 81, 82, 87, 88, 95, 146, 148, 149, 176, 178, 179, 180, 181, 185, 191, 269, 270, 271, 272, 291, 308, 310, 311, 312, 314, 317, 318, 324, 375, 321, 405, 415, 273, 282, 285, 295, 296, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347
        }
        
        # Get image dimensions
        h, w, _ = image.shape
        
        # Draw only face outline (avoid eyes, nose, mouth areas)
        # Draw selected connections manually to avoid facial features
        face_outline_connections = [
            # Face outline (chin to forehead)
            (172, 136), (136, 150), (150, 149), (149, 176), (176, 148), (148, 152), (152, 377), (377, 400), (400, 378), (378, 379), (379, 365), (365, 397), (397, 288), (288, 361), (361, 323), (323, 454), (454, 323), (323, 361), (361, 340), (340, 346), (346, 347), (347, 348), (348, 349), (349, 350), (350, 451), (451, 452), (452, 453), (453, 464), (464, 172),
            # Jawline
            (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162), (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
            # Forehead outline
            (70, 63), (63, 105), (105, 66), (66, 107), (107, 55), (55, 65), (65, 52), (52, 53),
            # Side face
            (234, 93), (93, 132), (132, 58), (58, 172), (172, 136), (136, 172), (172, 58),
            (288, 361), (361, 323), (323, 340), (340, 346), (346, 347), (347, 348)
        ]
        
        # Draw safe face outline connections
        for start_idx, end_idx in face_outline_connections:
            if start_idx < len(face_landmarks.landmark) and end_idx < len(face_landmarks.landmark):
                if start_idx not in face_features_to_hide and end_idx not in face_features_to_hide:
                    start_pt = face_landmarks.landmark[start_idx]
                    end_pt = face_landmarks.landmark[end_idx]
                    
                    # Convert normalized coordinates to pixel coordinates
                    start_x, start_y = int(start_pt.x * w), int(start_pt.y * h)
                    end_x, end_y = int(end_pt.x * w), int(end_pt.y * h)
                    
                    # Draw connection
                    cv2.line(image, (start_x, start_y), (end_x, end_y), (80, 110, 10), 1)
        
        # Draw only safe landmarks (not eyes, nose, mouth)
        for idx, landmark in enumerate(face_landmarks.landmark):
            if idx not in face_features_to_hide:
                # Convert normalized coordinates to pixel coordinates
                x, y = int(landmark.x * w), int(landmark.y * h)
                # Draw landmark point
                cv2.circle(image, (x, y), 1, (80, 256, 121), -1)

    def draw_pose_landmarks_with_face_control(image, pose_landmarks, show_face_features=True):
        """Draw pose landmarks with facial feature control"""
        if not pose_landmarks:
            return
            
        # Define facial pose landmarks (indices 0-10)
        facial_pose_indices = set(range(11))  # 0-10 inclusive
        
        # Get image dimensions
        h, w, _ = image.shape
        
        # Define pose connections
        pose_connections = mp_holistic.POSE_CONNECTIONS
        
        # Draw connections based on face control setting
        for connection in pose_connections:
            start_idx, end_idx = connection
            
            # Skip connections involving facial landmarks if face features are hidden
            if not show_face_features and (start_idx in facial_pose_indices or end_idx in facial_pose_indices):
                continue
                
            # Check if both landmarks exist
            if start_idx < len(pose_landmarks.landmark) and end_idx < len(pose_landmarks.landmark):
                start_pt = pose_landmarks.landmark[start_idx]
                end_pt = pose_landmarks.landmark[end_idx]
                
                # Convert normalized coordinates to pixel coordinates
                start_x, start_y = int(start_pt.x * w), int(start_pt.y * h)
                end_x, end_y = int(end_pt.x * w), int(end_pt.y * h)
                
                # Draw connection
                cv2.line(image, (start_x, start_y), (end_x, end_y), (80, 44, 121), 2)
        
        # Draw landmarks based on face control setting
        for idx, landmark in enumerate(pose_landmarks.landmark):
            # Skip facial landmarks if face features are hidden
            if not show_face_features and idx in facial_pose_indices:
                continue
                
            # Convert normalized coordinates to pixel coordinates
            x, y = int(landmark.x * w), int(landmark.y * h)
            
            # Draw landmark point
            cv2.circle(image, (x, y), 4, (80, 22, 10), -1)
            cv2.circle(image, (x, y), 2, (80, 44, 121), -1)

    def draw_styled_landmarks(image, results):
        """Draw MediaPipe landmarks on image with proper frame isolation"""
        # Create a copy of the image to prevent drawing on shared buffer
        annotated_image = image.copy()
        
            # Face connections (conditional based on toggle) - hiding eyes, nose, and mouth
        if show_face_landmarks and results.face_landmarks:
            # Draw face mesh with selective landmark hiding
            draw_face_mesh_with_hidden_features(annotated_image, results.face_landmarks)
        
        # Pose connections with face control and pose visibility toggle
        if show_pose_landmarks and results.pose_landmarks:
            draw_pose_landmarks_with_face_control(annotated_image, results.pose_landmarks, show_face_landmarks)
        
        # Hand connections
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
            )
        
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        
        return annotated_image

    def compute_distance(p1, p2):
        """Compute Euclidean distance between two 3D points."""
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def compute_angle(p1, p2, p3):
        """Compute angle at point p2 formed by p1-p2-p3."""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return angle

    def compute_palm_normal(hand_landmarks):
        """Compute palm normal vector."""
        if hand_landmarks is None or len(hand_landmarks) == 0:
            return np.zeros(3)
        
        wrist = hand_landmarks[0]
        index_mcp = hand_landmarks[5]
        pinky_mcp = hand_landmarks[17]
        
        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist
        
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            normal = normal / norm
        
        return normal

    def extract_location_features(pose_landmarks, left_hand_landmarks, right_hand_landmarks):
        """Extract location features for hands and UPPER BODY pose."""
        if left_hand_landmarks is not None:
            location_l_hand = left_hand_landmarks.flatten()
        else:
            location_l_hand = np.zeros(63)
        
        if right_hand_landmarks is not None:
            location_r_hand = right_hand_landmarks.flatten()
        else:
            location_r_hand = np.zeros(63)
        
        if pose_landmarks is not None:
            selected_indices = list(range(11)) + list(range(11, 17))
            location_pose = pose_landmarks[selected_indices].flatten()
        else:
            location_pose = np.zeros(48)
        
        return location_l_hand, location_r_hand, location_pose

    def extract_handshape_features(hand_landmarks):
        """Extract handshape features: angles and distances."""
        if hand_landmarks is None or len(hand_landmarks) == 0:
            return np.zeros(210)
        
        features = []
        
        finger_indices = [
            [0, 1, 2, 3, 4],
            [0, 5, 6, 7, 8],
            [0, 9, 10, 11, 12],
            [0, 13, 14, 15, 16],
            [0, 17, 18, 19, 20]
        ]
        
        for finger in finger_indices:
            for i in range(len(finger) - 2):
                angle = compute_angle(
                    hand_landmarks[finger[i]],
                    hand_landmarks[finger[i+1]],
                    hand_landmarks[finger[i+2]]
                )
                features.append(angle)
        
        fingertips = [4, 8, 12, 16, 20]
        for i in range(len(fingertips)):
            for j in range(i+1, len(fingertips)):
                dist = compute_distance(hand_landmarks[fingertips[i]], hand_landmarks[fingertips[j]])
                features.append(dist)
        
        for i in range(1, 21):
            dist = compute_distance(hand_landmarks[0], hand_landmarks[i])
            features.append(dist)
        
        for finger in finger_indices[1:]:
            for i in range(len(finger) - 1):
                dist = compute_distance(hand_landmarks[finger[i]], hand_landmarks[finger[i+1]])
                features.append(dist)
        
        mcps = [5, 9, 13, 17]
        for i in range(len(mcps)):
            for j in range(i+1, len(mcps)):
                dist = compute_distance(hand_landmarks[mcps[i]], hand_landmarks[mcps[j]])
                features.append(dist)
        
        features = np.array(features)
        if len(features) < 210:
            features = np.pad(features, (0, 210 - len(features)), 'constant')
        else:
            features = features[:210]
        
        return features

    def extract_palm_orientation_features(left_hand_landmarks, right_hand_landmarks):
        """Extract palm orientation features."""
        features = []
        
        for hand_landmarks in [left_hand_landmarks, right_hand_landmarks]:
            if hand_landmarks is None or len(hand_landmarks) == 0:
                features.extend(np.zeros(100))
                continue
            
            hand_features = []
            
            palm_normal = compute_palm_normal(hand_landmarks)
            hand_features.extend(palm_normal)
            
            if len(hand_landmarks) > 9:
                wrist_to_middle = hand_landmarks[9] - hand_landmarks[0]
                wrist_to_middle = wrist_to_middle / (np.linalg.norm(wrist_to_middle) + 1e-6)
                hand_features.extend(wrist_to_middle)
            else:
                hand_features.extend(np.zeros(3))
            
            fingertips = [4, 8, 12, 16, 20]
            mcps = [2, 5, 9, 13, 17]
            
            for tip_idx, mcp_idx in zip(fingertips, mcps):
                if len(hand_landmarks) > tip_idx:
                    finger_vec = hand_landmarks[tip_idx] - hand_landmarks[mcp_idx]
                    finger_vec = finger_vec / (np.linalg.norm(finger_vec) + 1e-6)
                    hand_features.extend(finger_vec)
                else:
                    hand_features.extend(np.zeros(3))
            
            for tip_idx, mcp_idx in zip(fingertips, mcps):
                if len(hand_landmarks) > tip_idx:
                    finger_vec = hand_landmarks[tip_idx] - hand_landmarks[mcp_idx]
                    finger_vec = finger_vec / (np.linalg.norm(finger_vec) + 1e-6)
                    cos_angle = np.dot(palm_normal, finger_vec)
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                    hand_features.append(angle)
                else:
                    hand_features.append(0.0)
            
            for i in range(len(fingertips) - 1):
                if len(hand_landmarks) > fingertips[i+1]:
                    v1 = hand_landmarks[fingertips[i]] - hand_landmarks[0]
                    v2 = hand_landmarks[fingertips[i+1]] - hand_landmarks[0]
                    cross = np.cross(v1, v2)
                    cross = cross / (np.linalg.norm(cross) + 1e-6)
                    hand_features.extend(cross)
                else:
                    hand_features.extend(np.zeros(3))
            
            hand_features = np.array(hand_features)
            if len(hand_features) < 100:
                hand_features = np.pad(hand_features, (0, 100 - len(hand_features)), 'constant')
            else:
                hand_features = hand_features[:100]
            
            features.extend(hand_features)
        
        return np.array(features)

    def extract_movement_features(current_frame_data, previous_frame_data):
        """Extract movement features: velocity."""
        if previous_frame_data is None:
            return np.zeros(126)
        
        curr_lh = current_frame_data['left_hand']
        curr_rh = current_frame_data['right_hand']
        
        prev_lh = previous_frame_data['left_hand']
        prev_rh = previous_frame_data['right_hand']
        
        if curr_lh is not None and prev_lh is not None:
            velocity_lh = (curr_lh - prev_lh).flatten()
        else:
            velocity_lh = np.zeros(63)
        
        if curr_rh is not None and prev_rh is not None:
            velocity_rh = (curr_rh - prev_rh).flatten()
        else:
            velocity_rh = np.zeros(63)
        
        return np.concatenate([velocity_lh, velocity_rh])

    def are_hands_detected(results):
        """Check if at least one hand is detected with sufficient landmarks."""
        left_hand_detected = results.left_hand_landmarks is not None and len(results.left_hand_landmarks.landmark) > 0
        right_hand_detected = results.right_hand_landmarks is not None and len(results.right_hand_landmarks.landmark) > 0
        return left_hand_detected or right_hand_detected

    # NEW: Function to cleanup incomplete predictions
    def cleanup_incomplete_sequences():
        """Clear incomplete prediction sequences when hands are lost for timeout period."""
        if len(st.session_state.prediction_sequence) > 0:
            cleared_count = len(st.session_state.prediction_sequence)
            st.session_state.prediction_sequence = []
            st.session_state.sequence_counter = 0
            print(f"[CLEANUP] Cleared {cleared_count} incomplete predictions due to hands loss timeout")
            return True
        return False

    def extract_all_subvectors(results, previous_frame_data=None):
        """Extract all 7 subvectors from MediaPipe results."""
        if results.pose_landmarks:
            pose_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        else:
            pose_landmarks = None
        
        if results.left_hand_landmarks:
            left_hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
        else:
            left_hand_landmarks = None
        
        if results.right_hand_landmarks:
            right_hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
        else:
            right_hand_landmarks = None
        
        raw_data = {
            'pose': pose_landmarks,
            'left_hand': left_hand_landmarks,
            'right_hand': right_hand_landmarks
        }
        
        location_l_hand, location_r_hand, location_pose = extract_location_features(
            pose_landmarks, left_hand_landmarks, right_hand_landmarks
        )
        
        handshape_l = extract_handshape_features(left_hand_landmarks)
        handshape_r = extract_handshape_features(right_hand_landmarks)
        
        palm_orientation = extract_palm_orientation_features(left_hand_landmarks, right_hand_landmarks)
        
        movement = extract_movement_features(raw_data, previous_frame_data)
        
        combined_features = np.concatenate([
            location_l_hand,
            location_r_hand,
            location_pose,
            handshape_l,
            handshape_r,
            palm_orientation,
            movement
        ])
        
        return combined_features, raw_data

    def split_into_subvectors(sequence):
        """Split combined feature sequence into 7 subvectors for model input."""
        subvectors = []
        start_idx = 0
        
        for name, size in SUBVECTOR_CONFIG.items():
            end_idx = start_idx + size
            subvector = sequence[:, start_idx:end_idx]
            subvectors.append(subvector)
            start_idx = end_idx
        
        return subvectors

    def predict_onnx(sequence_array):
        """Make prediction using ONNX Runtime."""
        try:
            # Split into 7 subvectors
            subvectors = split_into_subvectors(sequence_array)
            
            # Add batch dimension to each subvector and convert to float32
            subvectors_batched = [np.expand_dims(sv.astype(np.float32), axis=0) for sv in subvectors]
            
            # Create input dictionary for ONNX Runtime
            input_dict = {name: data for name, data in zip(input_names, subvectors_batched)}
            
            # Run inference with ONNX Runtime
            predictions = model.run([output_name], input_dict)[0]
            
            # Get prediction and confidence
            predictions = predictions[0]  # Remove batch dimension
            prediction_idx = np.argmax(predictions)
            confidence = predictions[prediction_idx]
            
            if 0 <= prediction_idx < len(actions):
                pred_label = actions[prediction_idx]
                return pred_label, confidence
            else:
                return "Invalid prediction", 0.0
                
        except Exception as e:
            st.error(f"[ERROR] ONNX Prediction error: {e}")
            return "Error", 0.0

    def add_to_prediction_sequence(prediction, confidence):
        """Add prediction to sequence buffer only if confidence meets threshold."""
        # Only add prediction if confidence meets the threshold
        if confidence >= confidence_threshold:
            st.session_state.prediction_sequence.append({
                'prediction': prediction,
                'confidence': confidence
            })
            st.session_state.sequence_counter += 1
            
            # Log high-confidence prediction
            print(f"[HIGH CONF] {prediction} ({confidence:.2%}) added to sequence")
        else:
            # Log filtered low-confidence prediction for debugging
            print(f"[FILTERED] Low confidence prediction: {prediction} ({confidence:.2%})")
        
        # Check if we have enough predictions for sequence voting (minimum 8)
        if len(st.session_state.prediction_sequence) >= 10:
            process_sequence_vote()
            # Clear sequence buffer for next round
            st.session_state.prediction_sequence = []
            st.session_state.sequence_counter = 0

    def process_sequence_vote():
        """Process majority voting for filtered predictions and determine final sequence."""
        if len(st.session_state.prediction_sequence) < 10:
            return
        
        # Count predictions
        prediction_counts = {}
        confidence_sums = {}
        
        for pred_data in st.session_state.prediction_sequence:
            pred = pred_data['prediction']
            conf = pred_data['confidence']
            
            if pred not in prediction_counts:
                prediction_counts[pred] = 0
                confidence_sums[pred] = 0
            
            prediction_counts[pred] += 1
            confidence_sums[pred] += conf
        
        # Find majority prediction
        majority_prediction = max(prediction_counts, key=prediction_counts.get)
        vote_count = prediction_counts[majority_prediction]
        avg_confidence = confidence_sums[majority_prediction] / vote_count
        total_predictions = len(st.session_state.prediction_sequence)
        
        # Store the final sequence result
        sequence_result = {
            'sequence': st.session_state.prediction_sequence.copy(),
            'final_prediction': majority_prediction,
            'vote_count': vote_count,
            'total_predictions': total_predictions,
            'average_confidence': avg_confidence,
            'timestamp': time.time()
        }
        
        st.session_state.final_sequences.append(sequence_result)
        
        # Limit total sequences to 4 for better display
        if len(st.session_state.final_sequences) > 4:
            st.session_state.final_sequences = st.session_state.final_sequences[-4:]
        
        # Print sequence result with updated info
        print(f"[SEQUENCE RESULT] {majority_prediction} ({vote_count}/{total_predictions} votes, {avg_confidence:.2%} avg confidence)")
        
        return sequence_result

    def deduplicate_sequences():
        """Remove consecutive duplicate predictions from final sequences."""
        if len(st.session_state.final_sequences) == 0:
            return []
        
        unique_sequences = []
        prev_prediction = None
        for seq in st.session_state.final_sequences:
            if seq['final_prediction'] != prev_prediction:
                unique_sequences.append(seq)
                prev_prediction = seq['final_prediction']
        
        return unique_sequences

    # --- Streamlit UI and Real-Time Logic ---
    col1, col2 = st.columns([2, 1])

    with col1:
        run = st.checkbox('Start Webcam', value=False)
        FRAME_WINDOW = st.image([])

    with col2:
        sequence_text = st.empty()

    # --- Main Application Loop ---
    if run:
        cap = cv2.VideoCapture(0)
        
        # Initialize session state variables
        if 'sequence' not in st.session_state:
            st.session_state.sequence = []
        if 'final_sequences' not in st.session_state:
            st.session_state.final_sequences = []
        if 'prediction_sequence' not in st.session_state:
            st.session_state.prediction_sequence = []
        if 'previous_frame_data' not in st.session_state:
            st.session_state.previous_frame_data = None
        if 'frame_count' not in st.session_state:
            st.session_state.frame_count = 0
        if 'sequence_counter' not in st.session_state:
            st.session_state.sequence_counter = 0
        # NEW: Initialize hands loss tracking variables
        if 'hands_lost_counter' not in st.session_state:
            st.session_state.hands_lost_counter = 0
        if 'hands_lost_threshold' not in st.session_state:
            st.session_state.hands_lost_threshold = hands_lost_timeout

        sequence_length = 30  # Updated for multistream_30 model

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened() and run:
                ret, frame = cap.read()
                if not ret:
                    st.write("The video capture has ended.")
                    break

                st.session_state.frame_count += 1
                
                image, results = mediapipe_detection(frame, holistic)
                image = draw_styled_landmarks(image, results)
                
                # Prediction logic with ONNX
                current_status = ""
                current_confidence = 0.0
                
                # Check if hands are detected
                hands_detected = are_hands_detected(results)
                
                # NEW: Hands loss timeout logic
                if hands_detected:
                    # Reset counter when hands are detected
                    st.session_state.hands_lost_counter = 0
                else:
                    # Increment counter when no hands detected
                    st.session_state.hands_lost_counter += 1
                    
                    # Check if timeout threshold is reached
                    if st.session_state.hands_lost_counter >= st.session_state.hands_lost_threshold:
                        # Cleanup incomplete sequences
                        cleanup_performed = cleanup_incomplete_sequences()
                        if cleanup_performed:
                            current_status = f"🧹 Cleared incomplete predictions (no hands for {hands_lost_timeout/30:.1f}s)"
                
                # Extract features every frame
                combined_features, raw_data = extract_all_subvectors(
                    results, 
                    st.session_state.previous_frame_data
                )
                
                st.session_state.sequence.append(combined_features)
                st.session_state.sequence = st.session_state.sequence[-sequence_length:]
                st.session_state.previous_frame_data = raw_data

                if len(st.session_state.sequence) == sequence_length and st.session_state.frame_count % 1 == 0 and hands_detected:
                    # Convert sequence to numpy array with proper shape
                    sequence_array = np.array(st.session_state.sequence)
                    if sequence_array.shape != (sequence_length, TOTAL_FEATURES):
                        # Ensure we have the correct shape, handle edge cases
                        sequence_array = sequence_array.reshape(sequence_length, -1)
                    
                    # Make prediction with ONNX
                    current_prediction, current_confidence = predict_onnx(sequence_array)
                    
                    # Add to prediction sequence for majority voting (no frame-by-frame display)
                    add_to_prediction_sequence(current_prediction, current_confidence)
                else:
                    if len(st.session_state.sequence) < sequence_length and not current_status:
                        current_status = f"Buffering {len(st.session_state.sequence)}/{sequence_length}"
                    elif not hands_detected and not current_status:
                        current_status = "No hands detected"

                # Deduplicate sequences for clean display
                unique_sequences = deduplicate_sequences()
                
                # Update UI elements - Only show final sequences, not frame-by-frame predictions
                if unique_sequences:
                    sequence_words = [seq['final_prediction'] for seq in unique_sequences]
                    sequences_text = " ".join(sequence_words)
                    sequence_text.markdown(f"# `{sequences_text}`")
                else:
                    sequence_text.markdown("# *Waiting for clear signs...*")
                
                # Display the frame
                FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            cap.release()
            cv2.destroyAllWindows()
    else:
        st.warning("Webcam is turned off.")
else:
    st.error("Model could not be loaded. The application cannot start.")
