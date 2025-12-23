import streamlit as st
from streamlit_autorefresh import st_autorefresh
import cv2
import numpy as np
import os
import mediapipe as mp
import onnxruntime as ort
from collections import deque
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av

# --- Simplified Global State ---
class GlobalPredictionState:
    def __init__(self):
        self.prediction = "Waiting..."
        self.confidence = 0.0
        self.sentence = []
        self.sequence = []
        self.previous_frame_data = None
        self.prediction_sequence = []
        self.final_sequences = []
        self.hands_lost_counter = 0
        self.hands_lost_threshold = 15  # 15 frames timeout
    
    def update(self, prediction, confidence, sentence):
        self.prediction = prediction
        self.confidence = float(confidence)
        self.sentence = sentence.copy()
    
    def get_state(self):
        return {
            'prediction': self.prediction,
            'confidence': self.confidence,
            'sentence': self.sentence.copy(),
            'sequence': list(self.sequence),
            'final_sequences': list(self.final_sequences),
            'prediction_sequence': list(self.prediction_sequence)
        }
    
    def append_to_sequence(self, features, max_length=30):
        self.sequence.append(features)
        self.sequence = self.sequence[-max_length:]
    
    def get_sequence(self):
        return list(self.sequence)
    
    def set_previous_frame_data(self, data):
        self.previous_frame_data = data
    
    def get_previous_frame_data(self):
        return self.previous_frame_data
    
    def add_final_sequence(self, sequence_result):
        self.final_sequences.append(sequence_result)
        if len(self.final_sequences) > 4:
            self.final_sequences = self.final_sequences[-4:]
    
    def get_final_sequences(self):
        return list(self.final_sequences)
    
    def add_to_prediction_sequence(self, prediction_data):
        self.prediction_sequence.append(prediction_data)
    
    def get_prediction_sequence(self):
        return list(self.prediction_sequence)
    
    def clear_prediction_sequence(self):
        self.prediction_sequence = []
    
    def clear_all(self):
        self.prediction = "Waiting..."
        self.confidence = 0.0
        self.sentence = []
        self.sequence = []
        self.previous_frame_data = None
        self.prediction_sequence = []
        self.final_sequences = []

# --- Subvector Configuration ---
SUBVECTOR_CONFIG = {
    'location_l_hand': 63,
    'location_r_hand': 63,
    'location_pose': 48,
    'handshape_l': 210,
    'handshape_r': 210,
    'palm_orientation': 200,
    'movement': 126
}
TOTAL_FEATURES = sum(SUBVECTOR_CONFIG.values())

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
#confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.4, 1.0, 0.9, 0.05)
confidence_threshold = 0.9  # Fixed confidence threshold at 90%

# --- Helper Functions for Model Loading ---
def get_available_providers():
    available_providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in available_providers:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        st.success("✅ GPU acceleration enabled for ONNX Runtime")
    else:
        providers = ['CPUExecutionProvider']
        st.info("ℹ️ Using CPU for ONNX Runtime")
    return providers

def validate_model(session, expected_inputs=7, expected_outputs=1):
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != expected_inputs:
        raise ValueError(f"Expected {expected_inputs} inputs, got {len(inputs)}")
    if len(outputs) != expected_outputs:
        raise ValueError(f"Expected {expected_outputs} outputs, got {len(outputs)}")
    for i, inp in enumerate(inputs):
        if len(inp.shape) != 3:
            raise ValueError(f"Input {i} has invalid shape: {inp.shape}")
        if inp.shape[1] != 30:
            raise ValueError(f"Input {i} has invalid sequence length: {inp.shape[1]}")
    return True

@st.cache_resource
def load_onnx_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        return None, None, None
    try:
        providers = get_available_providers()
        session = ort.InferenceSession(model_path, providers=providers)
        validate_model(session, expected_inputs=7, expected_outputs=1)
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        input_names = [inp.name for inp in inputs]
        output_name = outputs[0].name
        # st.success(f"✅ ONNX Model loaded successfully from {model_path}")
        # st.info(f"📊 Model: {len(inputs)} inputs, {len(outputs)} outputs")
        return session, input_names, output_name
    except Exception as e:
        st.error(f"❌ Error loading ONNX model: {e}")
        return None, None, None

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'bigger.onnx')
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

model, input_names, output_name = load_onnx_model(MODEL_PATH)

# Initialize global state
if 'global_state' not in st.session_state:
    st.session_state.global_state = GlobalPredictionState()

global_state = st.session_state.global_state

if model:
    try:
        actions = np.array([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    except FileNotFoundError:
        st.error(f"Data directory '{DATA_PATH}' not found.")
        actions = np.array(['Gesture_1', 'Gesture_2'])

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    def mediapipe_detection(image, model):
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        return image, results

    def draw_face_mesh_with_hidden_features(image, face_landmarks):
        if not face_landmarks:
            return
        face_features_to_hide = {33, 7, 163, 144, 362, 398, 384, 381, 382, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 237, 238, 239, 240, 241, 242, 243, 190, 221, 55, 65, 52, 53, 46, 124, 35, 234, 127, 162, 21, 54, 103, 67, 109, 13, 14, 78, 80, 81, 82, 87, 88, 95, 146, 148, 149, 176, 178, 179, 180, 181, 185, 191, 269, 270, 271, 272, 291, 308, 310, 311, 312, 314, 317, 318, 324, 375, 321, 405, 415, 273, 282, 285, 295, 296, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347}
        h, w, _ = image.shape
        face_outline_connections = [(172, 136), (136, 150), (150, 149), (149, 176), (176, 148), (148, 152), (152, 377), (377, 400), (400, 378), (378, 379), (379, 365), (365, 397), (397, 288), (288, 361), (361, 323), (323, 454), (454, 323), (323, 361), (361, 340), (340, 346), (346, 347), (347, 348), (348, 349), (349, 350), (350, 451), (451, 452), (452, 453), (453, 464), (464, 172), (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162), (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10), (70, 63), (63, 105), (105, 66), (66, 107), (107, 55), (55, 65), (65, 52), (52, 53), (234, 93), (93, 132), (132, 58), (58, 172), (172, 136), (136, 172), (172, 58), (288, 361), (361, 323), (323, 340), (340, 346), (346, 347), (347, 348)]
        for start_idx, end_idx in face_outline_connections:
            if start_idx < len(face_landmarks.landmark) and end_idx < len(face_landmarks.landmark):
                if start_idx not in face_features_to_hide and end_idx not in face_features_to_hide:
                    start_pt = face_landmarks.landmark[start_idx]
                    end_pt = face_landmarks.landmark[end_idx]
                    start_x, start_y = int(start_pt.x * w), int(start_pt.y * h)
                    end_x, end_y = int(end_pt.x * w), int(end_pt.y * h)
                    cv2.line(image, (start_x, start_y), (end_x, end_y), (80, 110, 10), 1)
        for idx, landmark in enumerate(face_landmarks.landmark):
            if idx not in face_features_to_hide:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(image, (x, y), 1, (80, 256, 121), -1)

    def draw_pose_landmarks_with_face_control(image, pose_landmarks, show_face_features=True):
        if not pose_landmarks:
            return
        facial_pose_indices = set(range(11))
        h, w, _ = image.shape
        pose_connections = mp_holistic.POSE_CONNECTIONS
        for connection in pose_connections:
            start_idx, end_idx = connection
            if not show_face_features and (start_idx in facial_pose_indices or end_idx in facial_pose_indices):
                continue
            if start_idx < len(pose_landmarks.landmark) and end_idx < len(pose_landmarks.landmark):
                start_pt = pose_landmarks.landmark[start_idx]
                end_pt = pose_landmarks.landmark[end_idx]
                start_x, start_y = int(start_pt.x * w), int(start_pt.y * h)
                end_x, end_y = int(end_pt.x * w), int(end_pt.y * h)
                cv2.line(image, (start_x, start_y), (end_x, end_y), (80, 44, 121), 2)
        for idx, landmark in enumerate(pose_landmarks.landmark):
            if not show_face_features and idx in facial_pose_indices:
                continue
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (x, y), 4, (80, 22, 10), -1)
            cv2.circle(image, (x, y), 2, (80, 44, 121), -1)

    def draw_styled_landmarks(image, results):
        annotated_image = image.copy()
        if show_face_landmarks and results.face_landmarks:
            draw_face_mesh_with_hidden_features(annotated_image, results.face_landmarks)
        if show_pose_landmarks and results.pose_landmarks:
            draw_pose_landmarks_with_face_control(annotated_image, results.pose_landmarks, show_face_landmarks)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        return annotated_image

    def compute_distance(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def compute_angle(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)

    def compute_palm_normal(hand_landmarks):
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
        if hand_landmarks is None or len(hand_landmarks) == 0:
            return np.zeros(210)
        features = []
        finger_indices = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
        for finger in finger_indices:
            for i in range(len(finger) - 2):
                angle = compute_angle(hand_landmarks[finger[i]], hand_landmarks[finger[i+1]], hand_landmarks[finger[i+2]])
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
        left_hand_detected = results.left_hand_landmarks is not None and len(results.left_hand_landmarks.landmark) > 0
        right_hand_detected = results.right_hand_landmarks is not None and len(results.right_hand_landmarks.landmark) > 0
        return left_hand_detected or right_hand_detected

    def cleanup_incomplete_sequences():
        prediction_seq = global_state.get_prediction_sequence()
        if len(prediction_seq) > 0:
            cleared_count = len(prediction_seq)
            global_state.clear_prediction_sequence()
            print(f"[CLEANUP] Cleared {cleared_count} incomplete predictions")
            return True
        return False

    def extract_all_subvectors(results, previous_frame_data=None):
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
        raw_data = {'pose': pose_landmarks, 'left_hand': left_hand_landmarks, 'right_hand': right_hand_landmarks}
        location_l_hand, location_r_hand, location_pose = extract_location_features(pose_landmarks, left_hand_landmarks, right_hand_landmarks)
        handshape_l = extract_handshape_features(left_hand_landmarks)
        handshape_r = extract_handshape_features(right_hand_landmarks)
        palm_orientation = extract_palm_orientation_features(left_hand_landmarks, right_hand_landmarks)
        movement = extract_movement_features(raw_data, previous_frame_data)
        combined_features = np.concatenate([location_l_hand, location_r_hand, location_pose, handshape_l, handshape_r, palm_orientation, movement])
        return combined_features, raw_data

    def split_into_subvectors(sequence):
        subvectors = []
        start_idx = 0
        for name, size in SUBVECTOR_CONFIG.items():
            end_idx = start_idx + size
            subvector = sequence[:, start_idx:end_idx]
            subvectors.append(subvector)
            start_idx = end_idx
        return subvectors

    def predict_onnx(sequence_array):
        try:
            subvectors = split_into_subvectors(sequence_array)
            subvectors_batched = [np.expand_dims(sv.astype(np.float32), axis=0) for sv in subvectors]
            input_dict = {name: data for name, data in zip(input_names, subvectors_batched)}
            predictions = model.run([output_name], input_dict)[0]
            predictions = predictions[0]
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
        if confidence >= confidence_threshold:
            global_state.add_to_prediction_sequence({'prediction': prediction, 'confidence': confidence})
            print(f"[HIGH CONF] {prediction} ({confidence:.2%}) added to sequence")
        else:
            print(f"[FILTERED] Low confidence prediction: {prediction} ({confidence:.2%})")
        prediction_seq = global_state.get_prediction_sequence()
        if len(prediction_seq) >= 10:
            process_sequence_vote()
            global_state.clear_prediction_sequence()

    def process_sequence_vote():
        prediction_seq = global_state.get_prediction_sequence()
        if len(prediction_seq) < 10:
            return
        prediction_counts = {}
        confidence_sums = {}
        for pred_data in prediction_seq:
            pred = pred_data['prediction']
            conf = pred_data['confidence']
            if pred not in prediction_counts:
                prediction_counts[pred] = 0
                confidence_sums[pred] = 0
            prediction_counts[pred] += 1
            confidence_sums[pred] += conf
        majority_prediction = max(prediction_counts, key=prediction_counts.get)
        vote_count = prediction_counts[majority_prediction]
        avg_confidence = confidence_sums[majority_prediction] / vote_count
        total_predictions = len(prediction_seq)
        sequence_result = {'sequence': prediction_seq.copy(), 'final_prediction': majority_prediction, 'vote_count': vote_count, 'total_predictions': total_predictions, 'average_confidence': avg_confidence, 'timestamp': time.time()}
        global_state.add_final_sequence(sequence_result)
        print(f"[SEQUENCE RESULT] {majority_prediction} ({vote_count}/{total_predictions} votes, {avg_confidence:.2%} avg confidence)")
        return sequence_result

    def deduplicate_sequences():
        final_sequences = global_state.get_final_sequences()
        if len(final_sequences) == 0:
            return []
        unique_sequences = []
        prev_prediction = None
        for seq in final_sequences:
            if seq['final_prediction'] != prev_prediction:
                unique_sequences.append(seq)
                prev_prediction = seq['final_prediction']
        return unique_sequences

    class SignLanguageProcessor(VideoProcessorBase):
        def __init__(self):
            self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.sequence_length = 30
            
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="rgb24")
            image, results = mediapipe_detection(img, self.holistic)
            image = draw_styled_landmarks(image, results)
            
            hands_detected = are_hands_detected(results)
            
            # Hands loss timeout logic
            if hands_detected:
                global_state.hands_lost_counter = 0
            else:
                global_state.hands_lost_counter += 1
                if global_state.hands_lost_counter >= global_state.hands_lost_threshold:
                    cleanup_incomplete_sequences()
            
            previous_data = global_state.get_previous_frame_data()
            combined_features, raw_data = extract_all_subvectors(results, previous_data)
            global_state.append_to_sequence(combined_features, max_length=self.sequence_length)
            global_state.set_previous_frame_data(raw_data)
            
            sequence = global_state.get_sequence()
            if len(sequence) == self.sequence_length and hands_detected:
                sequence_array = np.array(sequence)
                if sequence_array.shape != (self.sequence_length, TOTAL_FEATURES):
                    sequence_array = sequence_array.reshape(self.sequence_length, -1)
                current_prediction, current_confidence = predict_onnx(sequence_array)
                add_to_prediction_sequence(current_prediction, current_confidence)
                global_state.update(current_prediction, current_confidence, global_state.get_state()['sentence'])
            
            return av.VideoFrame.from_ndarray(image, format="rgb24")

    # --- Streamlit UI ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("### Webcam Feed")

        rtc_config= RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        # 🚀 Using Free TURN servers from OpenRelay (Metered.ca)
        # This bypasses firewalls without needing Twilio or API keys.
        #rtc_config = RTCConfiguration({
        #    "iceServers": [
        #        # STUN server
        #        {"urls": ["stun:openrelay.metered.ca:80"]},
        #        
                # TURN servers (UDP)
        #        {
        #            "urls": ["turn:openrelay.metered.ca:80"],
        #            "username": "openrelayproject",
        #            "credential": "openrelayproject"
        #        },
        #        {
        #            "urls": ["turn:openrelay.metered.ca:443"],
        #            "username": "openrelayproject",
        #            "credential": "openrelayproject"
        #        },
        #        # TURN server (TCP) - Good fallback if UDP is blocked
        #        {
        #            "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
        #            "username": "openrelayproject",
        #            "credential": "openrelayproject"
        #        }
        #    ]
        #})

        # The WebRTC Component
        webrtc_streamer(
            key="sign-language", 
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config, 
            media_stream_constraints={"video": True, "audio": False}, 
            video_processor_factory=SignLanguageProcessor, 
            async_processing=True
        )
    with col2:
        st.write("### Recognized Signs")
        
        # Permanent auto-refresh every 300ms for near real-time updates
        st_autorefresh(interval=300, key="prediction_refresh")
        
        # Display final sequences (voting results)
        final_sequences = global_state.get_final_sequences()
        unique_sequences = deduplicate_sequences()
        
        if unique_sequences:
            sequence_words = [seq['final_prediction'] for seq in unique_sequences]
            sequences_text = " ".join(sequence_words)
            st.markdown(f"# `{sequences_text}`")
        else:
            st.markdown("# *Waiting for clear signs...*")

else:
    st.error("Model could not be loaded. The application cannot start.")
