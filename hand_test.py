import cv2
import mediapipe as mp
import sys


# ============================================================
# SETTINGS
# ============================================================

CAMERA_WIDTH = 960
CAMERA_HEIGHT = 720

MODEL_PATH = "hand_landmarker.task"


# ============================================================
# MEDIAPIPE HAND TRACKER
# ============================================================

def create_hand_tracker():

    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,

        # Detect only one hand
        num_hands=1,

        # Detection settings
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    tracker = vision.HandLandmarker.create_from_options(
        options
    )

    return tracker


# ============================================================
# DRAW HAND
# ============================================================

def draw_hand(frame, hand_landmarks):

    height, width, _ = frame.shape

    # --------------------------------------------------------
    # Convert all landmarks into pixel coordinates
    # --------------------------------------------------------

    points = []

    for landmark in hand_landmarks:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    # --------------------------------------------------------
    # Hand connections
    # --------------------------------------------------------

    connections = [

        # Thumb
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        # Index finger
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        # Middle finger
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        # Ring finger
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        # Pinky
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        # Palm
        (5, 9),
        (9, 13),
        (13, 17)
    ]

    # --------------------------------------------------------
    # Draw connections
    # --------------------------------------------------------

    for start, end in connections:

        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            3
        )

    # --------------------------------------------------------
    # Draw all 21 landmarks
    # --------------------------------------------------------

    for i, point in enumerate(points):

        # Index fingertip = landmark 8
        if i == 8:

            # Large yellow circle
            cv2.circle(
                frame,
                point,
                14,
                (0, 255, 255),
                -1
            )

            # White border
            cv2.circle(
                frame,
                point,
                16,
                (255, 255, 255),
                2
            )

        else:

            # Normal landmark
            cv2.circle(
                frame,
                point,
                6,
                (0, 255, 0),
                -1
            )

    # --------------------------------------------------------
    # Index fingertip information
    # --------------------------------------------------------

    index_x, index_y = points[8]

    # Crosshair
    cv2.line(
        frame,
        (index_x - 25, index_y),
        (index_x + 25, index_y),
        (255, 255, 255),
        2
    )

    cv2.line(
        frame,
        (index_x, index_y - 25),
        (index_x, index_y + 25),
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "INDEX FINGERTIP",
        (index_x + 20, index_y - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"X: {index_x}   Y: {index_y}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    return index_x, index_y


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print()
    print("======================================")
    print("       HAND TRACKING TEST")
    print("======================================")
    print()
    print("Starting webcam...")
    print()
    print("Move your index finger around.")
    print("The yellow dot should follow it.")
    print()
    print("Press Q to quit.")
    print("Press ESC to quit.")
    print()

    # --------------------------------------------------------
    # Open webcam
    # --------------------------------------------------------

    camera = cv2.VideoCapture(0)

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    # Check camera
    if not camera.isOpened():

        print()
        print("ERROR!")
        print("The webcam could not be opened.")
        print()
        print("Check that:")
        print("1. Your webcam is connected.")
        print("2. No other application is using it.")
        print("3. Camera permission is enabled in Windows.")
        print()

        return

    # --------------------------------------------------------
    # Create MediaPipe tracker
    # --------------------------------------------------------

    try:

        tracker = create_hand_tracker()

    except Exception as error:

        print()
        print("ERROR STARTING MEDIAPIPE")
        print()
        print(error)
        print()

        camera.release()

        return

    # --------------------------------------------------------
    # Video timestamp
    # --------------------------------------------------------

    timestamp = 0

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    while True:

        # Get camera frame
        success, frame = camera.read()

        if not success:

            print("Could not read webcam.")

            break

        # ----------------------------------------------------
        # Mirror camera
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )

        # ----------------------------------------------------
        # Convert BGR → RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Create MediaPipe image
        # ----------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # ----------------------------------------------------
        # Increase timestamp
        # ----------------------------------------------------

        timestamp += 1

        # ----------------------------------------------------
        # Detect hand
        # ----------------------------------------------------

        result = tracker.detect_for_video(
            mp_image,
            timestamp
        )

        # ----------------------------------------------------
        # Display status
        # ----------------------------------------------------

        if result.hand_landmarks:

            cv2.putText(
                frame,
                "HAND DETECTED",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # First hand
            hand = result.hand_landmarks[0]

            # Draw hand
            draw_hand(
                frame,
                hand
            )

        else:

            cv2.putText(
                frame,
                "NO HAND DETECTED",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # ----------------------------------------------------
        # Instructions
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "Move your INDEX FINGER",
            (20, frame.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Q / ESC = EXIT",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Show window
        # ----------------------------------------------------

        cv2.imshow(
            "Hand Tracking Test",
            frame
        )

        # ----------------------------------------------------
        # Keyboard
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

        if key == 27:  # ESC

            break

    # --------------------------------------------------------
    # Clean up
    # --------------------------------------------------------

    camera.release()

    tracker.close()

    cv2.destroyAllWindows()

    print()
    print("Hand tracking stopped.")
    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()