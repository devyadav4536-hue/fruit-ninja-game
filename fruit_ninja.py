import sys
import time
import math
import random
from collections import deque

import cv2
import mediapipe as mp
import pygame


# ============================================================
# SETTINGS
# ============================================================

WIDTH = 960
HEIGHT = 720

FPS = 60
GAME_TIME = 60

MODEL_PATH = "hand_landmarker.task"

# How many fingertip positions are remembered
TRAIL_LENGTH = 15

# How long the slicing trail stays visible
TRAIL_LIFE = 0.30

# Extra collision area around the fruit
SLICE_DISTANCE = 30


# ============================================================
# FRUIT SETTINGS
# ============================================================

FRUIT_TYPES = [
    ("APPLE", (220, 40, 50), (255, 150, 150), 35),
    ("ORANGE", (245, 145, 20), (255, 220, 120), 34),
    ("PLUM", (130, 60, 170), (220, 170, 250), 32),
    ("LIME", (120, 210, 50), (220, 255, 150), 32),
    ("WATERMELON", (40, 170, 70), (200, 255, 190), 40),
]


# ============================================================
# HELPER FUNCTION
# ============================================================

def distance_point_to_line(
    px, py,
    ax, ay,
    bx, by
):
    """
    Finds the distance between a point and
    a line segment.
    """

    abx = bx - ax
    aby = by - ay

    length_squared = (
        abx * abx +
        aby * aby
    )

    if length_squared == 0:

        return math.hypot(
            px - ax,
            py - ay
        )

    t = (
        (px - ax) * abx +
        (py - ay) * aby
    ) / length_squared

    t = max(
        0,
        min(1, t)
    )

    closest_x = ax + t * abx
    closest_y = ay + t * aby

    return math.hypot(
        px - closest_x,
        py - closest_y
    )


# ============================================================
# HAND TRACKER
# ============================================================

class HandTracker:

    def __init__(self):

        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(
            model_asset_path=MODEL_PATH
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,

            # We only need one hand
            num_hands=1,

            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.detector = (
            vision.HandLandmarker
            .create_from_options(options)
        )

    def detect(self, frame):

        # ----------------------------------------------------
        # IMPORTANT:
        # The camera is already mirrored BEFORE this function.
        # This is the same arrangement as hand_test.py.
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # MediaPipe wants a monotonically increasing timestamp
        timestamp_ms = int(
            time.monotonic() * 1000
        )

        result = self.detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if not result.hand_landmarks:

            return None

        # First detected hand
        hand = result.hand_landmarks[0]

        # Landmark 8 = INDEX FINGERTIP
        fingertip = hand[8]

        h, w, _ = frame.shape

        x = int(
            fingertip.x * w
        )

        y = int(
            fingertip.y * h
        )

        # Convert camera coordinates to our game size
        game_x = int(
            x * WIDTH / w
        )

        game_y = int(
            y * HEIGHT / h
        )

        # Keep inside game screen
        game_x = max(
            0,
            min(WIDTH - 1, game_x)
        )

        game_y = max(
            0,
            min(HEIGHT - 1, game_y)
        )

        return game_x, game_y

    def close(self):

        self.detector.close()


# ============================================================
# PARTICLE
# ============================================================

class Particle:

    def __init__(self, x, y, color):

        angle = random.uniform(
            0,
            math.pi * 2
        )

        speed = random.uniform(
            100,
            350
        )

        self.x = x
        self.y = y

        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed

        self.color = color

        self.size = random.randint(
            3,
            7
        )

        self.life = random.uniform(
            0.4,
            0.8
        )

    def update(self, dt):

        self.vy += 500 * dt

        self.x += self.vx * dt
        self.y += self.vy * dt

        self.life -= dt

    def draw(self, screen):

        if self.life <= 0:
            return

        alpha = int(
            max(
                0,
                min(
                    255,
                    self.life * 300
                )
            )
        )

        surface = pygame.Surface(
            (
                self.size * 2,
                self.size * 2
            ),
            pygame.SRCALPHA
        )

        pygame.draw.circle(
            surface,
            (*self.color, alpha),
            (
                self.size,
                self.size
            ),
            self.size
        )

        screen.blit(
            surface,
            (
                int(self.x - self.size),
                int(self.y - self.size)
            )
        )


# ============================================================
# FRUIT
# ============================================================

class Fruit:

    def __init__(self):

        data = random.choice(
            FRUIT_TYPES
        )

        self.name = data[0]
        self.color = data[1]
        self.highlight = data[2]
        self.radius = data[3]

        self.x = random.randint(
            80,
            WIDTH - 80
        )

        self.y = HEIGHT + self.radius

        self.vx = random.uniform(
            -180,
            180
        )

        self.vy = random.uniform(
            -850,
            -650
        )

        self.rotation = random.uniform(
            0,
            360
        )

        self.rotation_speed = random.uniform(
            -180,
            180
        )

        self.sliced = False
        self.missed = False

    def update(self, dt):

        self.vy += 900 * dt

        self.x += self.vx * dt
        self.y += self.vy * dt

        self.rotation += (
            self.rotation_speed * dt
        )

        # Fruit has fallen below screen
        if self.y - self.radius > HEIGHT:

            self.missed = True

    def draw(self, screen):

        size = (
            self.radius * 2 + 20
        )

        surface = pygame.Surface(
            (size, size),
            pygame.SRCALPHA
        )

        center = (
            size // 2,
            size // 2
        )

        # Main fruit
        pygame.draw.circle(
            surface,
            self.color,
            center,
            self.radius
        )

        # Highlight
        pygame.draw.circle(
            surface,
            self.highlight,
            (
                center[0] - self.radius // 3,
                center[1] - self.radius // 3
            ),
            max(
                5,
                self.radius // 3
            )
        )

        # Border
        darker = tuple(
            max(
                0,
                c - 60
            )
            for c in self.color
        )

        pygame.draw.circle(
            surface,
            darker,
            center,
            self.radius,
            3
        )

        # Stem
        pygame.draw.line(
            surface,
            (80, 50, 20),
            (
                center[0],
                center[1] - self.radius
            ),
            (
                center[0] + 5,
                center[1] - self.radius - 12
            ),
            4
        )

        rotated = pygame.transform.rotate(
            surface,
            self.rotation
        )

        rect = rotated.get_rect(
            center=(
                int(self.x),
                int(self.y)
            )
        )

        screen.blit(
            rotated,
            rect
        )


# ============================================================
# BOMB
# ============================================================

class Bomb(Fruit):

    def __init__(self):

        super().__init__()

        self.name = "BOMB"

        self.color = (
            20,
            20,
            20
        )

        self.highlight = (
            90,
            90,
            90
        )

        self.radius = 36

    def draw(self, screen):

        x = int(self.x)
        y = int(self.y)

        # Bomb body
        pygame.draw.circle(
            screen,
            self.color,
            (x, y),
            self.radius
        )

        # Highlight
        pygame.draw.circle(
            screen,
            self.highlight,
            (
                x - 10,
                y - 10
            ),
            9
        )

        # Fuse
        pygame.draw.line(
            screen,
            (150, 100, 40),
            (
                x,
                y - self.radius
            ),
            (
                x + 7,
                y - self.radius - 20
            ),
            4
        )

        # Spark
        pygame.draw.circle(
            screen,
            (255, 220, 30),
            (
                x + 7,
                y - self.radius - 22
            ),
            6
        )


# ============================================================
# GAME
# ============================================================

class Game:

    START = 0
    PLAYING = 1
    GAME_OVER = 2

    def __init__(self):

        self.state = self.START

        self.reset()

    def reset(self):

        self.fruits = []
        self.particles = []

        self.score = 0
        self.lives = 3

        self.time_left = GAME_TIME

        self.spawn_timer = 0

        self.trail = deque(
            maxlen=TRAIL_LENGTH
        )

        self.game_over_reason = ""

    def start(self):

        self.reset()

        self.state = self.PLAYING

    # --------------------------------------------------------
    # SPAWN
    # --------------------------------------------------------

    def spawn(self):

        # Bomb probability
        if random.random() < 0.12:

            self.fruits.append(
                Bomb()
            )

        else:

            self.fruits.append(
                Fruit()
            )

    # --------------------------------------------------------
    # SLICE
    # --------------------------------------------------------

    def slice_fruit(self, fruit):

        fruit.sliced = True

        self.score += 10

        # Create particles
        for _ in range(20):

            self.particles.append(
                Particle(
                    fruit.x,
                    fruit.y,
                    fruit.highlight
                )
            )

    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    def update(
        self,
        dt,
        fingertip
    ):

        if self.state != self.PLAYING:

            return

        # ----------------------------------------------------
        # TIMER
        # ----------------------------------------------------

        self.time_left -= dt

        if self.time_left <= 0:

            self.time_left = 0

            self.state = self.GAME_OVER

            self.game_over_reason = (
                "TIME'S UP!"
            )

            return

        # ----------------------------------------------------
        # SPAWN FRUIT
        # ----------------------------------------------------

        self.spawn_timer += dt

        # Start slower and gradually get faster
        interval = max(
            0.35,
            0.90 - (
                (GAME_TIME - self.time_left)
                / 120
            )
        )

        if self.spawn_timer >= interval:

            self.spawn_timer = 0

            self.spawn()

            # Sometimes spawn second fruit
            if random.random() < 0.25:

                self.spawn()

        # ----------------------------------------------------
        # UPDATE FRUITS
        # ----------------------------------------------------

        for fruit in self.fruits:

            fruit.update(dt)

        # ----------------------------------------------------
        # MISSED FRUIT
        # ----------------------------------------------------

        remaining = []

        for fruit in self.fruits:

            if fruit.sliced:

                continue

            if fruit.missed:

                if not isinstance(
                    fruit,
                    Bomb
                ):

                    self.lives -= 1

                continue

            remaining.append(
                fruit
            )

        self.fruits = remaining

        if self.lives <= 0:

            self.lives = 0

            self.state = self.GAME_OVER

            self.game_over_reason = (
                "OUT OF LIVES!"
            )

            return

        # ----------------------------------------------------
        # PARTICLES
        # ----------------------------------------------------

        for particle in self.particles:

            particle.update(dt)

        self.particles = [
            p
            for p in self.particles
            if p.life > 0
        ]

        # ----------------------------------------------------
        # HAND TRAIL
        # ----------------------------------------------------

        now = time.time()

        if fingertip is not None:

            self.trail.append(
                (
                    fingertip[0],
                    fingertip[1],
                    now
                )
            )

        # Remove old trail points
        while self.trail:

            age = (
                now -
                self.trail[0][2]
            )

            if age <= TRAIL_LIFE:

                break

            self.trail.popleft()

        # ----------------------------------------------------
        # SLICE COLLISION
        # ----------------------------------------------------

        if len(self.trail) >= 2:

            previous = self.trail[-2]
            current = self.trail[-1]

            x1 = previous[0]
            y1 = previous[1]

            x2 = current[0]
            y2 = current[1]

            for fruit in self.fruits:

                if fruit.sliced:

                    continue

                distance = (
                    distance_point_to_line(
                        fruit.x,
                        fruit.y,
                        x1,
                        y1,
                        x2,
                        y2
                    )
                )

                if distance <= (
                    fruit.radius +
                    SLICE_DISTANCE
                ):

                    # Bomb
                    if isinstance(
                        fruit,
                        Bomb
                    ):

                        self.state = (
                            self.GAME_OVER
                        )

                        self.game_over_reason = (
                            "YOU SLICED A BOMB!"
                        )

                        return

                    # Fruit
                    else:

                        self.slice_fruit(
                            fruit
                        )

    # --------------------------------------------------------
    # DRAW
    # --------------------------------------------------------

    def draw(
        self,
        screen,
        camera_surface,
        big_font,
        medium_font,
        small_font
    ):

        # ----------------------------------------------------
        # WEBCAM BACKGROUND
        # ----------------------------------------------------

        screen.blit(
            camera_surface,
            (0, 0)
        )

        # Slight dark overlay
        overlay = pygame.Surface(
            (WIDTH, HEIGHT),
            pygame.SRCALPHA
        )

        overlay.fill(
            (0, 0, 0, 45)
        )

        screen.blit(
            overlay,
            (0, 0)
        )

        # ----------------------------------------------------
        # PARTICLES
        # ----------------------------------------------------

        for particle in self.particles:

            particle.draw(screen)

        # ----------------------------------------------------
        # FRUITS
        # ----------------------------------------------------

        for fruit in self.fruits:

            fruit.draw(screen)

        # ----------------------------------------------------
        # TRAIL
        # ----------------------------------------------------

        self.draw_trail(
            screen
        )

        # ----------------------------------------------------
        # HUD
        # ----------------------------------------------------

        hud = pygame.Surface(
            (WIDTH, 75),
            pygame.SRCALPHA
        )

        hud.fill(
            (0, 0, 0, 150)
        )

        screen.blit(
            hud,
            (0, 0)
        )

        # Score
        score_text = medium_font.render(
            f"SCORE: {self.score}",
            True,
            (255, 255, 255)
        )

        screen.blit(
            score_text,
            (20, 18)
        )

        # Timer
        timer_text = medium_font.render(
            str(
                int(self.time_left)
            ),
            True,
            (255, 220, 40)
        )

        timer_rect = (
            timer_text.get_rect(
                center=(
                    WIDTH // 2,
                    37
                )
            )
        )

        screen.blit(
            timer_text,
            timer_rect
        )

        # Lives
        hearts = "♥" * self.lives

        lives_text = medium_font.render(
            hearts,
            True,
            (255, 50, 70)
        )

        lives_rect = (
            lives_text.get_rect(
                right=WIDTH - 20,
                centery=37
            )
        )

        screen.blit(
            lives_text,
            lives_rect
        )

        # ----------------------------------------------------
        # START SCREEN
        # ----------------------------------------------------

        if self.state == self.START:

            self.draw_dark_screen(
                screen
            )

            title = big_font.render(
                "FRUIT NINJA",
                True,
                (255, 255, 255)
            )

            title_rect = (
                title.get_rect(
                    center=(
                        WIDTH // 2,
                        HEIGHT // 2 - 100
                    )
                )
            )

            screen.blit(
                title,
                title_rect
            )

            text = small_font.render(
                "Move your index finger to slice fruit!",
                True,
                (255, 255, 255)
            )

            text_rect = (
                text.get_rect(
                    center=(
                        WIDTH // 2,
                        HEIGHT // 2 - 20
                    )
                )
            )

            screen.blit(
                text,
                text_rect
            )

            bomb_text = small_font.render(
                "AVOID THE BOMBS!",
                True,
                (255, 220, 40)
            )

            bomb_rect = (
                bomb_text.get_rect(
                    center=(
                        WIDTH // 2,
                        HEIGHT // 2 + 25
                    )
                )
            )

            screen.blit(
                bomb_text,
                bomb_rect
            )

            start_text = small_font.render(
                "Press SPACE or CLICK to start",
                True,
                (255, 255, 255)
            )

            start_rect = (
                start_text.get_rect(
                    center=(
                        WIDTH // 2,
                        HEIGHT // 2 + 85
                    )
                )
            )

            screen.blit(
                start_text,
                start_rect
            )

        # ----------------------------------------------------
        # GAME OVER
        # ----------------------------------------------------

        if self.state == self.GAME_OVER:

            self.draw_dark_screen(
                screen
            )

            title = big_font.render(
                "GAME OVER",
                True,
                (255, 60, 60)
            )

            title_rect = (
                title.get_rect(
                    center=(
                        WIDTH // 2,
                        HEIGHT // 2 - 100
                    )
                )
            )

            screen.blit(
                title,
                title_rect
            )

            reason = small_font.render(
                self.game_over_reason,
                True,
                (255, 255, 255)
            )

            reason_rect = (
                reason.get_rect(
                    center=(
                        WIDTH // 2,
                        HEIGHT // 2 - 35
                    )
                )
            )

            screen.blit(
                reason,
                reason_rect
            )

            score = medium_font.render(
                f"FINAL SCORE: {self.score}",
                True,
                (255, 220, 40)
            )

            score_rect = (
                score.get_rect(
                    center=(
                        WIDTH // 2,
                        HEIGHT // 2 + 25
                    )
                )
            )

            screen.blit(
                score,
                score_rect
            )

            restart = small_font.render(
                "Press R to play again",
                True,
                (255, 255, 255)
            )

            restart_rect = (
                restart.get_rect(
                    center=(
                        WIDTH // 2,
                        HEIGHT // 2 + 90
                    )
                )
            )

            screen.blit(
                restart,
                restart_rect

            )

    # --------------------------------------------------------
    # DARK SCREEN
    # --------------------------------------------------------

    def draw_dark_screen(self, screen):

        dark = pygame.Surface(
            (WIDTH, HEIGHT),
            pygame.SRCALPHA
        )

        dark.fill(
            (0, 0, 0, 175)
        )

        screen.blit(
            dark,
            (0, 0)
        )

    # --------------------------------------------------------
    # SLICING TRAIL
    # --------------------------------------------------------

    def draw_trail(self, screen):

        if len(self.trail) < 2:

            return

        now = time.time()

        points = list(
            self.trail
        )

        for i in range(
            1,
            len(points)
        ):

            x1, y1, t1 = points[i - 1]

            x2, y2, t2 = points[i]

            age = now - t2

            strength = max(
                0,
                1 - age / TRAIL_LIFE
            )

            width = max(
                2,
                int(15 * strength)
            )

            pygame.draw.line(
                screen,
                (255, 255, 255),
                (
                    int(x1),
                    int(y1)
                ),
                (
                    int(x2),
                    int(y2)
                ),
                width
            )

        # Current fingertip
        x, y, _ = points[-1]

        pygame.draw.circle(
            screen,
            (255, 220, 40),
            (
                int(x),
                int(y)
            ),
            12
        )

        pygame.draw.circle(
            screen,
            (255, 255, 255),
            (
                int(x),
                int(y)
            ),
            12,
            3
        )


# ============================================================
# CAMERA FRAME → PYGAME
# ============================================================

def camera_to_pygame(frame):

    # Resize
    frame = cv2.resize(
        frame,
        (WIDTH, HEIGHT)
    )

    # OpenCV BGR → RGB
    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert to Pygame
    frame = frame.swapaxes(
        0,
        1
    )

    surface = pygame.surfarray.make_surface(
        frame
    )

    return surface


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("========================================")
    print("       HAND CONTROLLED FRUIT NINJA")
    print("========================================")
    print()
    print("Starting...")
    print()
    print("SPACE / CLICK = Start")
    print("R             = Restart")
    print("Q / ESC       = Quit")
    print()

    # --------------------------------------------------------
    # PYGAME
    # --------------------------------------------------------

    pygame.init()

    screen = pygame.display.set_mode(
        (
            WIDTH,
            HEIGHT
        )
    )

    pygame.display.set_caption(
        "Hand Fruit Ninja"
    )

    clock = pygame.time.Clock()

    # Fonts
    big_font = pygame.font.SysFont(
        "Arial",
        64,
        bold=True
    )

    medium_font = pygame.font.SysFont(
        "Arial",
        34,
        bold=True
    )

    small_font = pygame.font.SysFont(
        "Arial",
        24,
        bold=True
    )

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    camera = cv2.VideoCapture(0)

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        HEIGHT
    )

    if not camera.isOpened():

        print()
        print("ERROR: Webcam could not be opened.")
        print()

        pygame.quit()

        return

    # --------------------------------------------------------
    # HAND TRACKER
    # --------------------------------------------------------

    try:

        hand_tracker = HandTracker()

    except Exception as error:

        print()
        print("ERROR: MediaPipe could not start.")
        print()
        print(error)
        print()

        camera.release()

        pygame.quit()

        return

    # --------------------------------------------------------
    # GAME
    # --------------------------------------------------------

    game = Game()

    running = True

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    while running:

        # Time between frames
        dt = clock.tick(
            FPS
        ) / 1000.0

        # ----------------------------------------------------
        # CAMERA
        # ----------------------------------------------------

        success, frame = (
            camera.read()
        )

        if not success:

            print(
                "ERROR: Could not read webcam."
            )

            break

        # ----------------------------------------------------
        # MIRROR CAMERA
        #
        # IMPORTANT:
        # Flip BEFORE hand detection.
        # This matches our working hand_test.py.
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )

        # ----------------------------------------------------
        # HAND DETECTION
        # ----------------------------------------------------

        fingertip = (
            hand_tracker.detect(
                frame
            )
        )

        # ----------------------------------------------------
        # CAMERA → PYGAME
        # ----------------------------------------------------

        camera_surface = (
            camera_to_pygame(
                frame
            )
        )

        # ----------------------------------------------------
        # KEYBOARD / MOUSE EVENTS
        # ----------------------------------------------------

        for event in pygame.event.get():

            # Close button
            if event.type == pygame.QUIT:

                running = False

            # Keyboard
            if event.type == pygame.KEYDOWN:

                # Quit
                if event.key in (
                    pygame.K_ESCAPE,
                    pygame.K_q
                ):

                    running = False

                # Start
                elif (
                    event.key == pygame.K_SPACE
                    and game.state == Game.START
                ):

                    game.start()

                # Restart
                elif (
                    event.key == pygame.K_r
                    and game.state == Game.GAME_OVER
                ):

                    game.start()

            # Mouse
            if event.type == pygame.MOUSEBUTTONDOWN:

                if game.state == Game.START:

                    game.start()

        # ----------------------------------------------------
        # UPDATE GAME
        # ----------------------------------------------------

        game.update(
            dt,
            fingertip
        )

        # ----------------------------------------------------
        # DRAW GAME
        # ----------------------------------------------------

        game.draw(
            screen,
            camera_surface,
            big_font,
            medium_font,
            small_font
        )

        # ----------------------------------------------------
        # SHOW
        # ----------------------------------------------------

        pygame.display.flip()

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    camera.release()

    hand_tracker.close()

    pygame.quit()

    cv2.destroyAllWindows()

    sys.exit()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()