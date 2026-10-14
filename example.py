"""Capture perspective-corrected A4 paper: Space/S saves, Q quits.

Use --image photo.jpg to process a photo without the RealSense camera.
The camera defaults to 1920 x 1080 at 8 FPS; override with --width/--height/--fps.
"""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def order_corners(corners):
    """Order a convex quadrilateral clockwise, starting at top-left."""
    points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    return np.roll(points, -np.argmin(points.sum(axis=1)), axis=0).copy()


def detect_paper(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # Close small gaps in the outside edge; drawings can remain inside it.
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < frame.shape[0] * frame.shape[1] * 0.1:
            continue
        perimeter = cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(corners) == 4 and cv2.isContourConvex(corners):
            return order_corners(corners)
    return None


def normalize_a4(frame, corners, dpi=300, orientation="auto"):
    """Warp original color pixels onto a 210 x 297 mm A4 rectangle."""
    source = order_corners(corners)
    tl, tr, br, bl = source
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    landscape = orientation == "landscape" or (orientation == "auto" and width > height)
    output_width = round(210 / 25.4 * dpi)
    output_height = round(297 / 25.4 * dpi)
    if landscape:
        output_width, output_height = output_height, output_width
    destination = np.float32([
        [0, 0], [output_width - 1, 0],
        [output_width - 1, output_height - 1], [0, output_height - 1],
    ])
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(frame, transform, (output_width, output_height), flags=cv2.INTER_CUBIC)


def save_capture(frame, corners, args):
    normalized = normalize_a4(frame, corners, args.dpi, args.orientation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"paper_a4_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
    Image.fromarray(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)).save(path, dpi=(args.dpi, args.dpi))
    print(f"Saved {path.resolve()} ({normalized.shape[1]} x {normalized.shape[0]} pixels)")
    return normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Process an existing photo without a camera")
    parser.add_argument("--output-dir", type=Path, default=Path("captures"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--width", type=int, default=1920, help="Camera stream width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Camera stream height (default: 1080)")
    parser.add_argument("--fps", type=int, default=8, help="Camera frame rate (default: 8)")
    parser.add_argument("--orientation", choices=["auto", "portrait", "landscape"], default="auto")
    args = parser.parse_args()
    if not 1 <= args.dpi <= 1200:
        parser.error("--dpi must be between 1 and 1200")
    if min(args.width, args.height, args.fps) < 1:
        parser.error("--width, --height and --fps must be positive")

    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            parser.error(f"Could not read image: {args.image}")
        corners = detect_paper(frame)
        if corners is None:
            parser.error("No paper detected. Include all four corners with a contrasting background.")
        save_capture(frame, corners, args)
        return

    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    try:
        active_profile = pipeline.start(config)
    except RuntimeError as error:
        parser.error(f"Could not start camera at {args.width} x {args.height}, {args.fps} FPS: {error}")
    try:
        stream = active_profile.get_stream(rs.stream.color).as_video_stream_profile()
        print(f"Camera input: {stream.width()} x {stream.height()} at {stream.fps()} FPS")
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data()).copy()
            corners = detect_paper(frame)
            preview = frame.copy()
            if corners is not None:
                cv2.polylines(preview, [corners.astype(np.int32)], True, (0, 255, 0), 3)
            # Only shrink the display; detection and captures use the original pixels.
            preview_scale = min(1.0, 960 / preview.shape[1], 720 / preview.shape[0])
            if preview_scale < 1:
                preview = cv2.resize(preview, None, fx=preview_scale, fy=preview_scale, interpolation=cv2.INTER_AREA)
            label = "Space/S: capture A4 | Q: quit" if corners is not None else "No paper detected | Q: quit"
            cv2.putText(preview, label, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("Paper detection", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("s"), ord(" ")):
                if corners is None:
                    print("Cannot capture: no paper detected.")
                    continue
                normalized = save_capture(frame, corners, args)
                scale = min(800 / normalized.shape[1], 800 / normalized.shape[0])
                cv2.imshow("A4 capture", cv2.resize(normalized, None, fx=scale, fy=scale))
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
