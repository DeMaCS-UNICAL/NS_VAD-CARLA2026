#!/usr/bin/env python3
"""Interactive utility to define polygonal semantic areas on a frame."""

import argparse
import json
from pathlib import Path
from typing import Any

import cv2


def load_frame(input_path: str):
    path = Path(input_path)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if path.suffix.lower() in image_exts:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {path}")
        return frame

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Could not read first frame from video: {path}")

    return frame


def build_areas_payload(areas: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {"areas": areas}


def save_areas_json(output_path: str, areas: list[dict[str, Any]]) -> None:
    Path(output_path).write_text(
        json.dumps(build_areas_payload(areas), indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Define polygonal semantic areas on a video frame or image and export them as JSON."
    )
    parser.add_argument("--input", required=True, help="Path to an input video or image frame")
    parser.add_argument("--out", default="areas.json", help="Path to the output JSON file")
    args = parser.parse_args()

    frame = load_frame(args.input)
    original = frame.copy()

    areas = []
    current_polygon = []

    window_name = "Area Definition Tool"

    def redraw():
        canvas = original.copy()

        for area in areas:
            pts = area["polygon"]
            for i, point in enumerate(pts):
                cv2.circle(canvas, tuple(point), 5, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(canvas, tuple(pts[i - 1]), tuple(point), (0, 255, 0), 2)
            if len(pts) > 2:
                cv2.line(canvas, tuple(pts[-1]), tuple(pts[0]), (0, 255, 0), 2)

            cv2.putText(
                canvas,
                f"{area['id']} ({area['type']})",
                tuple(pts[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        for i, point in enumerate(current_polygon):
            cv2.circle(canvas, tuple(point), 5, (0, 0, 255), -1)
            if i > 0:
                cv2.line(canvas, tuple(current_polygon[i - 1]), tuple(point), (0, 0, 255), 2)

        cv2.putText(
            canvas,
            "Left click: add point | n: save area | u: undo point | r: reset current | q: quit",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return canvas

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_polygon.append([x, y])
            cv2.imshow(window_name, redraw())

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, redraw())

    print("\nControls:")
    print("  Left click = add polygon point")
    print("  n          = save current polygon as a new area")
    print("  u          = undo last point")
    print("  r          = reset current polygon")
    print("  q          = quit and save JSON")

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord("n"):
            if len(current_polygon) < 3:
                print("A polygon must have at least 3 points.")
                continue

            area_id = input("Area id: ").strip()
            area_type = input("Area type: ").strip()

            if not area_id or not area_type:
                print("Area id and type cannot be empty.")
                continue

            areas.append({
                "id": area_id,
                "type": area_type,
                "polygon": current_polygon.copy(),
            })

            print(f"Saved area: {area_id} ({area_type})")
            current_polygon.clear()
            cv2.imshow(window_name, redraw())

        elif key == ord("u"):
            if current_polygon:
                current_polygon.pop()
                cv2.imshow(window_name, redraw())

        elif key == ord("r"):
            current_polygon.clear()
            cv2.imshow(window_name, redraw())

        elif key == ord("q"):
            break

    save_areas_json(args.out, areas)

    print(f"\nSaved JSON to: {args.out}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
