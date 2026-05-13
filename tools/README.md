# Tools

This directory contains local helper utilities for preparing inputs and inspecting outputs. They are intended for setup, debugging, and offline analysis rather than the main runtime flow.

## `define_areas.py`

Interactive utility for defining polygonal semantic areas on a video frame or a still image.

### Purpose

Use it to create the area JSON consumed by the perception pipeline, for example traffic lanes, crosswalks, or other regions of interest.

### Inputs

- `--input`: path to a video or an image
- `--out`: path to the output JSON file, default `areas.json`

If the input is a video, the script uses the first frame. If the input is an image, it loads it directly.

### Output

It writes a JSON file with this structure:

```json
{
  "areas": [
    {
      "id": "lane_a",
      "type": "carlane",
      "polygon": [[x1, y1], [x2, y2], [x3, y3]]
    }
  ]
}
```

### Controls

- Left click: add a point to the current polygon
- `n`: save the current polygon as a new area
- `u`: undo the last point
- `r`: reset the current polygon
- `q`: quit and save the JSON

### Example

```bash
python tools/define_areas.py --input examples/u_turn/uturn.mp4 --out examples/u_turn/areas.json
```

## `anomaly_viewer.py`

Offline viewer for inspecting candidate events and anomalies from the `agent.log` format.

### Purpose

Use it to export per-event frames for:

- `candidate_anomaly`
- `anomaly`
- final VLM outcomes, when available

### Inputs

- `--video`: path to the original video
- `--log`: path to `agent.log`
- `--out`: output directory, default `./outputs/annotated_frames`
- `--show`: display frames in an OpenCV window while exporting
- `--hide-annotations`: export raw frames without drawing overlays

### Output

The script creates one subdirectory per event using this naming scheme:

```text
<kind>_<type>_object-<id or global>_<start>-<end>/
```

Inside each directory, it saves one JPG per `frame_id` in the event range where the target object is present:

```text
frame_<frame_id>.jpg
```

When annotations are enabled, overlays may include:

- `candidate` markers
- `anomaly` markers
- VLM status: `anomalous` or `not anomalous`
- a text box with frame range, description, and reason

### Example

```bash
python tools/anomaly_viewer.py --video examples/u_turn/uturn.mp4 --log outputs/agent.log
```

Example without overlays:

```bash
python tools/anomaly_viewer.py --video examples/u_turn/uturn.mp4 --log outputs/agent.log --hide-annotations
```
