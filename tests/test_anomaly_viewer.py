from __future__ import annotations

import unittest

from tools.anomaly_viewer import (
    AnomalyEvent,
    ObjectSnapshot,
    PerceptionSnapshot,
    select_snapshots_for_event,
)


class AnomalyViewerTests(unittest.TestCase):
    def test_temporal_anomaly_uses_full_frame_range(self) -> None:
        event = AnomalyEvent(
            object_id="car_1",
            type="u_turn",
            description="id=car_1",
            start_frame_id=80,
            end_frame_id=100,
            answer_frame_ids=(100,),
        )
        snapshots = [
            PerceptionSnapshot(
                frame_id=frame_id,
                objects={
                    "car_1": ObjectSnapshot(
                        object_id="car_1",
                        object_class="car",
                        x=10,
                        y=20,
                    )
                },
            )
            for frame_id in range(75, 106)
        ]

        selection = select_snapshots_for_event(event, snapshots)

        self.assertEqual(80, selection.viewer_window.start_frame_id)
        self.assertEqual(100, selection.viewer_window.end_frame_id)
        self.assertEqual(
            list(range(80, 101)),
            [snapshot.frame_id for snapshot in selection.snapshots],
        )


if __name__ == "__main__":
    unittest.main()
