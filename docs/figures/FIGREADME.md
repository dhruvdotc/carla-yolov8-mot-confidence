## Figures and Analysis

This section summarizes the behavior of the detection–tracking pipeline under different detection confidence thresholds using **ByteTrack** on CARLA simulation footage. Since no ground-truth multi-object tracking annotations are available, the figures focus on **self-consistency and temporal stability metrics** rather than absolute accuracy.

---

### Figure 1: Track Lifetime Distribution  
**(track_lifetime_histogram.png)**

This histogram shows the distribution of continuous track lifetimes (in seconds) for ByteTrack at detection confidence thresholds of 0.40, 0.50, and 0.65.

Most tracks are short lived, indicating a high rate of transient detections and ID fragmentation, which is expected under domain shift (COCO-trained detector on CARLA imagery). However, lower and moderate confidence thresholds (0.40-0.50) exhibit a longer tail of persistent tracks lasting up to tens of seconds, corresponding to vehicles that remain consistently visible and correctly associated over time.

This figure illustrates the tradeoff between detection strictness and temporal identity persistence.

---

### Figure 2: Short-Lived Track Churn (< 1s)  
**(short_track_churn_lt1s.png)**

This bar chart shows the fraction of tracks that terminate within the first second of their existence.

As detection confidence increases, the fraction of short-lived tracks rises significantly. Higher thresholds suppress detections in consecutive frames, preventing trackers from maintaining identity continuity. As a result, tracks frequently terminate immediately after initialization.

This metric captures **track churn**, a practical indicator of instability and false-positive-driven fragmentation in tracking pipelines.

---

### Figure 3: Average Active Tracks per Frame  
**(avg_active_tracks_per_frame.png)**

This plot reports the mean number of simultaneously active tracks per frame.

Lower confidence thresholds yield higher scene occupancy, with more objects being tracked at any time. At higher confidence (0.65), the number of active tracks drops sharply, indicating loss of recall rather than improved tracking quality.

This figure helps distinguish between apparent “cleanliness” and actual system coverage of the environment.

---

### Figure 4: Average Detections per Frame  
**(avg_detections_per_frame.png)**

This plot shows the average number of raw detector outputs per frame before tracking.

Detection volume decreases monotonically as the confidence threshold increases, and this reduction directly explains the trends observed in Figures 2 and 3. Fewer detections lead to fewer associations, increased temporal gaps, and higher track churn.

This figure establishes detection sparsity as the root cause of downstream tracking behavior changes.

---

## Summary Insight

Together, these figures demonstrate that increasing detection confidence improves visual cleanliness but degrades temporal stability by reducing detection recall. A moderate confidence threshold (0.50) provides the best tradeoff between noise suppression and persistent object tracking, motivating the use of different thresholds for visualization versus tracking.
