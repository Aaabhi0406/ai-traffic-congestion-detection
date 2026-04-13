from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(
    max_age=5,                # DELETE track after 5 frames of no detection
                              # (was 20 — caused ghost boxes expanding across frame)
    n_init=2,                 # confirm track after 2 consecutive detections
                              # (was 1 — any single noise detection became a track)
    nms_max_overlap=0.5,      # stricter overlap suppression (was 0.7)
    max_cosine_distance=0.25, # stricter appearance match (was 0.4 — too loose)
    nn_budget=100,
)
