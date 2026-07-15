# Assets

Drop your demo GIF/screenshot here as `demo.gif` (or `demo.png`) and it will
render in the main README (see the placeholder image link near the top).

## How to record the demo GIF

1. Run the dashboard: `streamlit run src/traffic_system/dashboard.py`
2. Select a video, hit **Start Processing**, let it run ~10-15 seconds so the
   detection boxes, ROI, and metrics are all visibly updating.
3. Screen-record that window:
   - **macOS**: Cmd+Shift+5 → record selected portion → save as .mov
   - **Windows**: Win+G (Xbox Game Bar) → record → save as .mp4
   - **Linux**: `peek` or `simplescreenrecorder`
4. Convert to an optimized GIF (keeps repo size sane):
   ```bash
   ffmpeg -i demo_raw.mov -vf "fps=12,scale=960:-1:flags=lanczos" -loop 0 assets/demo.gif
   ```
   Aim for under ~8MB — trim to 8-10 seconds if it's larger.
5. Commit `assets/demo.gif` and it'll show up in the README automatically.
