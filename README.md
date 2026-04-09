# lane-keeping-assist
Lane Keeping Assist with U-Net Segmentation  Real-time lane detection and steering assistance system trained on TuSimple dataset.  Analyzes dashcam video, calculates lane offset, generates steering commands (LEFT/RIGHT/STRAIGHT),  and provides confidence metrics + warnings. 

Pipeline: Input video → U-Net mask → Offset calculation → Commands → Real-time dashboard with FPS, angle, confidence.
Processing stages: Raw input → Binary lane mask → Overlay with offset metrics and visualization.

# Analyzed video example

<div align="center">
<video width="800" height="288" controls autoplay loop muted playsinline>
  <source src="analysis/analyzed_video.mp4" type="video/mp4">
  <p>Your browser doesn't support video.</p>
</video>
</div>
