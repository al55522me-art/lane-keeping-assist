# lane-keeping-assist
Lane Keeping Assist with U-Net Segmentation  Real-time lane detection and steering assistance system trained on TuSimple dataset.  Analyzes dashcam video, calculates lane offset, generates steering commands (LEFT/RIGHT/STRAIGHT),  and provides confidence metrics + warnings. 

Pipeline: Input video → U-Net mask → Offset calculation → Commands → Real-time dashboard with FPS, angle, confidence.
Processing stages: Raw input → Binary lane mask → Overlay with offset metrics and visualization.

lane-keeping-assist/
├── analysis/                 # Video analysis results
│   ├── analyzed_video.mp4    # Video with overlay
│   ├── lane_metrics.csv      # Per-frame metrics
│   └── lane_plot.png         # Offset/confidence charts
├── models/                   # Trained models
│   └── tusimple_best.pth     # U-Net ResNet34
├── live_demo/                # Real-time webcam demo
│   └── lane_keeping.py       # webcam processing
├── requirements.txt          # Dependencies
├── README.md                 # This file
└── LICENSE

# Analyzed video example

![Lane detection demo](analyzed_video.gif)
