# LAFFTrack Artemis
This is a fork of FORT/artemis (low level Computer Vision program for the FORmicidae Trackering) extended with various features, including marker-less object tracking using YOLO-based object detection and custom tracking techniques, multi-camera capturing, optional gui-less mode, the output of the resulting tracking in the CSV format, streaming from video files, etc.  
All the original features are retained, including: apriltags2 to detect fiducial markers, Euresys GenTL implementation to acquire data from CoaXPress framegrabber, OpenCV for live image conversion annotation and bookkeeping, and protocol buffer for realtime communication of the extracted data. However, some of those features have been extended and their naming might be refined.  

The application should be cross-platform, however, it has be validated only on Linux Ubuntu 20|22.04 x64 LTS / Debian.

Authors:  (c) Alexandre Tuleu, Artem Lutov &lt;&#108;&#97;v&commat;lumais&#46;&#99;om&gt;, Serhii Oleksenko  
License: The extension sources (LAFFTrack artemis) the original `FORT/artemis` are released under [Apache License, Version 2](www.apache.org/licenses/LICENSE-2.0.html)  

## Build and Prerequisites
There is a script `deploy.sh`, where all steps are specified for the manual build and all dependencies are listed. On top of original dependencies of FORT/artemis, it is necessary to install LibTorch and NVidia CUDA-related drivers and SDKs. The latest versions of those drivers and SDKs can be used, but that might require rebuilding of the ML models (which are not included into this repository).  
The build is performed by `cmake`. Here is an example of the manual build, specifying all external dependencies:
```sh
$ mkdir build
$ cd build && \
cmake -DCMAKE_CUDA_ARCHITECTURES=all -DCUDAToolkit_ROOT=/usr/local/cuda -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DOpenCV_DIR=/opt/xdk/opencv/build -DTORCH_INSTALL_PREFIX=/opt/xdk/libtorch-cxx11-gpu -DTorch_DIR=/opt/xdk/libtorch-cxx11-gpu/share/cmake/Torch .. && \
cmake --build . --config Release -j 4
```
The built `artemis` with its libs is generated to `build/bin`, transferring there all related assets via symlinks (scripts; ML models, and video files that are typically located in `data` as symlinks).

## Usage

```sh
build/bin$ ./artemis -h
low-level vision detection for the FORmicidae Tracker
option(s):
  -t/--ant-tracing-file        : Path where to save new detected ant pictures (default: '')
  --at-family                  : The apriltag family to use (default: '')
  --at-label-file              : The path to the label file (default: '')
  --at-quad-critical-radian    : Rejects quad with angle to close to 0 or 180 degrees (default: 0.174533)
  --at-quad-decimate           : Decimate original image for faster computation but worse pose estimation. Should be 1.0 (no decimation), 1.5, 2, 3 or 4 (default: 1)
  --at-quad-deglitch           : Deglitch only for noisy images
  --at-quad-max-line-mse       : MSE threshold to reject a fitted quad (default: 10)
  --at-quad-max-n-maxima       : Number of candidate to consider to fit quad corner (default: 10)
  --at-quad-min-bw-diff        : Difference in pixel value to consider a region black or white (default: 40)
  --at-quad-min-cluster        : Minimum number of pixel to consider it a quad (default: 5)
  --at-quad-sigma              : Apply a gaussian filter for quad detection, noisy image likes a slight filter like 0.8, for ant detection, 0.0 is almost always fine (default: 0)
  --at-refine-edges            : Refines the edge of the quad, especially needed if decimation is used.
  --at-tracking-model          : The path to the tracking model (default: '')
  --at-tracking-threads        : Number of threads for tracking detection (default: 1)
  --at-trophallaxis-model      : The path to the trophallaxis model (default: '')
  --at-trophallaxis-threads    : Number of threads for trophallaxis detection (default: 1)
  --at-useCUDA                 : Use CUDA?
  --camera-fps                 : Camera FPS to use (default: 8)
  --camera-id                  : Сamera ID (default: '')
  --camera-slave-height        : Camera Height argument for slave mode (default: 0)
  --camera-slave-width         : Camera Width argument for slave mode (default: 0)
  --camera-strobe              : Camera Strobe duration (default: '1.5ms')
  --camera-strobe-delay        : Camera Strobe delay (default: '0s')
  --fetch-resolution           : Print the camera resolution
  --frame-ids                  : Frame ID to consider in the frame sequence, if empty consider all (default: '')
  --frame-stride               : Frame sequence length (default: 1)
  -h/--help                    : Print this help message
  --highlight-tags             : Tag to highlight when drawing detections (default: '')
  --host                       : Host to send tag detection readout (default: '')
  --image-renew-period         : ant cataloguing and full frame export renew period (default: '2h0m0s')
  --input-frames               : Use a suite of input images instead of an actual framegrabber (default: '')
  --input-video                : Use of input video instead of an actual framegrabber (default: '')
  --legacy-mode                : Uses a legacy mode data output for ants cataloging and video output display. The data will be convertible to the data expected by the former Keller's group tracking system
  --log-output-dir             : Directory to put logs in (default: '')
  --new-ant-output-dir         : Path where to save new detected ant pictures (default: '')
  --new-ant-roi-size           : Size of the image to save when a new ant is found (default: 600)
  --no-gui                     : Disable GUI
  -p/--port                    : Port to send tag detection readout (default: 3002)
  --stub-image-paths           : Use a suite of stub images instead of an actual framegrabber (default: '')
  --test-mode                  : Test mode, adds an overlay detection drawing and statistics
  --trigger-mode               : Use a trigger to get a frame sequential/parallel (default: 'none')
  --uuid                       : The UUID to mark data sent over network (default: '')
  --version                    : Print version
  --video-output-file          : Sends video output to file(s) with this basename, automatically adding the suffix "_CamId-<CamId>.mp4" (default: '')
  --video-output-height        : Video Output height, width computed to maintain the aspect ratio, 0 means use frame height (default: 1080)
  --video-output-stdout        : Sends video output to stdout
  --video-output-stdout-header : Adds binary header to stdout output
```

### Parameter Selection
When reading from the input video, `--camera-fps` indicates the required streaming speed, and should be selected in a way to maximize the speed avoiding frame dropping. Frame dropping is indicated with the following warning in the terminal: `W0725 12:34:33.775141 1369935 ProcessFrameTask.cpp:288] Frame dropped due to over-processing. Total dropped: 1 (16.6667%)`. In the latter case, `--camera-fps` shuold be reduced.

### Examples of Execution

__Tracking of tagged ants with GUI__
```sh
build/bin$ ./artemis -t runs/trace-Mrubra_6-tagged.ssv --input-video data/video/TaggedAnts/Mrubra_6_ARTag_1dot3mm_test_2496px_10fps_0_MV-CH430-90XM-F-NF_clip.mp4 --at-family 16h5
```

__Tracking of tagged ants without GUI__
```sh
build/bin$ ./artemis --no-gui -t runs/trace-Mrubra_6-tagged.ssv --input-video data/video/TaggedAnts/Mrubra_6_ARTag_1dot3mm_test_2496px_10fps_0_MV-CH430-90XM-F-NF_clip.mp4 --at-family 16h5
```
Executes `artemis` without gui to detect and trace tagged (with fort-tags) ants from the video Mrubra_6 using apriltags2 for the barcode detection and tracking, outputting the tracking results into `runs/trace-Mrubra_6-tagged.ssv`:
```sh
build/bin$ ll -h
total 37M
drwxrwxr-x  5 lav lav 4.0K Jul 25 11:57 ./
drwxrwxr-x 10 lav lav 4.0K Jul 25 11:14 ../
-rwxrwxr-x  1 lav lav  37M Jul 25 11:11 artemis*
drwxrwxr-x  2 lav lav 4.0K Jul 25 01:46 configs/
lrwxrwxrwx  1 lav lav   73 Jul 25 11:14 data -> ../../data/
drwxrwxr-x  2 lav lav 4.0K Jul 25 11:11 lib/
drwxrwxr-x  2 lav lav 4.0K Jul 25 11:57 runs/

$ ls -sh runs
total 4.0K
4.0K trace-Mrubra_6-tagged.ssv

$ head runs/trace-Mrubra_6-tagged.ssv
# FrameId TagId ObjCenterX ObjCenterY
1 22 453.47 282.72
1 27 547.55 308.96
5 20 823.39 620.77
7 10 547.49 309.48
10 12 525.77 266.19
11 12 525.86 265.72
14 0 524.88 265.47
15 0 524.82 265.03
18 12 523.72 263.66
```

__Marker-less ant detection on CPU__
```sh
build/bin$ ./artemis --no-gui -t runs/trace-Mrubra_6-markerless.ssv --input-video data/video/TaggedAnts/Mrubra_6_ARTag_1dot3mm_test_2496px_10fps_0_MV-CH430-90XM-F-NF_clip.mp4 --at-trophallaxis-model data/models/AntED_yolo5_traced_992.pt --camera-fps 1
```

```txt
$ head runs/trace-Mrubra_6-markerless.ssv 
# FrameId TagId ObjCenterX ObjCenterY
0 0 394.25 79.82
0 0 555.72 241.85
0 0 503.00 296.83
0 0 863.00 385.00
0 0 559.75 859.74
0 0 587.28 660.00
0 0 825.00 651.00
1 0 394.56 79.83
1 0 554.41 242.04
```

__Marker-less ant tacking on CPU__
```sh
build/bin$ ./artemis --no-gui -t runs/trace-Mrubra_6-markerless_track.ssv --input-video data/video/TaggedAnts/Mrubra_6_ARTag_1dot3mm_test_2496px_10fps_0_MV-CH430-90XM-F-NF_clip.mp4 --at-tracking-model data/models/AntED_yolo5_traced_992.pt --camera-fps 0.2
```

```txt
$ head runs/trace-Mrubra_6-markerless_track.ssv
# FrameId TagId ObjCenterX ObjCenterY
0 0 805.60 682.67
1 1 801.71 635.54
1 2 611.77 624.04
1 3 637.10 299.28
1 4 440.45 279.23
2 1 805.94 639.32
2 2 609.86 615.54
2 3 643.37 306.86
2 4 447.57 278.87
```
