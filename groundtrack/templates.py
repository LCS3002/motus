"""Site config templates written by `groundtrack init`."""

COMMON_HEAD = """\
# motus site config: {site}
# Paths are relative to this file. Run `motus calibrate --config {fname}` first.
site: {site}
video: ../footage/{site}.mp4
geotiff: ../maps/{site}.tif              # north-up GeoTIFF in EPSG:27700 (QGIS / Digimap export)
homography: ../calibration/{site}_homography.json
lens: null                                # ../calibration/phone_1x_lens.json (optional)
output_dir: ../runs
models_dir: ../models                     # YOLO weights are downloaded here on first use
device: auto                             # auto | cuda | mps | cpu
fps: null                                 # set only if the video reports a wrong frame rate
"""

PLAZA = COMMON_HEAD + """
detection:
  model: yolo26m.pt         # yolo26n/s/m/l/x: bigger = better on small far people, slower
  imgsz: 1920               # phone video's long side: on a plaza clip 1920 found 34 % more
                            # people than 1280 at the same speed (smaller, farther people)
  conf: 0.25
  tracker: botsort          # bytetrack | botsort | path/to/custom.yaml
  track_buffer_s: 1.5       # people get occluded by each other for longer than cars
  vid_stride: 1
  roi: null                 # run `motus roi --config {fname}` to draw one
  record_predicted: true
  suppress_riders: true
  stabilize: auto           # measures camera motion first; compensates only if it moved

groups:
  people:
    classes: [person]
    speed_range: [0.0, 2.5]
    max_speed: 6.0
    smooth_window_s: 1.0
  cycles:
    classes: [bicycle]
    speed_range: [0.0, 10.0]
    max_speed: 20.0
    smooth_window_s: 0.8

projection:
  max_range_m: auto         # drop far people where 1 px of jitter > 0.25 m of depth

cleaning:
  min_track_s: 0.0
  max_gap_s: 2.0

grid:
  cell_size_m: 1.0
  include_predicted: false

field:                      # smoothed vector field for Houdini particle sims
  cell_size_m: 0.5
  smooth_m: 1.5
  time_window_s: null       # e.g. 10 for a field per 10 s (animated)

stats:
  histogram_bins: 25
  count_lines: []           # e.g. - {{name: north_entrance, a: [530120.0, 180340.0], b: [530128.0, 180336.0]}}

visuals:
  extent: data              # data | geotiff
  margin_m: 8.0
  line_width: 1.0

debug_video:
  enabled: false
  blur_people: true
"""

MOTORWAY = COMMON_HEAD + """
detection:
  model: yolo26m.pt
  imgsz: 1920               # cars are small from a high-rise: use the full width
  conf: 0.2                 # 0.2 found 21 % more vehicles than 0.3 on a high-rise clip, with
                            # longer tracks; cleaning removes the occasional false box
  tracker: bytetrack
  track_buffer_s: 1.0
  vid_stride: 1
  roi: null                 # strongly recommended: cut the far, blurry end of the road
  record_predicted: true
  suppress_riders: true
  stabilize: auto           # measures camera motion first; compensates only if it moved

groups:
  vehicles:
    classes: [car, motorcycle, bus, truck]
    speed_range: [0.0, 35.0]          # m/s (35 m/s = 126 km/h)
    max_speed: 70.0
    smooth_window_s: 0.6
    # half vehicle length; per class mapping allowed
    ground_offset_m: {{default: 2.2, bus: 5.5, truck: 5.0, motorcycle: 1.0}}
    offset_mode: travel               # travel | view   (see README: 'Vehicle ground point')
    vehicle_width_m: 1.8
  trains:                             # light rail / metro, if any is in view
    classes: [train]
    plane_height_m: auto              # elevated tracks: height fitted to the OSM rail lines

projection:
  max_range_m: auto         # drop far vehicles where 1 px of jitter > 1 m of depth

cleaning:
  min_track_s: 0.0
  max_gap_s: 1.5

grid:
  cell_size_m: 2.0
  include_predicted: false

field:                      # smoothed vector field for Houdini particle sims
  cell_size_m: 2.0
  smooth_m: 4.0             # keeps the two carriageways apart if they are > ~10 m apart
  time_window_s: 10         # traffic pulses (lights, queues) show up as an animated field

stats:
  histogram_bins: 28
  count_lines: []           # one line across each carriageway gives directional flows

visuals:
  extent: data
  margin_m: 15.0
  line_width: 0.8

debug_video:
  enabled: false
  blur_people: true
"""

TEMPLATES = {"plaza": PLAZA, "motorway": MOTORWAY}
