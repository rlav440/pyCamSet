from pyCamSet import load_CameraSet
from pyCamSet.utils.saving import camset_to_colmap

from pathlib import Path

camset_path = 'my/camset/path'
cams = load_CameraSet(camset_path)                # load the calibration
output_folder = Path(camset_path).parent / "sparse" / "0"
camset_to_colmap(cams, output_folder)             # export cameras.txt and rig_config.json
