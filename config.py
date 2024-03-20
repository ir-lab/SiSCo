from pathlib import Path
import os



proj_path = Path(__file__).parent.absolute()
_cache_dir = Path(proj_path, ".cache")
_svg_images_dir = Path(proj_path,".images")

if not os.path.isdir(_cache_dir): os.makedirs(_cache_dir)
if not os.path.isdir(_svg_images_dir): os.makedirs(_svg_images_dir)

PATHS = {
    "cache_dir": _cache_dir,
    "svg_images_dir": _svg_images_dir
}

# Even width and height size e.g. 100x100, 148x148 
ICON_RES = [180, 180]
camera_topic_name = "/camera/image"

