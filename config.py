from pathlib import Path
import os



# Obtain the absolute path to the directory where the current script is located.
# '__file__' is a special attribute that contains the path to the current file.
proj_path = Path(__file__).parent.absolute()

# Create a Path object for the '.cache' directory inside the project path.
# This directory is intended to be used for storing cache files.
# change cache directory as per your requirement
_cache_dir = Path(proj_path, ".cache")

# Create a Path object for the '.images' directory inside the project path.
# This directory is intended to store svg images.
_svg_images_dir = Path(proj_path, ".images")

# Check if the '.cache' directory exists. If it does not exist, create it.
if not os.path.isdir(_cache_dir):
    os.makedirs(_cache_dir)

# Check if the '.images' directory exists. If it does not exist, create it.
if not os.path.isdir(_svg_images_dir):
    os.makedirs(_svg_images_dir)

# Store the paths to the '.cache' and '.images' directories in a dictionary
# with the keys 'cache_dir' and 'svg_images_dir', respectively.
# This way, the paths can be easily accessed from other parts of the program.
PATHS = {
    "cache_dir": _cache_dir,
    "svg_images_dir": _svg_images_dir
}

# Define the size of icons which should be an even width and height,
# such as '100x100' or '148x148'. In this case, it's defined as '180x180'.
ICON_RES = [180, 180]

# Robot Operating System (ROS) message topic for about the camera's image stream.
camera_topic_name = "/camera/image"