import os
import argparse
import cv2
import numpy as np
import rospy
from mixed_reality.projections import Projections
from robot.motion_planner import Motion_Planner

from llm_utils import LLM_UTILS
from sensor_msgs.msg import Image
from context.context4task import *
from context.core import *

from utils import *
from config import *
from cv_bridge import CvBridge
import hashlib
from pynput import keyboard
from pynput.keyboard import Key
import PySimpleGUI as sg
import json
import threading
from threading import Thread

class EXPERIMENT(Motion_Planner):
    def __init__(self, proj_x=1.6, proj_y=0.8, model="4.0-turbo", debug_mode=1, seed=None, data_dir=None, nli_duration=240, debug=False,
                 move_only=False, same_user=False):
        super().__init__()
        
        """
        Initialize an EXPERIMENT object.

        Args:
        - proj_x (float): Length of projection in the X-axis (default: 1.6).
        - proj_y (float): Length of projection in the Y-axis (default: 0.8).
        - model (str): Model version (default: "4.0-turbo").
        - debug_mode (int): Debugging mode (default: 1 = NLS, 2 = VSIntPro, 3 = VSM).
        - seed (int or None): Random seed for reproducibility (default: None).
        - data_dir (str or None): Directory to store user data (default: None, uses "./user_exp_data").
        - nli_duration (int): Duration for Natural Language Interaction (NLI) in seconds (default: 240).
        - debug (bool): Enable debug mode (default: False).
        - move_only (bool): Flag to indicate whether the robot should only move to waypoints without grabbing objects (default: False).
        - same_user (bool): Flag to indicate whether to use the same user or not (default: False).
        """


        # Initialize class variables
        self.debug = debug
        self.move_only = move_only
        self.same_user = same_user

        # Display a message if move_only mode is enabled
        if self.move_only:
            print("\nATTENTION: Robot will only move to waypoints but will not grab objects\n")

        # Set default data directory or use provided data_dir
        self.data_dir = "./user_exp_data" if data_dir is None else data_dir
        print(f"user data dir: {self.data_dir}")

        # Create the data directory if it doesn't exist
        path = Path(__file__).parent.absolute()
        self.data_dir = path / self.data_dir
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)

        # Get list of users and sort them
        self.users = os.listdir(self.data_dir)
        self.users.sort()
        self.init_exp_paras = True

        # Initialize user-related parameters
        if self.same_user:
            self.user_name = self.get_username()
        else:
            self.user_name = None
            self.init_paras()
            if self.user_name is None or self.user_name == '':
                self.user_name = "default"
            user_file = open(".username.txt", "w")
            user_file.write(self.user_name)
            print(f"User: {self.user_name}")

        # Display message if same_user mode is enabled
        if self.same_user:
            print(f"\nATTENTION: loading saved user name: {self.user_name}\n")

        # Create user directory if it doesn't exist
        self.user_dir = self.data_dir / self.user_name
        if not os.path.isdir(self.user_dir):
            os.makedirs(self.user_dir)

        # Set other directories and paths
        self.svg_images_dir = PATHS.get("svg_images_dir")

        # Initialize time-related variables
        self.start_time = 0
        self.end_time = 0
        self.start_ts = None
        self.end_ts = None

        # Initialize random seed
        seed_seq = [int(ord(_)) for _ in self.user_name]
        np.random.seed(seed_seq)

        # Set modes based on debug mode
        if self.debug:
            if debug_mode == 1:
                self.modes = ["NLS"] * 6
            elif debug_mode == 2:
                self.modes = ["VSIntPro"] * 6
            elif debug_mode == 3:
                self.modes = ["VSM"] * 6
        else:
            self.modes = ['NLS', 'VSIntPro', 'VSIntPro', 'VSM', 'VSM', 'NLS']

        # Shuffle modes and structures
        self.structures = ['R', 'S', 'U', 'O', 'K', 'Z']
        np.random.shuffle(self.modes)
        np.random.shuffle(self.structures)
        print(f"Experiment modes order: {self.modes}")
        print(f"Experiment structures order: {self.structures}")

        # Initialize NLI counter and duration, projection lengths, and utility classes
        self.nli_counter = 1
        self.nli_duration = nli_duration  # seconds
        self.proj_x = proj_x
        self.proj_y = proj_y
        self.project = Projections(proj_x_len=self.proj_x, proj_y_len=self.proj_y)
        self.llms = LLM_UTILS(model=model)

        # kill thread event
        self.kill_thread_event = threading.Event()

        # Initialize key variables and start keyboard thread
        self.key = None
        self.keyboard_thread = Thread(target=self.key_listen, args=())
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()
        self.keyboard = keyboard.Controller()
        
        # Initialize image directory and webcam capture
        self.img_dir = self.user_dir / "images"
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FOCUS, 0)
        self.start_saving_imsgs = False
        self.cap_th = Thread(target=self.cv2_image_callback, args=())
        self.cap_th.start()

    
    def get_username(self):
        """
        Get the username from the ".username.txt" file.

        Returns:
        - str: User name read from the file.

        Raises:
        - ValueError: If the file ".username.txt" does not exist or if the file is empty.
        """
        # Check if the ".username.txt" file exists
        if not os.path.isfile(".username.txt"):
            raise ValueError("User name is not set or write user name in \".username.txt\" file")

        try:
            # Attempt to read the first line from the file
            with open(".username.txt", "r") as fh:
                data = fh.read().splitlines()[0]
        except Exception as e:
            # Return None if there was an error reading the file
            return None

        # Return the username read from the file
        return data
    
    def cv2_image_callback(self):
        """
        Continuously captures images from the webcam and saves them to the specified directory.

        This method runs in a separate thread and keeps capturing images until the ROS node is shut down.

        Returns:
        - None
        """
        while not rospy.is_shutdown():  # Continue capturing images until ROS node is shut down
            if not self.start_saving_imsgs:  # Check if image saving flag is set
                continue  # Skip image capture if saving flag is not set

            # Create the image directory if it doesn't exist
            if not os.path.isdir(self.img_dir):
                os.makedirs(self.img_dir)

            # Read an image from the webcam
            r, img = self.cap.read()

            # Uncomment the following lines to display the image (requires OpenCV GUI)
            # cv2.imshow("image", img)
            # cv2.waitKey(1)

            # Generate a unique filename based on the timestamp and save the image
            img_file = str(self.img_dir / f"{get_timestamp()}.jpg")
            cv2.imwrite(img_file, img)

            # Wait for a short duration before capturing the next image (assuming 60 FPS)
            rospy.sleep(1/60)

            # kill thread
            if self.kill_thread_event.is_set():
                break

        return 

    
    def key_listen(self):
        """
        Listens for keypress events using the keyboard library.

        This method creates a listener object that calls the self.on_press method
        whenever a key is pressed. It continues listening until the listener is stopped.

        Returns:
        - None
        """
        # Create a keyboard listener object with the on_press callback set to self.on_press
        with keyboard.Listener(on_press=self.on_press) as listener:
            # Start listening for keypress events
            listener.join()

    
    def on_press(self, key):
        """
        Handles keypress events captured by the keyboard listener.

        This method is called whenever a key is pressed. It checks for specific key
        presses and performs corresponding actions.

        Args:
        - key (Key or str): The key that was pressed.

        Returns:
        - None
        """

        # kill thread
        if self.kill_thread_event.is_set():
            return False

        try:
            # Check if the key pressed is the spacebar to end the experiment
            if key == Key.space:
                print("pressed \"space\"")
                # Record the end time and timestamp
                self.end_time = time.time_ns()
                self.end_ts = get_timestamp()
                self.key = "space"
                

            # Check if the key pressed is 's' to start the experiment
            if key.char == 's':
                print("pressed 's'")
                self.key = "s"

            

        except AttributeError:
            # Ignore non-character keys or attributes that are not present
            pass
        
    def init_paras(self):
        """
        Initializes parameters for the Human-Robot Teaming Experiment.

        This method creates a graphical user interface (GUI) window using the PySimpleGUI library
        to prompt the user to enter their name. The user's input is retrieved and stored as the
        'user_name' attribute.

        Returns:
        - None
        """
        # Define the layout of the GUI window using PySimpleGUI
        layout = [
            [sg.Text('Enter your name below')],  # Text prompt
            [sg.InputText(key='Input')],  # Input field
            [sg.Button('Ok', key="ok")]  # Ok button
        ]

        # Create the GUI Window
        window = sg.Window('Human Robot Teaming Experiment', layout, size=(900, 500), font=("Arial", 35),
                        finalize=True, location=(1080, 0))  # Window title, size, font, and location
        window['Input'].bind("<Return>", "_Enter")  # Bind the Enter key to the input field

        values = None  # Initialize values variable
        # Event Loop to process events and get input values
        while True:
            event, values = window.read()  # Read events and values from the GUI window
            self.user_name = values.get("Input")  # Get the user's input from the input field

            # Check for window closure or button clicks
            if event == sg.WIN_CLOSED or event == 'Cancel' or event == "ok":
                break
            # Check if the Enter key was pressed in the input field
            if event == "Input" + "_Enter":
                print(f"You pressed 'Enter'")
                break

        # Check if the user input is None and raise an error if so
        if values.get("Input") is None:
            self.terminate_th = True  # Set terminate flag to True
            raise ValueError("Please enter the user name")

        window.close()  # Close the GUI window
        self.init_exp_paras = False  # Update init_exp_paras flag
        return
    
    def synthesize_signals(self, properties, temperature=0.0, max_token=16000, obj_=True, inst_=True, nli_=True):
        """
        Synthesizes signals based on input properties and options.

        Args:
        - properties (dict): Dictionary containing task-related properties.
        - temperature (float): Temperature parameter for text generation (default: 0.0).
        - max_token (int): Maximum number of tokens for text generation (default: 16000).
        - obj_ (bool): Flag to enable object visual signal synthesis (default: True).
        - inst_ (bool): Flag to enable instruction visual signal synthesis (default: True).
        - nli_ (bool): Flag to enable natural language interaction signal synthesis (default: True).

        Returns:
        - tuple: Tuple containing synthesized object visual signal, instruction visual signal, and natural language interaction signal.
        """
        # Task manager prompt and response generation
        task_context = firstExperimentTaskDescription(
            structure=properties.get("structure"),
            object_description=properties.get("object_description"),
            object_color=properties.get("object_color"),
            orientation=properties.get("orientation"),
            instruction=properties.get("instruction"),
            goal_position=properties.get("goal_position")
        )
        tm_prompt = generate_task_master_prompt(task_context)
        tm_msg = self.llms.generate_openai_message(prompt=tm_prompt, printout=False)
        tm_rsp = self.llms.call_model_response(messages=tm_msg, temperature=temperature, max_tokens=max_token)

        if not isinstance(tm_rsp, str):
            tm_rsp = tm_rsp.choices[0].message.content

        # Object visual signal synthesis
        obj_vss_rsp = None
        if obj_:
            obj_vss_prompt = generate_obj_vss_prompt(response=tm_rsp)
            obj_vss_msg = self.llms.generate_openai_message(prompt=obj_vss_prompt, printout=False)
            obj_vss_rsp = self.llms.call_model_response(messages=obj_vss_msg, temperature=temperature, max_tokens=max_token)
            if not isinstance(obj_vss_rsp, str):
                obj_vss_rsp = obj_vss_rsp.choices[0].message.content

        # Instruction visual signal synthesis
        inst_vss_rsp = None
        if inst_:
            inst_vss_prompt = generate_inst_vss_prompt(response=tm_rsp)
            inst_vss_msg = self.llms.generate_openai_message(prompt=inst_vss_prompt, printout=False)
            inst_vss_rsp = self.llms.call_model_response(messages=inst_vss_msg, temperature=temperature, max_tokens=max_token)
            if not isinstance(inst_vss_rsp, str):
                inst_vss_rsp = inst_vss_rsp.choices[0].message.content

        # Natural language interaction signal synthesis
        nlis_rsp = None
        if nli_:
            nlis_prompt = generate_nlis_prompt(response=tm_rsp)
            nlis_msg = self.llms.generate_openai_message(prompt=nlis_prompt, printout=False)
            nlis_rsp = self.llms.call_model_response(messages=nlis_msg, temperature=temperature, max_tokens=max_token)
            if not isinstance(nlis_rsp, str):
                nlis_rsp = nlis_rsp.choices[0].message.content

        return obj_vss_rsp, inst_vss_rsp, nlis_rsp


    def raterize_projection_visual_signals(self, obj_vss_rsp, inst_vss_rsp, object_name=None):
        """
        Rasterizes projection visual signals based on object and instruction visual signal responses.

        Args:
        - obj_vss_rsp (str): Object visual signal response.
        - inst_vss_rsp (str): Instruction visual signal response.
        - object_name (str): Name of the object (default: None).

        Returns:
        - list: List containing rasterized visual signals and additional information.
        """
        # Get images from SVG code
        inst_img = svg_to_cv(svg_code=get_svg_files(inst_vss_rsp)[0], show=False)
        obj_img_init = svg_to_cv(svg_code=get_svg_files(obj_vss_rsp)[0], show=False)

        # Save SVG files
        save_svg(f"{str(self.svg_images_dir)}/{object_name}.svg", get_svg_files(obj_vss_rsp)[0])
        save_svg(f"{str(self.svg_images_dir)}/{object_name}_inst.svg", get_svg_files(inst_vss_rsp)[0])

        # Save JPEG image of the object
        cv2.imwrite(f"{str(self.svg_images_dir)}/{object_name}.jpg", obj_img_init)
        iih, iiw, _ = inst_img.shape

        # Parse transformation instructions from instruction visual signal
        (start_x, start_y), (goal_x, goal_y), ori = get_start_goal_orientation(inst_vss_rsp)
        icon_size = ICON_RES
        cv_icon_ctr_x = icon_size[0] // 2
        cv_icon_ctr_y = icon_size[1] // 2
        M = cv2.getRotationMatrix2D((cv_icon_ctr_x, cv_icon_ctr_y), -ori, 1.0)
        obj_img_rot = cv2.warpAffine(obj_img_init, M, (icon_size[1], icon_size[0]))

        # Create blank canvas for visualization
        outer_image = np.ones((int(1000 * self.proj_y), int(1000 * self.proj_x), 3), dtype=np.uint8) * 0
        copy_outer_image_1 = np.copy(outer_image)
        copy_outer_image_2 = np.copy(outer_image)
        copy_outer_image_3 = np.copy(outer_image)
        oih, oiw, _ = outer_image.shape

        # Calculate offsets for positioning images
        x_offset = int((outer_image.shape[1] - inst_img.shape[1]) / 2)
        y_offset = int((outer_image.shape[0] - inst_img.shape[0]) / 2)

        # Adjust start and goal positions with offsets
        start_x = start_x + x_offset
        start_y = start_y + y_offset
        goal_x = goal_x + x_offset
        goal_y = goal_y + y_offset

        # Copy images onto the blank canvas at appropriate positions
        copy_outer_image_1[start_y - cv_icon_ctr_y: start_y + cv_icon_ctr_y,
                        start_x - cv_icon_ctr_x: start_x + cv_icon_ctr_x,
                        :] = obj_img_init

        copy_outer_image_2[int((oih - iih) / 2):-int((oih - iih) / 2),
                        int((oiw - iiw) / 2):-int((oiw - iiw) / 2),
                        :] = inst_img

        copy_outer_image_3[goal_y - cv_icon_ctr_y: goal_y + cv_icon_ctr_y,
                        goal_x - cv_icon_ctr_x: goal_x + cv_icon_ctr_x,
                        :] = obj_img_rot

        # Compile visual signals and additional information into a list
        visual_signals = [outer_image, copy_outer_image_1, copy_outer_image_2, copy_outer_image_3, x_offset, y_offset]

        return visual_signals
   
    def raterize_screen_visual_signals(self, obj_vss_rsp, inst_vss_rsp, objects, object_goal_rot, object_goal_poses, object_name=None):
        """
        Rasterizes screen visual signals based on object and instruction visual signal responses.

        Args:
        - obj_vss_rsp (str): Object visual signal response.
        - inst_vss_rsp (str): Instruction visual signal response.
        - objects (list): List of object names.
        - object_goal_rot (list): List of rotation angles for each object.
        - object_goal_poses (list): List of goal poses for each object.
        - object_name (str): Name of the object (default: None).

        Returns:
        - tuple: Tuple containing visual signals and composite image.
        """
        # Rasterize projection visual signals to get base images
        outer_image, copy_outer_image_1, copy_outer_image_2, copy_outer_image_3, x_offset, y_offset = \
            self.raterize_projection_visual_signals(obj_vss_rsp, inst_vss_rsp, object_name=object_name)

        icon_size = ICON_RES
        cv_icon_ctr_x = icon_size[0] // 2
        cv_icon_ctr_y = icon_size[1] // 2

        _outer_image = np.copy(outer_image)

        objects = objects[:-1]  # Remove the last object from the list
        for oid, obj in enumerate(objects):
            tmp_img = np.copy(outer_image)

            # Load object image and resize if necessary
            img = cv2.imread(f"{str(self.svg_images_dir)}/{obj}.jpg")
            if img.shape[0] != icon_size[0]:
                img = cv2.resize(img, icon_size)

            # Apply rotation to the object image if required
            if object_goal_rot[oid] != 0:
                M = cv2.getRotationMatrix2D((cv_icon_ctr_x, cv_icon_ctr_y), -object_goal_rot[oid], 1.0)
                img = cv2.warpAffine(img, M, (icon_size[1], icon_size[0]))

            # Calculate coordinates to place the object image on the screen
            _x = object_goal_poses[oid][0] + x_offset
            _y = object_goal_poses[oid][1] + y_offset

            # Overlay the object image on the screen image
            tmp_img[_y - cv_icon_ctr_y: _y + cv_icon_ctr_y,
                    _x - cv_icon_ctr_x: _x + cv_icon_ctr_x,
                    :] = img

            # Blend the object image with the existing composite image
            _outer_image = cv2.addWeighted(_outer_image, 1, tmp_img, 1, 1.0)

        # Compile visual signals into a list
        visual_signals = [outer_image, copy_outer_image_1, copy_outer_image_2, copy_outer_image_3]

        return visual_signals, _outer_image
 
    def display_signal(self, visual_signals, _outer_image=None):
        """
        Displays the visual signals on the screen or projector.

        Args:
        - visual_signals (list): List containing visual signals.
        - _outer_image (numpy.ndarray): Composite image (default: None).

        Returns:
        - None
        """
        # Initialize alpha values for transparency blending
        alpha1 = 0.0
        alpha2 = 0.0
        alpha3 = 0.0
        delta = 0.08  # Step size for alpha increment

        # Loop until all alpha values reach 1.0
        while alpha1 < 1.0 or alpha2 < 1.0 or alpha3 < 1.0:
            # Generate animated image with varying transparency
            proj_image = animated_image(visual_signals, alpha1, alpha2, alpha3)

            # Display on screen if _outer_image is not provided
            if _outer_image is not None:
                proj_image_ = cv2.addWeighted(np.copy(_outer_image), 1, proj_image, 1, 1.0)
                cv2.namedWindow("screen", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("screen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.moveWindow("screen", 1080, 0)
                cv2.imshow("screen", proj_image_)
            else:
                # Rotate, adjust, and warp image for projection
                proj_image_ = cv2.rotate(proj_image, cv2.ROTATE_180)
                warped = self.project.adjust_and_warp(proj_image_, rotate_c_90=True)
                self.project.show_on_projector(image=warped)
                # cv2.imshow("debug", proj_image)  # Uncomment for debugging

            key = cv2.waitKey(1)  # Wait for a key press
            alpha1 += delta  # Increment alpha values
            alpha2 += delta
            alpha3 += delta


    def display_nli(self, nlis_rsp):
        """
        Displays natural language signals in a window.

        Args:
        - nlis_rsp (str): Natural language interaction response.

        Returns:
        - None
        """
        # Process and split the NLSS response
        text = nlis_rsp.replace("```", "").split("\n")
        filtered_text = ""
        layout = []
        counter = 0

        # Create layout for the window
        for tid, t in enumerate(text):
            if len(t) == 0:
                continue
            t = t.replace("\n", "")
            filtered_text = t + "\n\n"
            txt_size = len(filtered_text)
            if counter == 0:
                layout.append([sg.HSeparator()])
            layout.append([
                sg.Text(text=filtered_text, size=(txt_size, 2), font=('Roboto bold', 25),
                        text_color="white", key="NLI")
            ])
            layout.append([sg.HSeparator()])
            counter += 1

        # Create and display the window
        window = sg.Window(f'NLI (USER: {self.user_name})', layout, grab_anywhere=True, size=(1920, 1080),
                        font=('Roboto bold', 25), finalize=True, location=(1080, 0), resizable=True)

        print("starting loop")
        while True:
            if self.key == "space":  # Close window if 'space' key is pressed
                print(f"closing window...")
                window.close()
                break

        return
    
    def save_time(self, start_time, end_time, start_ts, end_ts, mode, id):
        """
        Saves time-related data to a JSON file.

        Args:
        - start_time (float): Start time of the experiment.
        - end_time (float): End time of the experiment.
        - start_ts (str): Start timestamp in string format.
        - end_ts (str): End timestamp in string format.
        - mode (str): Experiment mode.
        - id (str): Identifier for the data.

        Returns:
        - bool: True if data is successfully saved, False otherwise.
        """
        # Get current timestamp
        ts = get_timestamp()

        # Convert timestamp strings to lists of integers
        start_time_list = [int(t) for t in start_ts.split("_")]
        end_time_list = [int(t) for t in end_ts.split("_")]

        # Create file name with timestamp, mode, and id
        file_name = self.user_dir / f"{ts}_{mode}_{id}.json"

        # Prepare data for JSON file
        data = {
            "start_time": start_time,
            "end_time": end_time,
            "start_time_list": start_time_list,
            "end_time_list": end_time_list
        }

        # Write data to JSON file
        try:
            with open(file_name, "w", encoding='utf-8') as fh:
                json.dump(data, fh, indent=4)
            return True  # Return True if data is successfully saved
        except Exception as e:
            print(f"Error saving data: {e}")
        return False  # Return False if an error occurs during saving
         

    def main(self, skip_seq=7):
        """
        Main function to run the experiment.

        Args:
        - skip_seq (int): Number of sequences to skip (default: 7).

        Returns:
        - None
        """
        if not self.debug:
            self.go_home_v2(type=1)  # Move to home position

        # Iterate through experiment sequences
        for seq_id, exp_args in enumerate(zip(self.modes, self.structures), 1):
            mode, structure = exp_args

            if seq_id <= skip_seq:
                print(f"skipping structure: {structure}")
                continue

            print("press \"s\" to start experiment!...")
            while self.key != "s":  # Wait for 's' key press to start the experiment
                continue

            print(f"\n[MODE]: {mode}")

            # Get waypoints and other data for the experiment
            start_waypoints, end_waypoints, object_desc, object_color, skip_id, orientation, instructions, goal_poses = self.get_waypoints(shape=structure)
            objects = []
            object_goal_poses = []
            object_goal_rot = []

            # Iterate through waypoints and perform actions
            for i, waypoint in enumerate(zip(start_waypoints, end_waypoints)):
                start_waypoint_, end_waypoint = waypoint
                start_waypoint = start_waypoint_[0]
                objects.append(start_waypoint_[1])
                object_goal_poses.append(goal_poses[i][0])
                object_goal_rot.append(goal_poses[i][1][1])
                goal_object_to_pick = start_waypoint_[1]

                print(f"picking up: {goal_object_to_pick}")

                if skip_id == i:
                    # Start recording images for this sequence
                    self.img_dir = self.user_dir / f"images_{mode}_{seq_id}"
                    self.start_saving_imsgs = True

                    if not self.debug:
                        self.go_home_v2(type=2)  # Move to a specific position

                    self.start_time = time.time_ns()  # Record start time
                    self.start_ts = get_timestamp()  # Record start timestamp

                    properties = {
                        "structure": structure,
                        "object_description": object_desc,
                        "object_color": object_color,
                        "orientation": orientation[0],
                        "instruction": instructions,
                        "goal_position": goal_poses[i][0]
                    }

                    # Synthesize signals based on the properties
                    obj_vss_rsp, inst_vss_rsp, nlis_rsp = self.synthesize_signals(properties=properties, temperature=0.0)
                    self.key = None

                    if mode == "VSIntPro":
                        visual_signals = self.raterize_projection_visual_signals(obj_vss_rsp, inst_vss_rsp, object_name=object_desc)
                        while True:
                            self.display_signal(visual_signals[:-2])  # Display visual signals
                            if self.key == "space":  # Break loop if 'space' key is pressed
                                cv2.destroyAllWindows()
                                break
                            rospy.sleep(1)

                    elif mode == "VSM":
                        visual_signals, _outer_image = self.raterize_screen_visual_signals(obj_vss_rsp, inst_vss_rsp, objects, object_goal_rot, object_goal_poses,
                                                                                        object_name=object_desc)
                        while True:
                            self.display_signal(visual_signals, _outer_image)  # Display visual signals
                            if self.key == "space":  # Break loop if 'space' key is pressed
                                cv2.destroyAllWindows()
                                break
                            rospy.sleep(1)

                    elif mode == "NLS":
                        self.display_nli(nlis_rsp)  # Display NLI instructions

                    # Update end time if necessary
                    if self.end_time < self.start_time:
                        self.end_time = time.time_ns()
                        self.end_ts = get_timestamp()

                    # Save time-related data
                    self.save_time(start_time=self.start_time, end_time=self.end_time,
                                start_ts=self.start_ts, end_ts=self.end_ts, mode=mode, id=seq_id)
                    self.start_saving_imsgs = False  # Stop recording images
                    continue

                # Skip execution for debugging mode
                if self.debug:
                    continue

                # Move the robot through waypoints
                for wid, w in enumerate(start_waypoint):
                    if wid == 3:
                        break
                    goal_joints = self.get_ik_sol(c_joints=self.ur5Joints.positions, xyz=w[:3], rpy=w[3:6])
                    self.movej_msg(goal_joints=goal_joints, t=w[6])
                    if bool(w[7]) and not self.move_only:
                        self.r2fg_msg.position = self.max_r2fg
                    else:
                        self.r2fg_msg.position = 0
                    self.ur5_joint_publisher.publish(self.ur5_control_msg)
                    self.r2fg_publisher.publish(self.r2fg_msg)
                    rospy.sleep(w[6])

                for wid, w in enumerate(end_waypoint):
                    if wid == 3:
                        break
                    goal_joints = self.get_ik_sol(c_joints=self.ur5Joints.positions, xyz=w[:3], rpy=w[3:6])
                    self.movej_msg(goal_joints=goal_joints, t=w[6])
                    if bool(w[7]) and not self.move_only:
                        self.r2fg_msg.position = self.max_r2fg
                    else:
                        self.r2fg_msg.position = 50

                    self.ur5_joint_publisher.publish(self.ur5_control_msg)
                    self.r2fg_publisher.publish(self.r2fg_msg)
                    rospy.sleep(w[6])

            if not self.debug:
                self.go_home_v2(type=1)  # Move to home position at the end of the sequence

        print(f"Finished experiment!!!!!!!!!!")
        return
    
    def end_thread(self):
        # kill threads
        print("Wait... killing threads. or press \"CTRL + \\\" to brutally kill the process.")
        self.kill_thread_event.set()
        self.keyboard.press(keyboard.Key.enter)
        self.keyboard.release(keyboard.Key.enter)
        self.keyboard_thread.join()
        self.cap_th.join()
        return 

if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_seq", type=int, default=0, help="Number of structures to skip")
    parser.add_argument("--debug", type=int, default=0, help="Debug mode: 0 - Off, 1 - On")
    parser.add_argument("--debug_mode", type=int, default=1, help="Debug mode options: 1 - NLS, 2 - VSIntPro, 3 - VSM")
    parser.add_argument("--move_only", type=int, default=0, help="Move robot but don't grab objects")
    parser.add_argument("--same_user", type=int, default=0, help="Load the same user")
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Create an instance of the EXPERIMENT class with parsed arguments
    exp = EXPERIMENT(debug=bool(args.debug), debug_mode=args.debug_mode, move_only=bool(args.move_only), same_user=bool(args.same_user))
    
    # Run the main experiment method with parsed arguments
    exp.main(skip_seq=args.skip_seq)
    
    # Clean up resources or end threads if necessary
    exp.end_thread()