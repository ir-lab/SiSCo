import os
import sys
import rospy
import PySimpleGUI as sg
import time
from pathlib import Path
import numpy as np
from threading import Thread
import ctypes
import json
import argparse


from context.context4task import *
from context.core import *
from llm_utils import LLM_UTILS

# Define a class for SiSCo_Vis
class SiSCo_Vis:
    def __init__(self, model="4.0-turbo") -> None:
        """
        Initialize SiSCo_Vis class.

        Parameters:
        - model (str): The model to use. Defaults to "4.0-turbo".
        """
        # Initialize font settings
        self.font = 'Roboto'
        self.font_size = 20
        self.text_size = 20
        self.input_text_size = 50
        
        # Flag to control the display
        self.kill_display = False
        
        # Initialize LLM_UTILS object with the specified model
        self.llms = LLM_UTILS(model=model)

    def get_sg_font(self):
        """
        Get the font and font size used for PySimpleGUI elements.

        Returns:
        - tuple: A tuple containing the font name and font size.
        """
        return (self.font, self.font_size)

    def get_sg_text(self, text, size=None, justification='left'):
        """
        Get a PySimpleGUI Text element with specified text, size, and justification.

        Parameters:
        - text (str): The text to display in the Text element.
        - size (tuple or None): The size of the Text element. Defaults to None, which uses the class's default text size.
        - justification (str): The justification of the text within the element. Defaults to 'left'.

        Returns:
        - sg.Text: A PySimpleGUI Text element with the specified properties.
        """
        if size is None:
            size = self.text_size
        return sg.Text(text, font=self.get_sg_font(), size=size, justification=justification)

    def get_sg_input_text(self, key, size=None):
        """
        Get a PySimpleGUI Input element with specified key and size.

        Parameters:
        - key (str): The key of the Input element.
        - size (tuple or None): The size of the Input element. Defaults to None, which uses the class's default input text size.

        Returns:
        - sg.Input: A PySimpleGUI Input element with the specified properties.
        """
        if size is None:
            size = self.input_text_size
        return sg.Input(key=key, size=size)

    def get_sg_image(self, img_path, expand_x=False, expand_y=False):
        """
        Get a PySimpleGUI Image element with specified image path and expansion settings.

        Parameters:
        - img_path (str or Path): The path to the image file.
        - expand_x (bool): Whether to expand the image horizontally. Defaults to False.
        - expand_y (bool): Whether to expand the image vertically. Defaults to False.

        Returns:
        - sg.Image: A PySimpleGUI Image element with the specified properties.
        """
        if not isinstance(img_path, str):
            img_path = str(img_path)
        return sg.Image(img_path, expand_x=expand_x, expand_y=expand_y)


    def display_signal(self, obj_vss_rsp, inst_vss_rsp, only_icon=False):
        """
        Display visual signals using OpenCV.

        Parameters:
        - obj_vss_rsp (str): Object visual signal response.
        - inst_vss_rsp (str): Instruction visual signal response.
        - only_icon (bool): Whether to display only the icon without animation. Defaults to False.

        Returns:
        - None
        """
        # Convert SVG codes to images
        inst_img = svg_to_cv(svg_code=get_svg_files(inst_vss_rsp)[0], show=False)
        (start_x, start_y), (goal_x, goal_y), ori = get_start_goal_orientation(inst_vss_rsp)

        # Define transformations according to icon_size
        icon_size = ICON_RES
        cv_icon_ctr_x = icon_size[0] // 2
        cv_icon_ctr_y = icon_size[1] // 2

        # Create object icon and rotated object icon
        obj_img_init = svg_to_cv(svg_code=get_svg_files(obj_vss_rsp)[0], show=False)
        M = cv2.getRotationMatrix2D((cv_icon_ctr_x, cv_icon_ctr_y), -ori, 1.0)
        obj_img_rot = cv2.warpAffine(obj_img_init, M, (icon_size[1], icon_size[0]))

        iih, iiw, _ = inst_img.shape
        proj_x = 1.6  # meters
        proj_y = 0.8  # meters

        outer_image = np.ones((int(1000 * proj_y), int(1000 * proj_x), 3), dtype=np.uint8) * 0
        copy_outer_image_1 = np.copy(outer_image)
        copy_outer_image_2 = np.copy(outer_image)
        copy_outer_image_3 = np.copy(outer_image)
        copy_outer_image_4 = np.copy(outer_image)

        oih, oiw, _ = outer_image.shape
        x_offset = int((outer_image.shape[1] - inst_img.shape[1]) / 2)
        y_offset = int((outer_image.shape[0] - inst_img.shape[0]) / 2)

        start_x = start_x + x_offset
        start_y = start_y + y_offset
        goal_x = goal_x + x_offset
        goal_y = goal_y + y_offset

        copy_outer_image_1[start_y - cv_icon_ctr_y: start_y + cv_icon_ctr_y,
        start_x - cv_icon_ctr_x: start_x + cv_icon_ctr_x,
        :] += obj_img_init

        copy_outer_image_2[int((oih - iih) / 2):-int((oih - iih) / 2),
        int((oiw - iiw) / 2):-int((oiw - iiw) / 2),
        :] += inst_img

        copy_outer_image_3[goal_y - cv_icon_ctr_y: goal_y + cv_icon_ctr_y,
        goal_x - cv_icon_ctr_x: goal_x + cv_icon_ctr_x,
        :] += obj_img_rot

        copy_outer_image_4[goal_y - cv_icon_ctr_y: goal_y + cv_icon_ctr_y,
        goal_x - cv_icon_ctr_x: goal_x + cv_icon_ctr_x,
        :] += obj_img_init

        counter = 0
        alpha1 = 0.0
        alpha2 = 0.0
        alpha3 = 0.0
        delta = 0.08

        while counter < 1000:
            print("press \"q\" to stop the visualization..")
            if only_icon:
                cv2.imshow("visualize", copy_outer_image_4)
                key = cv2.waitKey(1)
                if key == ord('q') or self.kill_display:
                    counter = 100
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                    self.kill_display = False
                    return
                
            else:
                # Animation Starts
                while alpha1 < 1.0 or alpha2 < 1.0 or alpha3 < 1.0:
                    visual_signals = [outer_image, copy_outer_image_1, copy_outer_image_2, copy_outer_image_3]
                    proj_image = animated_image(visual_signals, alpha1, alpha2, alpha3)
                    proj_image_ = cv2.rotate(proj_image, cv2.ROTATE_180)
                    cv2.imshow("visualize", proj_image)
                    key = cv2.waitKey(1)
                    alpha1 += delta
                    alpha2 += delta
                    alpha3 += delta               
                    if key == ord('q') or self.kill_display:
                        counter = 100
                        cv2.destroyAllWindows()
                        cv2.waitKey(1)
                        self.kill_display = False
                        return
                alpha1 = 0.0
                alpha2 = 0.0
                alpha3 = 0.0
                counter += 1
            time.sleep(1)
        return
    
    def synthesize_signals(self, properties, temperature=0.0, max_token=16000, obj_=True, inst_=True, nli_=False):
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

    def show_signal(self, properties, temperature=0.0):
        """
        Synthesize visual signals based on properties and display them.

        Parameters:
        - properties (dict): Properties for synthesizing visual signals.
        - temperature (float): Temperature for model response, default is 0.0.

        Returns:
        - None
        """
        # Synthesize object and instruction visual signals
        obj_vss_rsp, inst_vss_rsp, _ = self.synthesize_signals(properties=properties, temperature=temperature)
        
        # Reset kill_display flag
        self.kill_display = False
        
        # Display the visual signals
        self.display_signal(obj_vss_rsp, inst_vss_rsp) 
        return


    def vis(self):
        """
        Display a GUI window to generate and visualize a unique visual signal.

        Returns:
        - None
        """
        # Set default temperature
        temperature = 0.0

        # Example input column
        col_exp = [[self.get_sg_text('Properties',size=100)],
                   [sg.HSeparator()],
                   [self.get_sg_text('Object Description', size=15)  , sg.VSeparator(), sg.Input('Carrots',disabled= True)],
                   [self.get_sg_text('Color', size = 15)             , sg.VSeparator(), sg.Input('Orange',disabled= True)],
                   [self.get_sg_text('Orientation', size = 15)       , sg.VSeparator(), sg.Input('Vertical',disabled= True)],
                   [self.get_sg_text('Instruction', size = 15)       , sg.VSeparator(), sg.Input('Insert from bottom-right', disabled= True)],
                   ]

        # Define the layout of the GUI window
        col_user = [
            [self.get_sg_text('Properties',size=100)],
            [sg.HSeparator()],
            [self.get_sg_text('Object Description', size=15), sg.VSeparator(), sg.Input(key='Object_Description')],
            [self.get_sg_text('Color', size=15), sg.VSeparator(), sg.Input(key='Color')],
            [self.get_sg_text('Orientation', size=15), sg.VSeparator(), sg.Input(key='Orientation')],
            [self.get_sg_text('Instruction', size=15), sg.VSeparator(), sg.Input(key='Instruction')],
            [sg.HSeparator()],
            [sg.Button("Generate Visual Signal (window will close once button is pressed)", key="VSS")],
            [sg.Text("Note: After clicking the button above, please wait for 20 seconds.", font=("Arial", 20),
                     text_color="blue")],
            [sg.HSeparator()],
            [sg.Button("OK", key="OK")]
        ]

        layout = [
            [self.get_sg_text(f"\nHere, you can create your own unique visual signal.", size=(100, 2))],
            [sg.HSeparator()],
            [self.get_sg_text("Here is example input!", size=100)],
            col_exp,
            [sg.HSeparator()],
            [self.get_sg_text("Provide your input below. It could be anything!", size=100)],
            col_user
        ]

        # Create the Window
        window = sg.Window(f'SiSCo Visualization', layout, font=self.get_sg_font(), size=(1200, 800), finalize=True,
                            location=(0, 0), keep_on_top=True, resizable=True)
        window[f'VSS'].bind("<Button>", "_Button")
        flag = False

        # Event Loop to process "events" and get the "values" of the inputs
        while True:
            event, sg_values = window.read()

            if event == "VSS_Button":
                # Prepare properties based on user input
                properties = {
                    "structure": " ",
                    "object_description": sg_values.get("Object_Description"),
                    "object_color": sg_values.get("Color"),
                    "orientation": sg_values.get("Orientation"),
                    "instruction": sg_values.get("Instruction"),
                    "goal_position": [500, 250]
                }

                # Start a new thread to show the visual signal
                if not flag:
                    print(f"\nWait while LLM synthesize the signal..\n")
                    th = Thread(target=self.show_signal, args=(properties, temperature))
                    th.start()
                    flag = True
                    break

            if event == "OK":
                break

        window.close()
        return


if __name__ == "__main__":
    # Create an instance of SiSCo_Vis
    sisco_vis = SiSCo_Vis()

    # Call the vis method to display the GUI window
    sisco_vis.vis()