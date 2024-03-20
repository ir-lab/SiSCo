import os
import sys
import openai
import cv2
from openai import OpenAI
from threading import Thread
import time
from context.context4task import *
import argparse
from utils import *
from config import ICON_RES
    
def generate_task_master_prompt(task_description):
    """
    Generates a prompt for a 'Task Master' to coordinate tasks between humans and robots,
    providing guidance on how to utilize available tools for communication.

    The prompt describes the scenario where a robot assembling a structure has malfunctioned
    and communication is needed to convey critical instructions. This is achieved using two
    VSS functions and one NLIS function, which synthesize visual signals and natural language
    instructions, respectively.

    Parameters:
    - task_description (str): The raw detailed task description provided to the Task Master.

    Returns:
    - str: A structured prompt including the task scenario, guidelines for using synthesis tools,
           and the task description encapsulated within specific markers.
    """

    # Define the prefix explaining the scenario and the available tools for communication.
    prefix = (f"You are an expert in coordinating human-robot teaming tasks. A robot is assembling "
              f"a structure and malfunctioned. To help the robot, you must communicate relevant "
              f"information to the human through visual signals and natural language. You have access "
              f"to two Visual Signal Synthesizer (VSS) functions and a Natural Language Instruction "
              f"Synthesizer (NLIS) function. The VSS functions take in detailed text prompts and "
              f"provide the code for descriptive scalable vector graphics (SVG) and additional "
              f"information as output. The individual VSS functions are described as follows:\n\n"
              f"1) ObjectVSS: This VSS function expects only the object color and description as inputs "
              f"and will generate an SVG icon.\n\n"
              f"2) InstructionVSS: This VSS function expects details on the environment size, object "
              f"movement instructions, 2D goal position, and object orientation as inputs, producing "
              f"an environmental SVG, the starting point for object movement, and an object rotation angle.\n\n"
              f"3) NLIS: This language synthesizer function requires the whole task description to produce "
              f"four bullet points summarizing the task.\n\n"
              f"We provide you with a Task Description. Extract the relevant information from the task "
              f"description and assign it to the appropriate VSS and NLIS functions. Use all the "
              f"information provided in the task description.\n\n"
              f"[TaskDescription]")

    # Define the suffix which structures the inclusion of the task description and provides an example.
    suffix = (f"[/TaskDescription]\n\n"
              f"Here is an example output:\n"
              f"'''\n"
              f"[ObjectVSS]\n"
              f"Prompt = Generate an icon for an object with the description 'race car' and of purple color.\n"
              f"[/ObjectVSS]\n\n"
              f"[InstructionVSS]\n"
              f"Prompt = The size of the 2D environment is 1000x500. Depict a trajectory to slide the "
              f"object from the left toward the goal position [x,y]. The object must have the orientation 90 deg.\n"
              f"[/InstructionVSS]\n\n"
              f"[NLIS]\n"
              f"Prompt = The environment has a width of 1000 and a height of 500. A robot is assembling "
              f"an O-shaped structure around the center of the environment using objects. Each object has "
              f"a description and a color. The robot malfunctioned and failed to place the object described "
              f"as 'race car' and colored purple in the environment. It must be placed with 90 deg orientation "
              f"at [x,y] by following the instruction to slide in from the left.\n"
              f"[/NLIS]\n"
              f"'''\n\n"
              f"Strictly follow the output format as shown above but you can be creative with the prompt generation.")

    # Combine prefix, task description, and suffix to create the final structured prompt.
    return f"{prefix}\n{task_description}\n{suffix}"


def generate_obj_vss_prompt(response):
    """
    Generates a prompt for a Visual Signaling Synthesizer (VSS) based on the provided response.
    
    This function extracts a section of the response that is marked by custom tags [ObjectVSS] and [/ObjectVSS]
    and prepares a complete prompt including a description of the task and requirements for the generated SVG.

    Parameters:
    - response (str): A string containing the task description enclosed within [ObjectVSS] tags.

    Returns:
    - str: A formatted string that includes the task context, the extracted description, and the requirements for the SVG.
    """
    # Define a regular expression pattern to extract content within [ObjectVSS] tags
    pattern = r"\[ObjectVSS\](.*?)\[/ObjectVSS\]"
    # Search for the pattern in the response text and return matches as a list
    prompt = re.findall(pattern, response, re.DOTALL)
    
    # Check if the prompt extraction was unsuccessful (list is empty)
    if len(prompt) == 0:
        pattern = r"\[ObjectVSS\](.*?)"  # Fallback pattern without closing tag
        prompt = re.findall(pattern, response, re.DOTALL)[0]  # Assume a single, possibly incomplete section
    else:
        prompt = prompt[0].replace("Prompt = ", "")  # Remove the "Prompt = " part if present
    
    # Define the prefix explaining the role of the VSS and its task
    prefix = ("You are a Visual Signaling Synthesizer (VSS), which produces scalable "
              "vector graphics (SVG) in text form. You design visual signals to help a "
              "human identify target objects. Generate an SVG that describes the following input:")
    
    # Define the suffix with SVG requirements
    suffix = (f"The SVG size must be {ICON_RES[0]}x{ICON_RES[1]}. Avoid 3D shapes. The object is at the center. "
              "The object is in vertical orientation facing up. The object is clearly visible against the black "
              "background. Be creative with object color shades.")
    
    # Return the full prompt concatenating prefix, extracted prompt, and suffix
    return f"{prefix}\n{prompt}\n{suffix}"

def generate_nlis_prompt(response):
    """
    Generates a prompt that can be used with a Natural Language Instructions Synthesizer (NLIS) model.
    
    The function extracts an instruction from the response input, formatting it into a structured prompt
    for assembling a structure in a robotics context. The prompt includes guidelines on how to describe
    the position of objects without using numbers.
    
    Parameters:
    - response (str): The raw input string containing the task description, potentially enclosed
                      within [NLIS] and [/NLIS] tags.
                      
    Returns:
    - str: A formatted string that combines an instruction prefix, the task description extracted from
           the input `response`, and a suffix containing an example of the expected output.
    """

    # Define the primary pattern to extract the NLIS instructions.
    pattern = r"\[NLIS\](.*?)\[/NLIS\]"
    # Find all occurrences of the pattern in the response, considering newlines (DOTALL flag).
    prompt = re.findall(pattern, response, re.DOTALL)

    # Check if no prompt was found with the primary pattern.
    if len(prompt) == 0:
        # Define an alternative pattern to extract NLIS instructions.
        # This pattern assumes that the closing [/NLIS] tag might be missing.
        pattern = r"\[NLIS\](.*?)"
        # Extract the first match of the alternate pattern.
        prompt = re.findall(pattern, response, re.DOTALL)[0]
    else:
        # If there was a match, take the first one and remove the "Prompt = " string if present.
        prompt = prompt[0].replace("Prompt = ", "")
    
    # Define the opening part of the NLIS prompt, giving context and instructions.
    prefix = (f"You are a Natural Language Instructions Synthesizer (NLIS), which summarizes a task description "
              f"into four bullet points. All tasks encompass that a robot assembles a structure. It malfunctions and "
              f"needs help to place an object in the environment. The position [0,0] denotes the left top. "
              f"The center of the structure is at [496, 262]. You cannot use numbers to describe object positions. "
              f"Instead, describe the x-position as at the [left xor middle xor right] of the structure and the "
              f"y-position as at the [top xor center xor bottom] of the structure. Positions above the center are at "
              f"the top and positions below the center are at the bottom of the structure.\n\n"
              f"Summarize this task description: ")

    # Define the closing part of the NLIS prompt, including the expected output format example.
    suffix = (f"You strictly provide four bullet points similar to this example output:\n"
              f"```\n"
              f"* The robot assembles ...\n"
              f"* The robot malfunctioned when trying to place a purple race car.\n"
              f"* Place the race car at the right top of the structure in 90 deg orientation from vertical.\n"
              f"* Make sure to slide the car from the left into its position.\n"
              f"```\n")

    # Combine the prefix, the task description, and the suffix into the final prompt.
    return f"{prefix}\n{prompt}\n{suffix}"



def generate_inst_vss_prompt(response):
    """
    Generates a prompt for an Instruction Visual Signaling Synthesizer (InstVSS).
    
    This function extracts instructions within [InstructionVSS] tags from the given response
    and formats those instructions into a prompt to be processed by the VSS for generating SVG output.
    
    Parameters:
    - response (str): A string containing the instruction details to be visualized, enclosed within [InstructionVSS] tags.
    
    Returns:
    - str: A formatted string prompt that includes conditions for SVG creation, the extracted instruction details, and an example structure of expected output.
    """
    
    # Define patterns for extracting content within [InstructionVSS] tags
    pattern = r"\[InstructionVSS\](.*?)\[/InstructionVSS\]"
    prompts = re.findall(pattern, response, re.DOTALL)
    
    # Check if no instructions are found with closing tag, use a fallback pattern without closing tag
    if len(prompts) == 0:
        pattern = r"\[InstructionVSS\](.*?)"
        prompts = re.findall(pattern, response, re.DOTALL)
        # Protect against an empty list (no matches), which would raise an IndexError
        prompt = prompts[0] if prompts else ""
    else:
        # Clean up the found prompt by removing an unnecessary prefix if present
        prompt = prompts[0].replace("Prompt = ", "")
    
    # Define prefix providing context and constraints for VSS operation
    prefix = ("You are a Visual Signaling Synthesizer (VSS), which provides visual signals as scalable vector "
              "graphics (SVG) in text form. Your task is to draw clear instructions for moving an object in a 2D "
              "environment. Coordinates are represented as [x,y], with [0,0] being the top left corner. You produce "
              "three things:\n"
              "1) The SVG depicting the trajectory in the environment\n"
              "2) The start and goal positions\n"
              "3) The orientation in degrees clockwise from vertical\n\n"
              "Produce these for the following task:")
    
    # Define suffix that outlines SVG requirements and output structure
    suffix = ("Indicate movement direction clearly. If an arrow is used, its tip must precisely point to the goal "
              "position. The start and goal positions should be within 200 units of each other. Trajectories should "
              "be indicated with a minimum line thickness of 5. The background of the SVG must be black. "
              "Strictly follow this example output structure: \n"
              "```\n"
              "<!-- 1) the SVG -->\n"
              "<svg>\n"
              "<!-- Background -->\n"
              "<!-- Trajectory -->\n"
              "</svg>\n\n"
              "<!-- 2) positions x,y -->\n"
              "<start_position>[ , ]</start_position>\n"
              "<goal_position>[ , ]</goal_position>\n\n"
              "<!-- 3) orientation in degrees -->\n"
              "<orientation>0</orientation>\n"
              "```")
    
    # Combine the prefix, prompt, and suffix for the final structured output
    return f"{prefix}\n{prompt}\n{suffix}"
