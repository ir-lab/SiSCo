import os
import sys


def firstExperimentTaskDescription(structure="",
                                   object_description="",
                                   object_color="",
                                   goal_position="",
                                   orientation="",
                                   instruction=""):
    """
    Generates a task description for an experiment where a robot must place 
    an object within a specified environment. The description includes details about 
    the environment dimensions, the structure being assembled, object traits, 
    and placement instructions.

    Each parameter provided to this function will be included in the output string
    to tailor the task description to specific experiment scenarios.

    Args:
        structure (str): The shape of the structure being assembled by the robot.
        object_description (str): A description of the object that needs to be placed.
        object_color (str): The color of the object to be placed.
        goal_position (str): The target position where the object should be placed.
        orientation (str): The required orientation of the object when placed.
        instruction (str): Additional instructions for how the object should be placed.

    Returns:
        str: A formatted string describing the task environment, the malfunction event,
             and the specific instructions for placing the object.
    """
    # Create a task description string, filling in placeholders with the provided parameter values.
    desc = f"The environment has a width of 1000 and a height of 500. A robot is assembling an {structure}-shaped structure around the center of the environment using objects. Every object has an object description and an object color. The robot has malfunctioned and failed to place the object with the description {object_description} and color {object_color} in the environment. It must be placed in {orientation} orientation at {goal_position}. It must be placed by following the instruction {instruction}."
    
    # Return the completed description string.
    return desc