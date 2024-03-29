


import os
import sys
import openai 
import cv2
import base64
import requests
import numpy as np
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
from vertexai.language_models import ChatModel, InputOutputTextPair
from openai import OpenAI

from datetime import datetime
from utils import *
from config import *

openai_api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = openai_api_key
headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
           }        

class LLM_UTILS():
    """
    LLM_UTILS is a utility class for working with Language Learning Models (LLMs) like GPT models.

    Attributes:
        client: An instance of the OpenAI API client.
        openai_models: A dictionary mapping model names to their respective OpenAI model identifiers.
        model: The active GPT model being used.
        openai_messages: A list that contains default messages for initializing a systemic role for the assistant.
        m: A hashlib.sha256 object for creating cryptographic hashes, possibly for caching purposes.
        cache_dir: The directory where cached items are to be stored.
        cache: A list of filenames currently in the cache directory.
    """
    
    def __init__(self, model='4.0-turbo', cache_dir=None):
        """
        Initializes the LLM_UTILS class instance with defaults or provided parameters.

        Args:
            model (str): The predefined model identifier to select the initial model.
            cache_dir (str or None): The path to the cache directory. If None, defaults to 
                                     the 'cache_dir' value specified in config.PATHS.
        """        
        # Create an instance of the OpenAI API client.
        self.client = OpenAI()
        
        # Set up a dictionary mapping friendly names to OpenAI model identifiers.
        self.openai_models = {
            "3.5"              : "gpt-3.5-turbo-16k",
            "3.5-turbo"        : "gpt-3.5-turbo-0125", 
            "4.0"              : "gpt-4", 
            "4.0-32k"          : "gpt-4-32k-0613",
            "4.0-turbo"        : "gpt-4-1106-preview",
            "4.0-turbo-vision" : "gpt-4-vision-preview"
        }
        
        # Initialize the selected model based on the provided model name.
        self.model = self.set_model(model)
        
        # Seed system role with a creative and intelligent persona.
        self.openai_messages = [{"role": "system", "content": "You are a creative and intelligent assistant."}]
        
        # A hashlib object for hashing, which may be used for caching mechanisms.
        self.m = hashlib.sha256()
        
        # Set up cache directory and contents.
        self.cache_dir = cache_dir or config.PATHS["cache_dir"]
        self.cache = os.listdir(self.cache_dir)
        return 

    def set_model(self,model):
        return self.openai_models.get(model)


    def call_model_response(self, messages, n=1, temperature=0.0, max_tokens=1000):
        """
        Calls the model for a response using provided messages and other parameters. 
        Responses are either retrieved from a cache or generated by the model if not cached.

        Args:
            messages (list): A list of messages formatted as expected by the OpenAI API.
            n (int): The number of completions to generate for each prompt.
            temperature (float): Controls the randomness of the response generated by the model.
            max_tokens (int): The maximum number of tokens to generate in total for the response (unused in this function).

        Returns:
            dict or list: The model's response, depending on what the OpenAI API client returns.
        """
        # Debugging statement to show which model and temperature are being used for the API call.
        print(f"using model: {self.model} and temp: {temperature}")
        
        # Concatenate the model, temperature, and messages into a single string to create a unique cache key.
        content = f"model: {self.model}\ntemperature: {temperature}\nmessages: {messages}"
        
        # Compute a SHA-256 hash of the content string, to be used as a cache key.
        hashed = hashlib.sha256(content.encode('utf-8')).hexdigest()
        
        # Get the cache directory from the configuration.
        cache_dir = config.PATHS["cache_dir"]
        
        # List all available cache files in the cache directory.
        avail_cache = os.listdir(cache_dir)
        
        # Check if the generated hash key is in the cache.
        if hashed in avail_cache:
            # If the content is already cached, retrieve the cached response.
            # print(f"item available in cache")  # Uncomment for debugging
            content, response = get_cache_item(hashed)  # Assumes existence of 'get_cache_item' function
        else:
            # If the content is not in the cache, add a new item.
            print(f"new item added to the cache")
            # Make an API call to generate a response using the client's chat completion method.
            response = self.client.chat.completions.create(model=self.model,
                                                           messages=messages,
                                                           n=n,
                                                           temperature=temperature)
            
            # Store the new response in the cache for future use.
            write_cache_item(hashed, [content, response])  # Assumes existence of 'write_cache_item' function
        
        # Return the response from the model or cache.
        return response                 

    def call_gemini_response(self, project_id: str = "intpro-research", location: str = "us-central1", 
                             temperature: float = 0.0,
                             model = "gemini-pro", messages = "", max_tokens=16000) -> str:
        
        # chat_model = ChatModel.from_pretrained("chat-bison-32k")
        chat_model = GenerativeModel(model)
        parameters = {
            "temperature": temperature,  # Temperature controls the degree of randomness in token selection.
            "max_output_tokens": 8000,  # Token limit determines the maximum amount of text output.
            "top_p": 0.95,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
            "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
        }

        # vertexai.init(project=project_id, location=location)
        # history=[f"""You are a creative and intelligent assistant. \n{self.prefix}\n{self.suffix}"""]
        chat = chat_model.start_chat()
        response = chat.send_message(
                # f"""You are a creative and intelligent assistant. \n{self.prefix} + f"\n The task is:\n{messages}\n" + {VSS_LLM_SUFFIX}"""
                messages[0].get("content")
        )
        
        return response.text

    def generate_openai_message(self, prompt, printout : bool = True):
        messages = [{"role": "user", "content": prompt}]
        if printout:
            print(f"input to gpt:\n{prompt}")
        return messages

    def generate_gemini_message(self, message = ""):
        messages = [{"content": message}]
        return messages



if __name__ == "__main__":
    llms = LLM_UTILS()
    from context4task import firstExperimentTaskDescription
    from core import *
    properties = {"structure"          : "S",
                  "object_description" : "Bunny",
                  "object_color"       : "Orange",
                  "orientation"        : "upright",
                  "instruction"        : "some zig zags from bottom to top",
                  "goal_position"      : [500,100]}
    task_context = firstExperimentTaskDescription(structure          = properties.get("structure"),
                                                  object_description = properties.get("object_description"),
                                                  object_color       = properties.get("object_color"),
                                                  orientation        = properties.get("orientation"),
                                                  instruction        = properties.get("instruction"),
                                                  goal_position      = properties.get("goal_position"))
    tm_prompt = generate_task_master_prompt(task_context)
    tm_gemini_msg = llms.generate_gemini_message(message=tm_prompt)
    response = llms.call_gemini_response(messages=tm_gemini_msg)

    obj_vss_prompt = generate_obj_vss_prompt(response=response)
    obj_gemini_msg = llms.generate_gemini_message(message=obj_vss_prompt)
    response = llms.call_gemini_response(messages=obj_gemini_msg)

    print(f"gemini response of object svg code: \n{response}")