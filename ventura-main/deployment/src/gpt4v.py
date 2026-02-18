import os
import io
import base64
import requests
from typing import List, Tuple
import numpy as np
import cv2
import json
from PIL import Image

from abc import ABC, abstractmethod
from typing import List, Tuple

class Baseline(ABC):
    def __init__(self, use_cot: bool = False):
        """
        Initialize the baseline model.
        This method should be implemented in each child class.
        """
        super().__init__()
        self.use_cot: bool = use_cot
        # past conversation example: [('user', 'Hello!'), ('assistant', 'Hi!')]
        self.past_conversations: List[Tuple[str, str]] = []

    def generate_text(self, text: str, image_filepaths: List[str] = []):
        """
        Generate text based on the provided text and optionally an image.

        :param text: The input text to process.
        :param image_filepaths: The optional input image to process.
        :return: The generated text.
        """
        if self.use_cot:
            return self.generate_text_using_past_conversations(text, image_filepaths)
        else:
            return self.generate_text_individual(text, image_filepaths)

    @abstractmethod
    def generate_text_individual(self, text: str, image_filepaths: List[str] = []):
        """
        Generate text based on the provided text and images.

        :param text: The input text to process.
        :param image_filepaths: The optional input image to process.
        :return: The generated text.
        """
        pass

    @abstractmethod
    def generate_text_using_past_conversations(
        self, text: str, image_filepaths: List[str] = []
    ):
        """
        Generate text based on the provided text, images, and past conversations.

        :param text: The input text to process.
        :param image_filepaths: The optional input image to process.
        :return: The generated text.
        """
        pass

    def add_to_conversation_history(self, conversation: Tuple[str, str]):
        """
        Set the past text to be used in generating the next response.
        This is primarily used when using chain-of-thought with ground truth.

        :param text: The past text to set.
        """
        self.past_conversations += [conversation]

    def clear_conversation_history(self):
        """
        Clear the conversation history.
        """
        self.past_conversations = []


class APIBaseline(Baseline):
    def __init__(self, api_key_env_var: str):
        """
        Initialize the baseline model.
        This method should be implemented in each child class.
        """
        super().__init__(api_key_env_var)
        self.api_key = self.load_api_key(api_key_env_var)

    def load_api_key(self, api_key_env_var: str):
        """
        Load the API key from an environment variable.

        :param api_key_env_var: The environment variable name where the API key is stored.
        :return: The API key as a string.
        :raises EnvironmentError: If the API key is not found in the environment.
        """
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            raise EnvironmentError(
                f"API key not found in environment variable: {api_key_env_var}"
            )
        return api_key

    def generate_text_individual(self, text: str, image_filepaths: List[str] = []):
        """
        Generate text based on the provided text and images.

        :param text: The input text to process.
        :param image_filepaths: The optional input image to process.
        :return: The generated text.
        """
        pass

    def generate_text_using_past_conversations(
        self, text: str, image_filepaths: List[str] = []
    ):
        """
        Generate text based on the provided text, images, and past conversations.

        :param text: The input text to process.
        :param image_filepaths: The optional input image to process.
        :return: The generated text.
        """
        pass


class BaseO4Mini(APIBaseline):    
    def __init__(self, api_key_env_var: str, model_name: str = "o4-mini-2025-04-16"):
        """
        Initialize the OpenAI GPT-4 model with optional images.
        
        :param api_key_env_var: The environment variable name where the API key is stored.
        :param model_name: The name of the OpenAI model to use.
        """
        super().__init__(api_key_env_var)
        self.model_name = model_name
        self.last_usage_info = None

    def parse_response(self, json_response: str, expected_key: str = "points"):
        """
        Parse the JSON response to extract the 'points' list.

        :param json_response: A string representation of a JSON object.
        :return: The list of points if present, otherwise None.
        """
        json_response
        json_response = json_response.strip("```")
        json_response = json_response.strip("```json")
        try:
            # Parse the JSON string
            data = json.loads(json_response)

            # Extract the 'points' key if it exists
            if expected_key in data:
                return data[expected_key]
            else:
                raise f"Key {expected_key} not found in the response"
        except json.JSONDecodeError as e:
            raise f"Failed to parse JSON: {e}"

    def encode_image(self, image_or_path):
        """
        Encode an image to base64.

        Accepts either:
        • str path to an image file, or
        • PIL.Image.Image

        Returns base64-encoded JPEG bytes (quality=95, 4:4:4).
        """
        # File path case
        if isinstance(image_or_path, str):
            if not os.path.exists(image_or_path):
                raise FileNotFoundError(image_or_path)
            with open(image_or_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        # PIL image case
        if isinstance(image_or_path, Image.Image):
            im = image_or_path.convert("RGB")  # ensures 3-channel JPEG
            buf = io.BytesIO()
            # subsampling=0 → 4:4:4 (no chroma subsampling)
            im.save(buf, format="JPEG", quality=95, subsampling=0, optimize=True)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        raise TypeError("image must be a file path (str) or a PIL.Image.Image")

    def generate_text_individual(self, text: str, image_filepaths: List[str] = []):
        """
        Generate text based on the provided text and images.

        :param text: The input text to process.
        :param image_filepaths: The optional input image to process.
        :return: The generated text.
        """
        return self.generate_text_api_call(text, image_filepaths)

    def generate_text_using_past_conversations(
        self, text: str, image_filepaths: List[str] = []
    ):
        """
        Generate text based on the provided text, images, and past conversations.

        :param text: The input text to process.
        :param image_filepaths: The optional input image to process.
        :return: The generated text.
        """
        return self.generate_text_api_call(
            text, image_filepaths, self.past_conversations
        )

    def generate_text_api_call(
        self,
        text: str,
        image_filepaths: List = [],               # ← leave name/type for minimal surface change
        past_conversations: List[Tuple[str, str]] = [],
        include_past_imgs_as_context: bool = True,
    ):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        message_content = [{"type": "text", "text": text}]

        # Accept path OR np.ndarray
        if image_filepaths:
            for img in image_filepaths:
                base64_image = self.encode_image(img)
                message_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    }
                )

        # past conversations (unchanged except the image handling mirrors above)
        past_conversations_dict = [
            {"role": role, "content": [{"type": "text", "text": content}]}
            for role, content in past_conversations
        ]

        if include_past_imgs_as_context and past_conversations_dict:
            for m in past_conversations_dict:
                if m.get("role") == "user":
                    for img in image_filepaths:
                        base64_image = self.encode_image(img)
                        m["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            }
                        )

        current_message = {"role": "user", "content": message_content}
        messages = past_conversations_dict + [current_message] if past_conversations_dict else [current_message]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": 300,
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )
        try:
            data = response.json()
        except ValueError as e:
            print(f"Error parsing JSON response: {e}")
            return ""

        try:
            print("data", data)
            text_out = data["choices"][0]["message"]["content"]
        except KeyError as e:
            print(f"KeyError: {e}")
            text_out = ""
        self.last_usage_info = data.get("usage", {})
        return text_out
