import os


API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
IMAGES_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
