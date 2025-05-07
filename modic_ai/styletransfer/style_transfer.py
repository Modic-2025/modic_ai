from PIL import Image
from io import BytesIO
import uuid
import cv2
import numpy as np

from PIL import Image, UnidentifiedImageError, ImageOps
import magic
import torch

# def run_style_transfer(content_file, style_file, prompt, model_ref):
#     detector = model_ref["detector"]
#     styleshot = model_ref["styleshot"]
#
#     # Convert content image to PIL via cv2
#     content_arr = cv2.imdecode(np.frombuffer(content_file.read(), np.uint8), cv2.IMREAD_COLOR)
#     content_arr = cv2.cvtColor(content_arr, cv2.COLOR_BGR2RGB)
#     content_arr = detector(content_arr)
#     content_pil = Image.fromarray(content_arr)
#
#     # Style image
#     style_pil = Image.open(style_file)
#
#     generation = styleshot.generate(style_image=style_pil, prompt=[[prompt]], content_image=content_pil)
#
#     output_image = generation[0][0]
#     output_image.save(f"./media/outputs/{uuid.uuid4()}.png")
#     buf = BytesIO()
#     output_image.save(buf, format="PNG")
#     buf.seek(0)
#
#     return buf

def validate_and_load_image(file_obj, use_cv=False, max_size=300):
    try:
        # MIME 타입 검사 (선택 사항)
        mime = magic.Magic(mime=True)
        mime_type = mime.from_buffer(file_obj.read(2048))
        if not mime_type.startswith("image/"):
            raise ValueError("Not an image file.")
        file_obj.seek(0)

        # Pillow로 열기
        img = Image.open(file_obj)
        img.verify()  # 손상된 이미지 확인
        file_obj.seek(0)  # 다시 읽기 위해 리셋

        if use_cv:
            img_bytes = file_obj.read()
            img_arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img_arr is None:
                raise ValueError("cv2 failed to decode image.")

            h, w = img_arr.shape[:2]
            max_dim = max(h, w)
            if max_dim > max_size:
                scale = max_size / max_dim
                new_w, new_h = int(w * scale), int(h * scale)
                img_arr = cv2.resize(img_arr, (new_w, new_h), interpolation=cv2.INTER_AREA)

            return cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        else:
            img = Image.open(file_obj).convert("RGB")
            img = ImageOps.exif_transpose(img)  # <-- 회전 보정

            w, h = img.size
            max_dim = max(w, h)
            if max_dim > max_size:
                scale = max_size / max_dim
                new_size = (int(w * scale), int(h * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            return img
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise ValueError(f"Image validation failed: {e}")

def run_style_transfer(content_file, style_file, prompt, model_ref):
    try:
        content_arr = validate_and_load_image(content_file, use_cv=True)
        style_pil = validate_and_load_image(style_file, use_cv=False)

        content_arr = model_ref["detector"](content_arr)
        content_pil = Image.fromarray(content_arr)

        # Style transfer
        generation = model_ref["styleshot"].generate(
            style_image=style_pil, prompt=[[prompt]], content_image=content_pil
        )

        output_image = generation[0][0]
        output_image.save(f"./media/outputs/{uuid.uuid4()}.png")

        buf = BytesIO()
        output_image.save(buf, format="PNG")
        buf.seek(0)
        return buf

    except Exception as e:
        print(f"[ERROR] Style transfer failed: {e}")
        raise
