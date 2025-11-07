import os
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
import magic
import uuid
from io import BytesIO
from torchvision import transforms

from .models import StyTR as StyTR
from .models import transformer as transformer
from .static.model_path import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def content_transform(size=512):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])

def style_transform(h, w):
    return transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor()
    ])

def output_resolution(orig_w, orig_h):
    max_dim = max(orig_w, orig_h)
    scale = 512 / max_dim

    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    return new_w, new_h

class StyTR2:
    def __init__(self):
        self.model = self.load_model()

    def inference(self, content, style):
        try:
            result = self.run_model(content, style)
            return result
        except Exception as e:
            print(e)
            return None

    def validate_and_load_image(self, file_obj, use_cv=False, max_size=300):
        try:
            mime = magic.Magic(mime=True)
            mime_type = mime.from_buffer(file_obj.read(2048))
            if not mime_type.startswith("image/"):
                raise ValueError("Not an image file.")
            file_obj.seek(0)

            img = Image.open(file_obj).convert("RGB")
            img.verify()
            return img
            # if use_cv:
            #     img_bytes = file_obj.read()
            #     img_arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            #     if img_arr is None:
            #         raise ValueError("cv2 failed to decode image.")
            #
            #     h, w = img_arr.shape[:2]
            #     max_dim = max(h, w)
            #     if max_dim > max_size:
            #         scale = max_size / max_dim
            #         new_w, new_h = int(w * scale), int(h * scale)
            #         img_arr = cv2.resize(img_arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            #     return cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            # else:
            #     img = Image.open(file_obj).convert("RGB")
            #     img = ImageOps.exif_transpose(img)
            #     w, h = img.size
            #     max_dim = max(w, h)
            #     if max_dim > max_size:
            #         scale = max_size / max_dim
            #         new_size = (int(w * scale), int(h * scale))
            #         img = img.resize(new_size, Image.Resampling.LANCZOS)
            #     return img
        except (UnidentifiedImageError, OSError, ValueError) as e:
            raise ValueError(f"Image validation failed: {e}")

    def load_model(self):
        vgg = StyTR.vgg
        vgg.load_state_dict(torch.load(os.path.join(BASE_DIR, vgg_path)))
        vgg = nn.Sequential(*list(vgg.children())[:44])

        decoder = StyTR.decoder
        Trans = transformer.Transformer()
        embedding = StyTR.PatchEmbed()

        decoder.load_state_dict(self._load_weights(os.path.join(BASE_DIR, decoder_path)))
        Trans.load_state_dict(self._load_weights(os.path.join(BASE_DIR, Trans_path)))
        embedding.load_state_dict(self._load_weights(os.path.join(BASE_DIR, embedding_path)))

        vgg.eval()
        decoder.eval()
        Trans.eval()
        embedding.eval()

        model = StyTR.StyTrans(vgg, decoder, embedding, Trans)  # args 제거
        model.eval().to(device)
        return model

    def _load_weights(self, path):
        state_dict = torch.load(path)
        return {k: v for k, v in state_dict.items()}

    def run_model(self, content_file, style_file):
        try:
            content_img = self.validate_and_load_image(content_file)
            style_img = self.validate_and_load_image(style_file)

            orig_w, orig_h = content_img.size
            output_w, output_h = output_resolution(orig_w, orig_h)


            content_tensor = content_transform(512)(content_img)
            h, w = content_tensor.shape[1], content_tensor.shape[2]
            style_tensor = style_transform(h, w)(style_img)

            content_tensor = content_tensor.to(device).unsqueeze(0)
            style_tensor = style_tensor.to(device).unsqueeze(0)

            # content_img = Image.fromarray(content_img)
            # content_tensor = test_transform(512)(content_img)
            # h, w = content_tensor.shape[1:]
            #
            #
            # content_tensor = content_tensor.unsqueeze(0).to(device)
            # style_tensor = style_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                output = self.model(content_tensor, style_tensor)
            output_image = output[0].cpu()

            # 💡 Tensor -> PIL.Image
            output_image = transforms.ToPILImage()(torch.clamp(output_image[0], 0, 1))

            # 💡 Resize to original content image size
            output_image = output_image.resize((output_w, output_h), Image.Resampling.LANCZOS)

            # 저장 및 버퍼 반환
            # result_path = f"./outputs/{uuid.uuid4()}.png"
            # output_image.save(result_path)
            # save_image(output_image, result_path)

            buf = BytesIO()
            # save_image(output_image, buf, format="PNG")
            output_image.save(buf, format="PNG")
            buf.seek(0)
            del self.model
            torch.cuda.empty_cache()

            return buf
        except Exception as e:
            print(f"[ERROR] StyTR2 failed: {e}")
            del self.model
            torch.cuda.empty_cache()
            raise
