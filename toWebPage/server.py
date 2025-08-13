# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torch import nn
import numpy as np
from PIL import Image, ImageDraw
from matplotlib.colors import ListedColormap
from transformers import BertModel, BertLMHeadModel, BertTokenizer
import base64
import io
import unicodedata
from typing import Optional, Tuple

# --- 모델 클래스 정의 ---
class ArtistX(nn.Module):
    def __init__(self, model_name: str = 'bert-base-uncased') -> None:
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.encoder = BertModel.from_pretrained(model_name)
        # is_decoder=True 로 설정해 디코더로 사용
        self.decoder = BertLMHeadModel.from_pretrained(model_name, is_decoder=True)


class ArtistY(nn.Module):
    def __init__(self, model_name: str = 'bert-base-uncased') -> None:
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.encoder = BertModel.from_pretrained(model_name)
        self.decoder = BertLMHeadModel.from_pretrained(model_name, is_decoder=True)


# --- 유틸리티 함수들 ---
def combine_jamo(text: str) -> str:
    """한글 자모 결합(NFC) 처리"""
    return unicodedata.normalize('NFC', text)


def encode_data_to_image(
    tensor: torch.Tensor,
    original_length: int,
    target_size: Tuple[int, int] = (1280, 1280),
) -> Image.Image:
    """잠재 벡터(tensor)와 원본 길이(original_length)를 RGBA 이미지에 인코딩"""
    # 1) 텐서를 바이트 스트림으로 결합
    tensor_np = tensor.squeeze(0).cpu().numpy().astype(np.float32)
    tensor_bytes = tensor_np.tobytes()
    length_bytes = original_length.to_bytes(4, byteorder='big')
    combined_bytes = length_bytes + tensor_bytes

    # 2) 이미지 버퍼 생성
    image_array = np.zeros((target_size[1], target_size[0], 4), dtype=np.uint8)
    byte_array = np.frombuffer(combined_bytes, dtype=np.uint8)
    image_array.flat[: len(byte_array)] = byte_array

    # 3) 시각화용 추상화 이미지 생성 (256×256)
    tensor_norm = tensor_np - tensor_np.min()
    tensor_norm = (tensor_norm / tensor_norm.max() * 255).astype(np.uint8)
    vis_img = Image.fromarray(tensor_norm).resize((256, 256), Image.Resampling.BICUBIC)
    vis_array = np.array(vis_img)

    block_size = 256 // 8
    picasso_canvas = np.zeros_like(vis_array)
    for r in range(8):
        for c in range(8):
            block = vis_array[
                r * block_size : (r + 1) * block_size,
                c * block_size : (c + 1) * block_size,
            ]
            choice = np.random.randint(4)
            if choice == 1:
                block = np.flipud(block)
            elif choice == 2:
                block = np.fliplr(block)
            elif choice == 3:
                block = np.rot90(block)
            picasso_canvas[
                r * block_size : (r + 1) * block_size,
                c * block_size : (c + 1) * block_size,
            ] = block

    picasso_colors = [
        '#2E4A34',
        '#523A62',
        '#F2D338',
        '#C82D31',
        '#D9D9D9',
        '#343434',
    ]
    cmap = ListedColormap(picasso_colors)
    norm_canvas = (picasso_canvas - picasso_canvas.min()) / (
        picasso_canvas.max() - picasso_canvas.min()
    )
    colored = (cmap(norm_canvas)[:, :, :3] * 255).astype(np.uint8)
    final_vis = Image.fromarray(colored)

    # 그리드 라인 추가
    draw = ImageDraw.Draw(final_vis)
    for i in range(1, 8):
        x = i * block_size
        draw.line([(x, 0), (x, 255)], fill='black', width=2)
        draw.line([(0, x), (255, x)], fill='black', width=2)

    # 4) 최종 이미지 중앙에 시각화 삽입
    vis_w, vis_h = final_vis.size
    start_x = (target_size[0] - vis_w) // 2
    start_y = (target_size[1] - vis_h) // 2
    image_array[start_y : start_y + vis_h, start_x : start_x + vis_w, :3] = np.array(
        final_vis
    )
    image_array[start_y : start_y + vis_h, start_x : start_x + vis_w, 3] = 255

    return Image.fromarray(image_array, 'RGBA')


def decode_data_from_image(
    image: Image.Image, original_shape: Tuple[int, int] = (128, 768)
) -> Tuple[torch.Tensor, int]:
    """RGBA 이미지에서 바이트를 추출해 tensor와 original_length 반환"""
    image_array = np.array(image)
    image_bytes = image_array.tobytes()

    # 처음 4바이트는 original_length
    length_bytes = image_bytes[:4]
    original_length = int.from_bytes(length_bytes, byteorder='big')

    # 나머지 바이트를 텐서로 복원
    count = original_shape[0] * original_shape[1] * 4  # float32
    tensor_bytes = image_bytes[4 : 4 + count]
    tensor_np = np.frombuffer(tensor_bytes, dtype=np.float32).reshape(original_shape)
    tensor = torch.from_numpy(tensor_np).unsqueeze(0)

    return tensor, original_length


# --- Flask 앱 설정 ---
app = Flask(__name__)
CORS(app)

# --- 모델 로딩 ---
print("서버 시작 중... 모델을 불러옵니다.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_x = ArtistX()
model_x.load_state_dict(torch.load("artist_x_best_model.pth", map_location=device))
model_x.to(device).eval()

model_y = ArtistY()
model_y.load_state_dict(torch.load("artist_y_standalone.pth", map_location=device))
model_y.to(device).eval()

print(f"✅ 모델 로딩 완료. (Device: {device})")


# --- API 엔드포인트 ---
@app.route('/encode', methods=['POST'])
def encode():
    payload = request.get_json(silent=True) or {}
    text: Optional[str] = payload.get('text')
    mode: str = payload.get('mode', 'y')

    if not text:
        return jsonify({"error": "텍스트가 없습니다."}), 400

    model = model_y if mode == 'y' else model_x

    with torch.no_grad():
        inputs = model.tokenizer(
            text,
            return_tensors='pt',
            max_length=128,
            padding='max_length',
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        enc_output = model.encoder(
            inputs['input_ids'], attention_mask=inputs['attention_mask']
        )
        latent_vector = enc_output.last_hidden_state
        # .item() 은 Python number 반환, int() 로 명시적 보장
        original_length = int(torch.sum(inputs['attention_mask']).item())

    image = encode_data_to_image(latent_vector, original_length)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_str = base64.b64encode(buf.getvalue()).decode()

    return jsonify({"image": img_str})


@app.route('/decode', methods=['POST'])
def decode():
    payload = request.get_json(silent=True) or {}
    base64_image: Optional[str] = payload.get('image')
    mode: str = payload.get('mode', 'y')

    if not base64_image:
        return jsonify({"error": "이미지 데이터가 없습니다."}), 400

    try:
        raw = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(raw))
        latent_vector, original_length = decode_data_from_image(image)
    except Exception as e:
        return jsonify({"error": f"이미지 처리 오류: {e}"}), 400

    model = model_y if mode == 'y' else model_x
    latent_vector = latent_vector.to(device)

    with torch.no_grad():
        attention_mask = torch.ones(latent_vector.shape[:-1], dtype=torch.long).to(device)
        if mode == 'x':
            logits = model.decoder(
                inputs_embeds=latent_vector, attention_mask=attention_mask
            ).logits
            pred_ids = torch.argmax(logits, dim=-1)
            clean_ids = pred_ids[0][:original_length]
            raw_text = model.tokenizer.decode(clean_ids, skip_special_tokens=True)
            result_text = combine_jamo(raw_text)
        else:  # mode 'y'
            outputs = model.decoder.generate(
                inputs_embeds=latent_vector,
                attention_mask=attention_mask,
                max_new_tokens=50,
                do_sample=True,
                top_k=50,
                pad_token_id=model.tokenizer.pad_token_id,
            )
            raw_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
            result_text = combine_jamo(raw_text)

    return jsonify({"text": result_text})


if __name__ == '__main__':
    app.run(debug=True)
