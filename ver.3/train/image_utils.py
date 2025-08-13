# image_utils.py
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def encode_data_to_image(tensor: torch.Tensor, original_length: int, target_size=(1280, 1280)) -> Image.Image:
    """
    잠재 벡터 텐서와 '원본 토큰 길이'를 추상 PNG 이미지에 함께 인코딩합니다.
    """
    # 1. 텐서를 CPU로 옮기고 float32 NumPy 배열로 변환
    tensor_np = tensor.squeeze(0).cpu().numpy().astype(np.float32)
    
    # 2. 텐서 데이터를 바이트 스트림으로 변환 (정보 보존)
    tensor_bytes = tensor_np.tobytes()
    
    # 💡 --- 새로운 부분: 토큰 길이를 4바이트 데이터로 변환 --- 💡
    length_bytes = original_length.to_bytes(4, byteorder='big')
    
    # 최종 데이터 = 길이 정보 + 텐서 정보
    combined_bytes = length_bytes + tensor_bytes
    
    # 4. 1280x1280 RGBA 이미지를 담을 수 있는 빈 NumPy 배열 생성
    image_array = np.zeros((target_size[1], target_size[0], 4), dtype=np.uint8)
    
    # 5. 합쳐진 바이트를 이미지 배열에 덮어쓰기
    byte_array = np.frombuffer(combined_bytes, dtype=np.uint8)
    image_array.flat[:len(byte_array)] = byte_array
    
    # --- 6. 시각적 추상화 생성 ---
    vis_img = Image.fromarray(tensor_np)
    vis_img_resized = vis_img.resize((256, 256), Image.Resampling.BICUBIC)
    
    vis_array = np.array(vis_img_resized)
    
    normalized_array = (vis_array - vis_array.min()) / (vis_array.max() - vis_array.min())
    
    colored_array = (plt.get_cmap('viridis')(normalized_array)[:, :, :3] * 255).astype(np.uint8)
    
    final_visual_img = Image.fromarray(colored_array)
    
    vis_w, vis_h = final_visual_img.size
    start_x = (target_size[0] - vis_w) // 2
    start_y = (target_size[1] - vis_h) // 2
    
    image_array[start_y:start_y+vis_h, start_x:start_x+vis_w, :3] = np.array(final_visual_img)
    image_array[start_y:start_y+vis_h, start_x:start_x+vis_w, 3] = 255
    
    return Image.fromarray(image_array, 'RGBA')

def decode_data_from_image(image: Image.Image, original_shape=(128, 768)) -> (torch.Tensor, int):
    """
    PNG 이미지 파일로부터 텐서와 '원본 토큰 길이'를 함께 복원합니다.
    """
    # 1. 이미지를 NumPy 배열로 변환
    image_array = np.array(image)
    image_bytes = image_array.tobytes()
    
    # 💡 --- 새로운 부분: 길이와 텐서 데이터 분리 --- 💡
    # 2. 앞 4바이트를 읽어 '원본 토큰 길이'를 복원
    length_bytes = image_bytes[:4]
    original_length = int.from_bytes(length_bytes, byteorder='big')
    
    # 3. 나머지 바이트로부터 텐서 복원
    required_bytes = original_shape[0] * original_shape[1] * 4
    tensor_bytes = image_bytes[4 : 4 + required_bytes] # 4바이트 이후부터 읽기
    
    tensor_np = np.frombuffer(tensor_bytes, dtype=np.float32).reshape(original_shape)
    tensor = torch.from_numpy(tensor_np).unsqueeze(0)
    
    # 텐서와 길이를 함께 반환
    return tensor, original_length
