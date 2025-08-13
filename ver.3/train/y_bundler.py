# bundle_y_model.py
import torch
from artist_x_train import ArtistX
from artist_y_train import ArtistY

print("Y를 독립 실행 파일로 만들기 위해 X의 인코더를 Y에 탑재합니다...")

# 1. 학습된 X 모델 로드
model_X = ArtistX()
model_X.load_state_dict(torch.load("artist_x_best_model.pth"))
print("✅ X 모델 로드 완료.")

# 2. Y 모델을 만들고 X의 인코더를 복사
model_Y = ArtistY()
model_Y.load_encoder_from_x(model_X)
print("✅ Y에 X의 인코더 탑재 완료.")

# 3. Y의 전체 가중치(인코더 + 디코더)를 하나의 파일로 저장
torch.save(model_Y.state_dict(), "artist_y_standalone.pth")
print("\n🎉 성공: 독립 실행 가능한 Y 모델 'artist_y_standalone.pth'이 생성되었습니다.")