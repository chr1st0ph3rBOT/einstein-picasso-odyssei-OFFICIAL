# download_models.py
from transformers import BertTokenizer, BertModel, BertLMHeadModel

# 프로젝트에서 사용하는 모델의 이름
MODEL_NAME = 'bert-base-uncased'

def download_and_cache_model():
    """
    Hugging Face Hub에서 모델 파일을 다운로드하여 로컬 컴퓨터에 캐시합니다.
    """
    print(f"'{MODEL_NAME}' 모델 다운로드를 시작합니다...")
    print("인터넷 연결이 필요하며, 처음에는 시간이 다소 걸릴 수 있습니다.")
    
    try:
        # 1. 토크나이저 다운로드
        print("토크나이저 다운로드 중...")
        BertTokenizer.from_pretrained(MODEL_NAME)
        
        # 2. 인코더용 모델 다운로드
        print("인코더 모델 다운로드 중...")
        BertModel.from_pretrained(MODEL_NAME)
        
        # 3. 디코더용 모델 다운로드
        print("디코더 모델 다운로드 중...")
        BertLMHeadModel.from_pretrained(MODEL_NAME, is_decoder=True)
        
        print("\n🎉 성공: 모든 모델 파일이 성공적으로 다운로드 및 캐시되었습니다.")
        print("이제 다른 앱들을 인터넷 연결 없이 실행할 수 있습니다.")

    except Exception as e:
        print(f"\n❌ 오류: 모델 다운로드 중 문제가 발생했습니다.")
        print(f"에러 메시지: {e}")
        print("인터넷 연결 상태를 확인하고 다시 시도해 주세요.")

if __name__ == '__main__':
    download_and_cache_model()
