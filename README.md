# DeepLearning_project

## AI 기반 실시간 수어 번역 프로그램

 프로젝트 주제(부제 : 구체적인 서술)
 - AI 수어 번역 서비스( 수어 데이터를 입력 받아 번역하여 텍스트 데이터를 출력하는 번역 서비스 ) 
 - 농인, 비농인 간 원활한 의사소통을 위한 AI 기반 수어 통역기(or 통역 서비스)
 
 기획의도
 - 일반인과 농인(청각장애인)들의 의사소통을 원활하게 하고자 함.
 - 농인 간 일상 속 대화는 시각적 언어인 수어를 통해 의사소통하지만, 비농인과 농인의 의사소통은 서로의 언어가 상이하여 의사소통의 어려움을 겪음.

 개발일정
 - 2021년 09월 14일 ~ 2021년 11 06일 

 데이터 설명
 - Python Mediapipe를 사용하여 손을 이용한 수어 데이터 수집 

  * MediaPipe [Hands]
  - 머신러닝(ML)/ 딥러닝(DL) 을 사용하여 단일 프레임에서 손의 21개 3D 랜드마크를 추출합니다.
  - ML 파이프라인
    hands detector에 의해 정의된 이미지 영역에서 작동하고 충실도가 높은 3D hands keypoints를 반환하는 hand landmark model
  - hand landmark model
![image](https://user-images.githubusercontent.com/67953299/143383410-f5a13ca6-2de6-441d-9901-52a68f15f9a6.png)
  
  
  
  ### 머신러닝 과 딥러닝 알고리즘 두개로 나누어서 학습 시켜 봤습니다.
 1. 머신러닝(ML) 알고리즘
  -  Mediapipe로 인식한 손의 각 부분 벡터의 사이 각도를 구함 (각 제스처의 각도를 testmodel.h5 파일로 저장)
     각 제스처의 각도를 저장한 데이터셋을 RNN 알고리즘을 사용하여 알아냄
  - KNN_Mediapipe_Hangul-Sign-Languge
    저장된 Dataset를 사용하여 정확도 측정
    KNN을 사용했을 시 정확도는 k=1 ~ 10까지 알아봤을 때 k=1 일 경우 가장 높은 정확도를 보임 (최소 95이상)
 2. 딥러닝 알고리즘 
  - Mediapipe로 인식한 손의 각 부분 벡터의 사이 각도를 구함 (각 제스처의 각도를 csv 파일로 저장)
    각 제스처의 각도를 저장한 데이터셋을 KNN 최근접 알고리즘을 사용하여 알아냄
       
     
 분석 모델 검토
 - 머신러닝과 딥러닝을 이용하여 수어를 번역하는 자연어 처리솔루션 
   (CNN과 RNN 알고리즘을 응용)
• 시각 언어체계 변환을 위해 영상이 입력되면 
  이를 텍스트로 변환하는 방식 (vision to text)

 - 영상 -> text 
 - input -> output
 -> 수화영상을 실시간 텍스트 /음성 뽑아주는 서비스


 사용기술
 - Image Processing(OpenCV, MediaPipe)
 - Machine Learning : KNN 
 - Deep Learning : RNN(LSTM)

 최종산출물
 - Web Service(Django)

------------------------------------------------------------------------------

### django 실행
python manage.py runserver 

