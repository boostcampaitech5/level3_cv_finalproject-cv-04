# 프라이버시 보호를 위한 AI 영상 마스킹 서비스 
## 1. Project overview

유튜브 등 동영상 플랫폼의 성장으로 컨텐츠 크리에이터들이 폭발적으로 증가하고 있습니다. 많은 크리에이터들의 영상 촬영 환경을 보면 전문적인 스튜디오가 아닌, 길거리, 음식점, 카페와 같이 일상 속에서 다양한 형태로 영상 촬영이 이루어지고 있습니다. 그러나, 사전 동의 없이 타인을 촬영한 영상을 업로드하는 것은 초상권 침해 행위이므로, 이를 예방하기 위해서는 개인정보를 비식별화할 수 있는 효율적인 영상 편집 기술이 필요합니다.

본 프로젝트에서는 instance segmentation과 얼굴 인식 기술을 활용하여 영상 속 주요 인물을 인식한 후, 주요 인물을 제외한 인물을 자동으로 blur처리 하여 편집 작업을 효율화 하고자 합니다. 특히, 얼굴 외 신체에도 초상권이 존재하기 때문에 사람의 몸 전체에 대해 마스킹 함으로써 더욱 면밀하게 개인 정보를 보호하고 범죄에 악용될 확률을 줄이는 데 기여하고자 합니다.

- 시연 영상 링크: https://youtu.be/6N0XQx7YGjA

## 2. Development process

- 프로젝트 기간: 2023/06/23 ~ 2023/07/28
- 개발 환경
    - 사용 언어: Python
    - 개발 환경: linux, GCP, V100 GPU 3대
    - 프레임워크 및 주요 라이브러리: PyTorch, OpenCV, FFMPeg, Pytorch, BentoML, Ultralytics
    - 데이터
      - roboflow: [COCO Dataset Limited (Person Only)](https://universe.roboflow.com/shreks-swamp/coco-dataset-limited--person-only)
- 서비스 플로우
![service_flow](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-04/assets/19367749/cb6099ee-94c9-49ab-bf98-37f6dec906e7)
## 4. Proto type
### 1. 주요 인물의 사진과 편집할 영상을 업로드 
![image](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-04/assets/19367749/8561fabb-9072-4ba2-98ec-89859dcf2106)
### 2. 업로드한 사진과 영상을 확인 
![image](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-04/assets/19367749/5fac2e54-b6f9-4916-9a9e-169240903987)
### 3. 예상 소요 시간 확인 후 진행 버튼 클릭  
![image](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-04/assets/19367749/dfb1dc1b-ce85-4693-a9cd-f3bb9d3437f5)
### 4. 편집 완료 후, download 버튼을 눌러 영상 다운로드
![image](https://github.com/boostcampaitech5/level3_cv_finalproject-cv-04/assets/19367749/927a0374-3a25-410a-a25b-3c1122c94596)



## 5. Members
<center>
  
| 강동화 | 박준서 | 한나영 |
| :---: | :---: | :---: |
| <img src = "https://user-images.githubusercontent.com/98503567/235584352-e7b0568f-3699-4b6e-869f-cc675631d74c.png" width="120" height="120"> | <img src = "https://user-images.githubusercontent.com/89245460/234033594-cb90a3c0-f0dc-4218-9e11-2abc8db2be67.png" width="120" height="120"> |<img src = "https://user-images.githubusercontent.com/76798969/233944944-7ff16045-a005-4e4e-bf59-632766194d7f.png" width="120" height="120" />|
| [@oktaylor](https://github.com/oktaylor) | [@Pjunn](https://github.com/Pjunn) |  [@Bandi120424](https://github.com/Bandi120424) |
| 데이터 클렌징, Human segmentation 모델 fine-tuning | 얼굴 인식 모델 구성 및 실험 | 데이터 전처리, 모델 서빙, 영상 편집 파트|

</center>
