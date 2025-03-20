<h1> 충치 탐지 프로젝트</h1> 

# 1. 프로젝트 개요

본 프로젝트는 컴퓨터 비전 기술을 활용하여 충치를 자동으로 탐지하는 딥러닝 모델을 개발하는 것을 목표로 한다. 충치는 전 세계적으로 흔한 구강 질환으로 조기에 발견하여 치료하는게 매우 중요하다. 하지만 의사들이 직접 이미지를 분석하는 과정은 많은 시간이 소요된다.  따라서 본 프로젝트는 딥러닝 기반 모델을 활용하여 치과 진단 및 구강 건강 관리등 이러한 분야에서 문제를 해결하고자 한다.


# 2. 데이터세트
#### 데이터 다운로드: https://universe.roboflow.com/duong-duc-cuong/cavity-n3ioq/dataset/2

### 2.1. 데이터세트 구성

<img src="https://github.com/user-attachments/assets/f7b7c8a6-9e89-4963-ab7e-31762e6c7dcd"  width="250" height="250"/><img src="https://github.com/user-attachments/assets/b626ecf2-d4bc-4aef-80e7-1ac48e6d8190"  width="250" height="250"/>

- **총 이미지 수**: 418장
  - **훈련 세트**: 80% (336장)
  - **검증 세트**: 19.8% (81장)
  - **테스트 세트**: 0.2% (1장)

# 3. 모델
본 프로젝트에서는 **Faster R-CNN과 YOLOv5 모델을 활용하여 충치 데이터 학습을 진행하였다.

<img src="https://github.com/user-attachments/assets/a96ccb64-0cbb-4677-957a-6e92efb78e99"  width="600" height="300"/>

- **Faster R-CNN**
  - 객체 탐지와 분류를 위한 딥러닝 모델
  - Fast R-CNN 모델의 속도와 정확도 문제를 개선한 모델
  - 바운딩 박스 제안 영역을 생성하는 Region Proposal Network(RPN)를 추가하여 효율적으로 객체 탐지
- **YOLOv5**
  - 객체 탐지를 위한 딥러닝 모델
  - Anchor-Free 방식을 사용하여 객체의 위치와 크기 예측
  - 기존 YOLO 모델들의 성능을 개선하고 다양한 크기의 객체를 정확하게 탐지

## 3.1 데이터 로드
  Roboflow에서 제공하는 데이터를 활용한다.

  - COCO 형식의 데이터는 Faster R-CNN 모델 학습에 사용
  - YOLO 형식 데이터는 YOLOv5 모델 학습에 사용

### 3.1.1 COCO 데이터 로드
COCO 데이터 형식은 json 파일을 포함하고 bbox 정보와 category_id 등을 포함한다.
아래는 COCO 형식 데이터를 로드하여 사용자 정의 CavityDataset 클래스를 사용하여 데이터 세트를 로드하는 코드이다.

```
class CavityDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # COCO 객체
        coco = self.coco
        # 이미지 ID
        img_id = self.ids[index]
        # 어노테이션 ID 가져오기
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # 어노테이션 로드
        coco_annotation = coco.loadAnns(ann_ids)
        # 이미지 경로 설정
        path = coco.loadImgs(img_id)[0]['file_name']
        # 이미지 열기
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        
        # 객체 수 확인
        num_objs = len(coco_annotation)
        
        # 바운딩 박스 변환 (COCO 형식 → PyTorch 형식)
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # 라벨 가져오기
        labels = [coco_annotation[i]['category_id'] for i in range(num_objs)]
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # 어노테이션 딕셔너리 생성
        my_annotation = {"boxes": boxes, "labels": labels}
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, my_annotation

    def __len__(self):
        return len(self.ids)

cavity_dataset = CavityDataset(root=train_data_dir,
                                annotation=train_coco,
                                transforms=get_transform())
```

### 3.1.2 YOLO 데이터 로드
YOLOv5의 데이터 형식은 이미지, txt 파일로 저장되는 label, data.yaml으로 구성되어 있다.

Roboflow에서는 YOLO 데이터 형식을 지원하기 때문에 아래와 같이 바로 다운로드 할 수 있다.

![image](https://github.com/user-attachments/assets/aea16d3d-e6ae-4e5a-b2e6-7cce86a095f9)

# 4. 모델 평가 및 비교

- **Faster R-CNN**
  
  <img src="https://github.com/user-attachments/assets/a6a7eb51-65d1-47df-ac06-1de2261d1ca4"  width="500" height="300"/>

  - train_loss가 지속적으로 감소하며 안정적으로 학습이 진행
  - 하지만 test_loss가 점점 증가하는 경향이 있어 과적합의 가능성이 있음
  - train_map은 거의 1에 수렴하지만 test_map은 0.8 근처에서 변동하는것으로 모델의 일반화 성능이 불안정

- **YOLOv5**

  <img src="https://github.com/user-attachments/assets/5bc595b6-89bf-4807-9d83-46686b25baa5"  width="500" height="300"/>

  - train/box_loss, train/obj_loss, train/cls_loss 모두 빠르게 감소하며 안정적으로 수렴
  - val/box_loss, val/obj_loss도 비슷한 양상을 보이지만 val/cls_loss에서 변동성이 큰 것으로 보아 클래스 분류에 대한 불안정성이 존재
  - metrics/mAP_0.5는 0.8, metrics/mAP_0.5:0.95는 약 0.4
  - metrics/precision, metrics/recall 모두 초반에는 급격히 증가하고 이후에는 안정적인 값을 유지

- **Faster R-CNN**

  <img src="https://github.com/user-attachments/assets/8d1534c7-db31-4d68-869c-33cb3b59db61"  width="500" height="300"/>
  
  <img src="https://github.com/user-attachments/assets/68486c8d-f6da-4262-acbf-2e19616560dc"  width="500" height="300"/>

- **YOLOv5**
  
  <img src="https://github.com/user-attachments/assets/3328de47-c9b1-42f5-a210-0c162b15c46e"  width="500" height="300"/>
  
  <img src="https://github.com/user-attachments/assets/0f23f7ce-3535-4da1-93d9-6aedefa9781b"  width="500" height="300"/>


# 5. 결론

- 실험 결과를 종합하면 YOLOv5가 Faster R-CNN보다 전반적으로 더 우수한 성능을 보인다. YOLOv5는 빠른 학습 속도와 안정적인 검증 성능을 유지하고 mAP도 비교적 일정하게 유지되는 반면, Faster R-CNN은 학습 데이터에서는 높은 성능을 보이지만 검증 데이터에서 성능 변동이 크고 과적합(overfitting) 가능성이 존재한다.
- Faster R-CNN의 성능을 개선하기 위해서는 데이터 증강(Data Augmentation)을 하여 일반화 성능을 향상시키고, 학습률, 배치 사이즈 등을 최적화하는 하이퍼 파라미터 튜닝이 필요해 보인다. 또한 더 많은 데이터를 학습에 사용하면 과적합 문제를 완화할 수 있을 것이다.

