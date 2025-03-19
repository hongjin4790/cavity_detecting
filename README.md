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
- **이미지 유형**: 데이터는 정상 치아와 충치가 있는 치아의 이미지로 구성되어 있으며 COCO 데이터 세트 형식으로 구성되어 있다. JSON 파일에는 파일 경로, 파일 이름, bbox, area 등의 라벨링 데이터가 포함되어 있다.

### 2.2. 데이터 로드

본 프로젝트에서는 Roboflow에서 제공하는 COCO 형식의 데이터를 활용하여 충치 탐지 모델을 학습한다. 

아래는 COCO 형식의 충치 데이터를 로드하고 변환한 후 사용자 정의 CavityDataset 클래스를 사용하여 데이터 세트를 로드하는 코드이다.
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

# 3. 모델 학습

# 4. 실험 결과


# 5. 추론(Inference)


