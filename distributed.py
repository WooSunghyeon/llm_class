from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, AdamW, get_scheduler
from accelerate import Accelerator

import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

accelerator = Accelerator()

# 데이터 셋 적재
raw_datasets = load_dataset("glue", "mrpc")
# 사전학습 언어모델 checkpoint 이름 지정
checkpoint = "bert-base-uncased"
# 지정된 사전학습 언어모델에서 토크나이저 인스턴스화
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# 토크나이저 함수 사용자 정의화 (sentence1, sentence2 컬럼에 대해서만 토크나이징 수행)
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


# 토크나이징 수행
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
# 배치(batch)별 패딩(padding)을 위한 data collator 정의
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 불필요한 입력 컬럼을 제거하고 사전학습 언어모델에 필요한 입력만 남김.
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
# 데이터셋의 label 컬럼명을 labels로 변경
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# 데이터셋의 유형을 PyTorch tensor로 변경
tokenized_datasets.set_format("torch")

# 변경된 컬럼 출력
print(tokenized_datasets["train"].column_names)


# 각 종류별 데이터 로더 생성
train_dataloader = DataLoader(tokenized_datasets["train"], 
                              shuffle=True, 
                              batch_size=8, 
                              collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_datasets["validation"],
                             shuffle=True,
                             batch_size=8,
                             collate_fn=data_collator)

# 사전학습 언어모델 인스턴스화
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
# 최적화 함수 정의
optimizer = AdamW(model.parameters(), lr=5e-5)

train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

# 에포크 개수 설정
num_epochs = 3
# 학습 스텝 수 계산
num_training_steps = num_epochs * len(train_dataloader)
# 학습 스케쥴러 설정
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# GPU로 모델을 이동
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# 진행 상황바 정의
progress_bar = tqdm(range(num_training_steps))

# 모델을 학습 모드로 전환
model.train()
# 학습 루프 시작
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # 현재 배치 중에서 입력값을 모두 GPU로 이동.
        batch = {k: v.to(device) for k, v in batch.items()}
        # 모델 실행
        outputs = model(**batch)
        # 손실값 가져오기
        loss = outputs.loss
        # 역전파 수행
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# 평가 메트릭 가져오기
metric = load_metric("glue", "mrpc")
# 모델을 평가 모드로 전환
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

# 평가 결과 계산 및 출력 
metric.compute()