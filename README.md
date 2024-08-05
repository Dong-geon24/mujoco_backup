# mujoco_backup

#### train: learning에서 해당하는 학습을 가져와 진행
    #위치: /home/arclab/.mujoco273/mujoco-2.3.7/gym_solutions/
    #학습수행
    python sb3.py [학습모델_이름] [알고리즘] -t 
    #학습 결과 시뮬
    python sb3.py [학습모델_이름] [알고리즘] -s [학습데이터_경로}
    #텐서보드 - 학습 실시간 확인
    tensorboard --logdir logs

> sb3.py: 3가지 알고리즘, 학습과정 전부 저장, 원하는 과정에서 실행가능
> sb3v2.py: 알고리즘을 확장
> sb3v3.py: 확장된 알고리즘, best_model만 저장(목표치에 도달하면 정지), log_file 분할
> custom: 확장된 알고리즘, best_model(학습에 진전이 없으멵 정지), log_file 분할(여러모델 돌려도 안겹치게)

#### learning: 실질적인 강화학습 알고리즘
    #위치: /home/arclab/.local/lib/python3.10/site-packages/gymnasium/envs/mujoco

> 학습파일 Register하기:
> 외부 init.py
> 경로:/home/arclab/.local/lib/python3.10/site-packages/gymnasium/envs


> 내부 init.py
> 경로: /home/arclab/.local/lib/python3.10/site-packages/gymnasium/envs/mujoco
![image](https://github.com/user-attachments/assets/254d4a17-dea2-4604-b6f7-953eeedff2ed)
