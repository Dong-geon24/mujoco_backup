# mujoco_backup

#### train: learning에서 해당하는 학습을 가져와 진행
    #위치: /home/arclab/.mujoco273/mujoco-2.3.7/gym_solutions/
    #학습수행
    python sb3.py [학습모델_이름] [알고리즘] -t 
    #학습 결과 시뮬
    python sb3.py [학습모델_이름] [알고리즘] -s [학습데이터_경로}
    #텐서보드 - 학습 실시간 확인
    tensorboard --logdir logs

#### learning: 실질적인 강화학습 알고리즘
    #위치: /home/arclab/.local/lib/python3.10/site-packages/gymnasium/envs/mujoco

> 학습파일 Register하기:
> 외부 init.py
> 경로:/home/arclab/.local/lib/python3.10/site-packages/gymnasium/envs


> 내부 init.py
> 경로: /home/arclab/.local/lib/python3.10/site-packages/gymnasium/envs/mujoco
![image](https://github.com/user-attachments/assets/254d4a17-dea2-4604-b6f7-953eeedff2ed)
