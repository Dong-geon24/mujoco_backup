# mujoco_backup


#### train: learning에서 해당하는 학습을 가져와 진행
    #위치: /home/arclab/.mujoco273/mujoco-2.3.7/gym_solutions/
    #학습수행
    #1. train
    python sb3.py [모델이름] [강화학습알고리즘] -t
    #2. test
    python sb3.py [모델이름] [강화학습알고리즘] -s [모델패스]
    #텐서보드 - 학습 실시간 확인
    tensorboard --logdir logs

> sb3.py: 3가지 알고리즘, 학습과정 전부 저장, 원하는 과정에서 실행가능    
> sb3v2.py: 알고리즘을 확장    
> sb3v3.py: 확장된 알고리즘, best_model만 저장(목표치에 도달하면 정지), log_file 분할    
> custom: 확장된 알고리즘, best_model(학습에 진전이 없으면 정지), log_file 분할(여러모델 돌려도 안겹치게)
<pre>
학습종료 조건
`stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=10000, verbose=1)`        
최소 10000번의 평가 후에 시작됩니다 (min_evals=10000).
그 이후 5번의 연속된 평가에서 모델 성능 향상이 없으면 학습을 종료합니다 (max_no_improvement_evals=5).    
</pre>

#### learning: 실질적인 강화학습 알고리즘
    #위치: /home/arclab/.local/lib/python3.10/site-packages/gymnasium/envs/mujoco
    코드는 덧붙여 작성하면 train에서 자동으로 호출됨

> 학습파일 Register하기:
> 외부 init.py
> 경로:/home/arclab/.local/lib/python3.10/site-packages/gymnasium/envs


> 내부 init.py
> 경로: /home/arclab/.local/lib/python3.10/site-packages/gymnasium/envs/mujoco
![image](https://github.com/user-attachments/assets/254d4a17-dea2-4604-b6f7-953eeedff2ed)
>

#### Future Work
1. 12bar model 파라미터 수정 - 진동을 최소화 하도록
2. 강화학습 알고리즘 수정 - 평형상태(구모양)을 유지하는 보상을 추가
3. train파일 custom 수정 - 베스트모델만 남길 수 있게.

##### custom train execute
    python sb3_custom.py [학습모델_이름] [알고리즘] [실험이름]
    #학습 결과 시뮬
    python sb3_custom.py [학습모델_이름] [알고리즘] [실험이름] --test
