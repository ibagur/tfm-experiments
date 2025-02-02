python main_rl_minigrid.py --algo ppo --use-gae --date 2023-01-29 \
    --experiment minigrid_5e5_doorkeylava-doorkey-wallgap-lavagap-redbluedoor \
    --num-env-steps 500000 --no-cuda \
    --lr 2.5e-4 --clip-param 0.2 --value-loss-coef 0.5 \
    --num-processes 16 --num-steps 128 --num-mini-batch 256 \
    --log-interval 10 --eval-interval 10 \
    --entropy-coef 0.01 --approach blip --F-prior 1e-16 --seed 1 --tasks-sequence 6 && \
python main_rl_minigrid.py --algo ppo --use-gae --date 2023-01-29 \
    --experiment minigrid_5e5_doorkeylava-doorkey-wallgap-lavagap-redbluedoor \
    --num-env-steps 500000 --no-cuda \
    --lr 2.5e-4 --clip-param 0.2 --value-loss-coef 0.5 \
    --num-processes 16 --num-steps 128 --num-mini-batch 256 \
    --log-interval 10 --eval-interval 10 \
    --entropy-coef 0.01 --approach blip --F-prior 1e-16 --seed 2 --tasks-sequence 6 && \
python main_rl_minigrid.py --algo ppo --use-gae --date 2023-01-29 \
    --experiment minigrid_5e5_doorkeylava-doorkey-wallgap-lavagap-redbluedoor \
    --num-env-steps 500000 --no-cuda \
    --lr 2.5e-4 --clip-param 0.2 --value-loss-coef 0.5 \
    --num-processes 16 --num-steps 128 --num-mini-batch 256 \
    --log-interval 10 --eval-interval 10 \
    --entropy-coef 0.01 --approach blip --F-prior 1e-16 --seed 3 --tasks-sequence 6 && \
python main_rl_minigrid.py --algo ppo --use-gae --date 2023-01-29 \
    --experiment minigrid_5e5_doorkeylava-doorkey-wallgap-lavagap-redbluedoor \
    --num-env-steps 500000 --no-cuda \
    --lr 2.5e-4 --clip-param 0.2 --value-loss-coef 0.5 \
    --num-processes 16 --num-steps 128 --num-mini-batch 256 \
    --log-interval 10 --eval-interval 10 \
    --entropy-coef 0.01 --approach ewc --ewc-lambda 5000 --ewc-online True --seed 1 --tasks-sequence 6 && \
python main_rl_minigrid.py --algo ppo --use-gae --date 2023-01-29 \
    --experiment minigrid_5e5_doorkeylava-doorkey-wallgap-lavagap-redbluedoor \
    --num-env-steps 500000 --no-cuda \
    --lr 2.5e-4 --clip-param 0.2 --value-loss-coef 0.5 \
    --num-processes 16 --num-steps 128 --num-mini-batch 256 \
    --log-interval 10 --eval-interval 10 \
    --entropy-coef 0.01 --approach ewc --ewc-lambda 5000 --ewc-online True --seed 2 --tasks-sequence 6 && \
python main_rl_minigrid.py --algo ppo --use-gae --date 2023-01-29 \
    --experiment minigrid_5e5_doorkeylava-doorkey-wallgap-lavagap-redbluedoor \
    --num-env-steps 500000 --no-cuda \
    --lr 2.5e-4 --clip-param 0.2 --value-loss-coef 0.5 \
    --num-processes 16 --num-steps 128 --num-mini-batch 256 \
    --log-interval 10 --eval-interval 10 \
    --entropy-coef 0.01 --approach ewc --ewc-lambda 5000 --ewc-online True --seed 3 --tasks-sequence 6 && \
python main_rl_minigrid.py --algo ppo --use-gae --date 2023-01-29 \
    --experiment minigrid_5e5_doorkeylava-doorkey-wallgap-lavagap-redbluedoor \
    --num-env-steps 500000 --no-cuda \
    --lr 2.5e-4 --clip-param 0.2 --value-loss-coef 0.5 \
    --num-processes 16 --num-steps 128 --num-mini-batch 256 \
    --log-interval 10 --eval-interval 10 \
    --entropy-coef 0.01 --approach fine-tuning --seed 1 --tasks-sequence 6 && \
python main_rl_minigrid.py --algo ppo --use-gae --date 2023-01-29 \
    --experiment minigrid_5e5_doorkeylava-doorkey-wallgap-lavagap-redbluedoor \
    --num-env-steps 500000 --no-cuda \
    --lr 2.5e-4 --clip-param 0.2 --value-loss-coef 0.5 \
    --num-processes 16 --num-steps 128 --num-mini-batch 256 \
    --log-interval 10 --eval-interval 10 \
    --entropy-coef 0.01 --approach fine-tuning --seed 2 --tasks-sequence 6 && \
python main_rl_minigrid.py --algo ppo --use-gae --date 2023-01-29 \
    --experiment minigrid_5e5_doorkeylava-doorkey-wallgap-lavagap-redbluedoor \
    --num-env-steps 500000 --no-cuda \
    --lr 2.5e-4 --clip-param 0.2 --value-loss-coef 0.5 \
    --num-processes 16 --num-steps 128 --num-mini-batch 256 \
    --log-interval 10 --eval-interval 10 \
    --entropy-coef 0.01 --approach fine-tuning --seed 3 --tasks-sequence 6
