python main_rl_minigrid.py --algo ppo --use-gae \
    --experiment minigrid_5e5_redbluedoor-lavagap-doorkey-wallgap \
    --num-env-steps 500000 --no-cuda \
    --lr 2.5e-4 --clip-param 0.2 --value-loss-coef 0.5 \
    --num-processes 16 --num-steps 128 --num-mini-batch 256 \
    --log-interval 10 --eval-interval 10 \
    --entropy-coef 0.01 --approach blip --F-prior 5e-17 --seed 1 --tasks-sequence 1 && \
python main_rl_minigrid.py --algo ppo --use-gae \
    --experiment minigrid_5e5_redbluedoor-lavagap-doorkey-wallgap \
    --num-env-steps 500000 --no-cuda \
    --lr 2.5e-4 --clip-param 0.2 --value-loss-coef 0.5 \
    --num-processes 16 --num-steps 128 --num-mini-batch 256 \
    --log-interval 10 --eval-interval 10 \
    --entropy-coef 0.01 --approach blip --F-prior 5e-17 --seed 2 --tasks-sequence 1 && \
python main_rl_minigrid.py --algo ppo --use-gae \
    --experiment minigrid_5e5_redbluedoor-lavagap-doorkey-wallgap \
    --num-env-steps 500000 --no-cuda \
    --lr 2.5e-4 --clip-param 0.2 --value-loss-coef 0.5 \
    --num-processes 16 --num-steps 128 --num-mini-batch 256 \
    --log-interval 10 --eval-interval 10 \
    --entropy-coef 0.01 --approach blip --F-prior 5e-17 --seed 3 --tasks-sequence 1 && \
python main_rl_minigrid.py --algo ppo --use-gae \
    --experiment lavagap-doorkey-emptyrand-redbluedoor-wallgap \
    --num-env-steps 500000 --no-cuda \
    --lr 2.5e-4 --clip-param 0.2 --value-loss-coef 0.5 \
    --num-processes 16 --num-steps 128 --num-mini-batch 256 \
    --log-interval 10 --eval-interval 10 \
    --entropy-coef 0.01 --approach blip --F-prior 5e-17 --seed 1 --tasks-sequence 3 && \
python main_rl_minigrid.py --algo ppo --use-gae \
    --experiment lavagap-doorkey-emptyrand-redbluedoor-wallgap \
    --num-env-steps 500000 --no-cuda \
    --lr 2.5e-4 --clip-param 0.2 --value-loss-coef 0.5 \
    --num-processes 16 --num-steps 128 --num-mini-batch 256 \
    --log-interval 10 --eval-interval 10 \
    --entropy-coef 0.01 --approach blip --F-prior 5e-17 --seed 2 --tasks-sequence 3 && \
python main_rl_minigrid.py --algo ppo --use-gae \
    --experiment lavagap-doorkey-emptyrand-redbluedoor-wallgap \
    --num-env-steps 500000 --no-cuda \
    --lr 2.5e-4 --clip-param 0.2 --value-loss-coef 0.5 \
    --num-processes 16 --num-steps 128 --num-mini-batch 256 \
    --log-interval 10 --eval-interval 10 \
    --entropy-coef 0.01 --approach blip --F-prior 5e-17 --seed 3 --tasks-sequence 3
