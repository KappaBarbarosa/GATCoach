export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=247e23f9da34555c8f9d172474c4d49ad150e88d
python3 src/main.py --config=qmix_coach --env-config=sc2 with env_args.map_name=5m_vs_6m