mkdir checkpoints > /dev/null 2>&1
mkdir logs > /dev/null 2>&1
python /home/baifan/RL/SceneMover/src/GA3C/GA3C.py "$@"
