#!/bin/bash
cd ~ 
git clone https://github.com/PaddlePaddle/FluidDoc
git clone https://github.com/tianshuo78520a/PaddlePaddle.org.git
sh ~/FluidDoc/doc/fluid/api/gen_doc.sh
pip install ${PADDLE_ROOT}/build/opt/paddle/share/wheels/*.whl
apt-get update && apt-get install -y python-dev build-essential
cd ~/PaddlePaddle.org/portal
pip install -r requirements.txt
#If the default port is not occupied, you can use port 8000, you need to replace it with a random port on the CI.
sed -i "s#8000#$1#g" runserver
nohup ./runserver --paddle ~/FluidDoc &
