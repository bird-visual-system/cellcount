#!/bin/bash

## build image:
#docker build -t ptitmatheux/jupyter-cellcount .

## run image in interactive bash mode:
# AWS_ACCESS_KEY_ID=$(aws --profile default configure get aws_access_key_id)
# AWS_SECRET_ACCESS_KEY=$(aws --profile default configure get aws_secret_access_key)
# docker run -it -e USER=$USER -e USERID=$UID \
#  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
#  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
#  -e AWS_DEFAULT_REGION=eu-west-3 \
#  -v /data:/data \
#  -v /home/$USER/Projects:/home/$USER/Projects \
#  ptitmatheux/jupyter_grib_aws bash

## run Jupyter notebook
# AWS_ACCESS_KEY_ID=$(aws --profile default configure get aws_access_key_id)
# AWS_SECRET_ACCESS_KEY=$(aws --profile default configure get aws_secret_access_key)
# docker run -it -p 8888:8888 -e USER=$USER -e USERID=$UID \
#  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
#  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
#  -e AWS_DEFAULT_REGION=eu-west-3 \
#  -v /home/michel:/home/jovyan \
#  ptitmatheux/jupyter_grib_aws /opt/conda/bin/jupyter lab


#-v /data:/data \
#-v /media/sf_mic/Documents:/home/jovyan/Documents \
#-v /media/sf_mic/Proj:/home/jovyan/Proj \

#-v /home/$USER/.aws:/home/$USER/.aws


docker run -it -p 8888:8888 -e USER=$USER -e USERID=$UID \
 -v /home/michel:/home/jovyan \
 ptitmatheux/jupyter-cellcount
