#FROM jupyter/minimal-notebook:latest
FROM jupyter/tensorflow-notebook:latest

USER root

RUN apt-get update && apt-get install -y \
    python3-distutils \
    libproj-dev \
    gdal-bin

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

## General purpose Python libraries:
#RUN pip install Pillow

## Base computing libraries:
# RUN pip install scipy \ 
# 	matplotlib \
#     numpy \
#     pandas

## Machine learning librares :
# RUN pip install scikit-learn \
# 	scikit-image

## Deep learning libraries :
# RUN pip install tensorflow==2.4.0 \
# 	keras==2.4.3



## The following lines are needed in order to run a jupyter notebook in docker from vscode:
## Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
# RUN pip install pexpect python-dateutil -t /home/jovyan/.local/lib/python3.8/site-packages --upgrade
# ENV TINI_VERSION v0.6.0
# ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
# RUN chmod +x /usr/bin/tini


USER $NB_UID
WORKDIR $HOME

# copy files for standalone execution:
COPY . /usr/src/onunet


#ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

# for standalone execution of a script:
#CMD ["bash"]