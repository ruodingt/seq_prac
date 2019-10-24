FROM tensorflow/tensorflow:1.12.3-gpu-py3

#FROM tensorflow/tensorflow:1.13.1-gpu-py3-jupyter

ARG user
ARG password

RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo "$user:$password" | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
#RUN echo "prohibit-password/PermitRootLogin yes" >> /etc/ssh/sshd_config
#RUN echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

RUN apt-get update && apt-get install -y tmux

RUN apt-get install -y libsm6 libxrender1 libfontconfig1


COPY requirement.txt requirement.txt
RUN pip3 install -r requirement.txt

EXPOSE 22
EXPOSE 6006
EXPOSE 8888
CMD ["/usr/sbin/sshd", "-D"]



