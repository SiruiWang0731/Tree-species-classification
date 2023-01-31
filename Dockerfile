# this fetches a pre-build base image of pytorch compiled for CUDA > 11.3,
# please use this as the base image for the RS server
FROM rsseminar/pytorch:latest

# install dependencies
# RUN conda install -c conda-forge cupy  
RUN pip install opencv-python
RUN pip install scipy rasterio natsort matplotlib scikit-image tqdm natsort
RUN pip install s2cloudless
RUN pip install Pillow
RUN pip install dominate
RUN pip install visdom

COPY requirements.txt ./tmp/
RUN pip install --requirement ./tmp/requirements.txt
COPY . ./tmp/

RUN apt update && apt install  openssh-server sudo -y
# Create a user “sshuser” and group “sshgroup”
RUN groupadd sshgroup && useradd -ms /bin/bash -g sshgroup sshuser
# Create sshuser directory in home
RUN usermod -aG sudo sshuser
RUN mkdir -p /home/sshuser/.ssh
# Copy the ssh public key in the authorized_keys file. The idkey.pub below is a public key file you get from ssh-keygen. They are under ~/.ssh directory by default.
COPY id_rsa.pub /home/sshuser/.ssh/authorized_keys
COPY id_rsa_mac.pub  /home/sshuser/.ssh/authorized_keys
# change ownership of the key file. 
RUN chown sshuser:sshgroup /home/sshuser/.ssh/authorized_keys && chmod 600 /home/sshuser/.ssh/authorized_keys
# Start SSH service
RUN service ssh start
# Expose docker port 22
RUN sed -ri 's/PermitEmptyPasswords no/PermitEmptyPasswords yes/' /etc/ssh/sshd_config
RUN sed -ri 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/^UsePAM yes/UsePAM no/' /etc/ssh/sshd_config

# Delete root password (set as empty)
RUN passwd -d root
EXPOSE 22
CMD ["/usr/sbin/sshd","-D"]

RUN echo 'root:Docker!' | chpasswd


# bake repository into dockerfile
RUN mkdir -p ./npy
# RUN mkdir -p ./models
# RUN mkdir -p ./options
# RUN mkdir -p ./util

ADD npy ./npy
# ADD models ./models
# ADD options ./options
# ADD util ./util
ADD . ./

# this is setting your pwd at runtime
WORKDIR /workspace