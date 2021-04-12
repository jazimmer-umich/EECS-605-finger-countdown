FROM ubuntu:18.04

# Copying over the requirements.txt
COPY ./server /server

# Installing packages needed for installing python, pip, and vim.
RUN apt-get update
RUN apt -y install python3-pip
RUN apt -y install vim
RUN apt -y install curl
RUN apt -y install net-tools

# Installing requirements.txt packages.
RUN pip3 install -r /server/requirements.txt

# Changing working directory.
WORKDIR /server

# Starting notebook server with password.
CMD jupyter notebook --allow-root --ip=0.0.0.0 --port=$PORT --NotebookApp.password='argon2:$argon2id$v=19$m=10240,t=10,p=8$mDc6an2H2g46DiAIetbReA$ytlOxA4XBnNXoR9UI3HX/w'
