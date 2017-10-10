import os
import paramiko

ssh = paramiko.SSHClient()
ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# server = "flogo3.local"
# username="poppy"
# password="poppy"

server = "tegra-ubuntu.local"
username="ubuntu"
password="ubuntu"


ssh.connect(server, username=username, password=password)
sftp = ssh.open_sftp()


localpath = os.path.realpath(__file__)
print (localpath)
remotepath = "/mnt/2TB/flo-robot-data/ladida-bums.py"

sftp.put(localpath, remotepath)

sftp.close()
ssh.close()