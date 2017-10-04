import os

def progress_read(file_path):
    out = 0
    if os.path.isfile(file_path):
        f = open(file_path, "r")
        line = f.readline()
        print ("line", line)
        if len(line) > 0 and int(line) > 0:
            out = int(line)
        f.close()
    return out


def progress_write(file_path, prog_num):
    f = open(file_path, "w")
    f.writelines([str(prog_num)])
