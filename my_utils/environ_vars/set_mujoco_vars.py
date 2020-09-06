#PYTHONUNBUFFERED=1;
# LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/michael/.mujoco/mujoco200/bin;
# LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
import os

environ_var_names = ['PYTHONUNBUFFERED', 'LD_LIBRARY_PATH', 'LD_PRELOAD']


def set_mujoco_environ_vars():
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/home/michael/.mujoco/mujoco200/bin'
    os.environ['LD_PRELOAD'] = "/usr/lib/x86_64-linux-gnu/libGLEW.so"

def show_mujoco_environ_var():
    for name in environ_var_names:
        if name in os.environ:
            print(os.environ[name])
        else:
            print("{} is not set yet".format(name))
