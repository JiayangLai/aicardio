import os


def check_n_mkdir(path_in):
    if os.path.exists(path_in):
        print(path_in,'already exsit.')
    else:
        os.mkdir(path_in)
        print(path_in, 'created.')


def chk_mk_model(net_name):
    path = "./models_saved/" + net_name
    check_n_mkdir(path)
