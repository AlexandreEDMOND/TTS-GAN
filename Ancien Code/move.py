import scipy
import os
import random

def copie(input_file, input_folder, output_folder, nb=5000):
    file_list = open(input_file, 'r')
    for i in file_list.readlines():
        os.system('''mv "'''+input_folder+'"/"'+i[:-1]+'" "'+output_folder+'"/"'+i[:-1]+'''"''')
    file_list.close()

if '__main__' == __name__:
    copie('listMat.txt', '/home/user/Documents/DonnéesEntrainement', '/home/user/Documents/DonnéesTest', 5000)
    print("Done")