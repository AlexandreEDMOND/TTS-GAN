import matplotlib.pyplot as plt
import numpy as np
import os
import scipy


def get_data_from_file(filepath, seq_len):
    data = scipy.io.loadmat(filepath)
    return data['voltage'][0][:seq_len].astype(np.float64), data['current'][0][:seq_len].astype(np.float64), data['f1'][0][:seq_len].astype(np.float64), data['f2'][0][:seq_len].astype(np.float64), data['f3'][0][:seq_len].astype(np.float64)


def generate_data(dossier, selected_data, seq_len=1900):

    data = []
    
    for file in os.listdir(dossier):
        filepath = os.path.join(dossier, file)
        if os.path.isfile(filepath):
            data_v, data_c, data_f1, data_f2, data_f3 = get_data_from_file(filepath, seq_len)
            
            if selected_data == 'voltage':
                list_data_from_file = [data_v]
            elif selected_data == 'current':
                list_data_from_file = [data_c]
            elif selected_data == 'f1':
                list_data_from_file = [data_f1]
            elif selected_data == 'f2':
                list_data_from_file = [data_f2]
            elif selected_data == 'f3':
                list_data_from_file = [data_f3]

            data.append(list_data_from_file)

    nb_files = len(data)
    return np.array(data).reshape(nb_files, seq_len)

def affichage_data(data, seq_len=1900, num_signal_display=5):

    # Diviser le plot
    _, axs = plt.subplots(num_signal_display, 1, figsize=(12, 2*num_signal_display), sharex=True)

    # Affichage du signal réel
    for i in range(num_signal_display):
        axs[i].plot(data[np.random.randint(0, data.shape[0])][:seq_len], label=f'Signal réel {i+1}')
        axs[i].grid()
    
    plt.suptitle("Signal")
    plt.show()
