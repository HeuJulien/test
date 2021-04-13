#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 15:13:39 2019

@author: julienh
"""
from import_data import import_data_new, import_data_CO2_new
from descriptors import create_data_SOAP_new, create_is_train_new, create_scaler_new,scale_descriptors_new, create_PCA_subdivided_new, create_PCA_bulk_new, apply_PCA_subdivided_new, apply_PCA_bulk_new, create_data_SOAP_quippy, apply_PCA_subdivided_quippy, create_PCA_subdivided_quippy, security_barrier,create_data_ACSF_new,create_data_ACSF_runner
from predictors import energy_predictor,cross_validation, energy_predictor_behler, create_model_bulk_new,create_scaler_energy_new,train_NN_bulk_new, create_individual_models_bulk_new, average_weights, save_model, load_model, delete_weights,create_model_extended, second_train_NN_bulk_new, average_weights_SWA
from graphics import make_report,plot_error
from pretraitement import get_N_structures_new, get_pairwise_distances_new,plot_pairwise_distances_new, plot_SOAP_width_new,calc_etas_behler
from test_on_model import plot_loss_over_epoch_new, results_NN, plot_weight_distribution_new, plot_individual_outputs_new
from monte_carlo_simu import generate_data_MC_CO2_new,generate_data_MC_write_file_new,generate_data_MC_CO2_runner_new,test_RuNNer,extend_system, generate_data_MC_cell_list
from graph_MC import plot_pie_chart_new, plot_compare_hist_energy, plot_energy_over_time,plot_hist_std,plot_compare_g_of_r,first_neighbors, plot_compare_angles_water

import datetime
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import ase
import tqdm
import keras









metadata_system = {
'datetime':datetime.datetime.now().isoformat(),

'system_name':'CO2', #Zundel or CO2
'energy_calculation_method':'DFT', #fitted force field on CC for Zundel DFT for CO2

#'file_name':'/home/julienh/Desktop/quantum_dyn_NN/data/CO2_clean/9.8/merge_2000K_2500K/',       
'file_name':'/home/julienh/Desktop/quantum_dyn_NN/data/CO2_clean/9.8/2000K/2-run/', 
#'file_name':'/home/julienh/Desktop/quantum_dyn_NN/data/CO2_clean/9.2/2000K/1-run/',      
#'file_name':'/media/julienh/Einstein Re/CO2/8.82/3000K/',       
#'file_name':'/media/julienh/Einstein Re/CO2/9.8/2000K/',       
#'file_name':'/home/julienh/Desktop/quantum_dyn_NN/data/timothee/',

'trajec_name':'TRAJEC_db.xyz',
'energy_name':'ENERGIES_db',
        
#'is_MC':True, #Wether or not doing MC simulations (only for Zundel)



# Algo voisins
# calculs grandes Rc
# Changer refus en dessous pourcentage au lieu borne
        # Automatiser choix bins





"path_to_output":"../output/",
'folder_to_downfolded':'../downfolded_zundel/',  

'units_energy':'Hartree',
'units_distance':'Bohr',
'box_size':[9.8,9.8,9.8],
#'box_size':[9.2,9.2,9.2],
#'box_size':[13,13,13],

}


metadata_descriptor = {
'descriptor_type':'SOAP',               #Choice of descriptor
'rbf':'gto', #gto or polynomial #Usefull only with dscribe
'module':'dscribe', #quippy or dscribe

'scale_b4_PCA':True,

'architecture_PCA':'bulk', #bulk or subdivided, you HAVE TO choose subdivided if you choose subdivided architecture in the NN, but you can have a subdivided PCA with a bulk NN
'do_PCA':False,
'N_PCA':25,                            #In the bulk architecture, the number of dimension of the PCA
'N_PCA_per_inter':30,       #In the subdivided architecture, the number of dimension per interaction of the PCA

'scale_after_PCA':False,

#SOAP


#'scaler':None,             #False if no scaling, None if scaling
'sigma_SOAP':0.002,
'rcut':4.510879909123032,#5.0,#6.4,#4.510879909123032, #Add a parameter to check dimension => Probably better to impose that file are in certain units...
'nmax':4,
'lmax':4,        

'scaler_desc_b4_PCA':'standard',
'scaler_desc':'standard', #minmax or standard
'scaler_energy':'minmax',#'minmax', #minmax or standard or None (None is broken for diff sizes)
}

metadata_NN = {
        
'memory_steps':0,       #int for number of step to use before, (0 == no memory) NO NEGATIVE NUMBER...
'architecture_NN':'bulk', # bulk / subdivised / subdivised_sum
'average_weight':False,
'dropout':0,#0.013940628705723705,
'recurrent_dropout':0,
'test_size':0.1,
'use_valid_split':False,
'validation_split':0.10,

'period_SWA':20,
'N_period_SWA':10,
'cyclic_lr':5e-4,

'activation_function':'tanh',
'loss_function':'mean_squared_error',
#'loss_function':'custom',
'optimizer':'Adam', 
'learning_rate':0.004,#0.0004024912351654585,
'update_lr_factor':0.6,#0.6923843290184596,
'min_lr':2e-6,#2.724245435177284e-05,
'batch_size':32,                                                   #64
'epochs':5000,
'patience':25,#25,
'N_nodes_per_specie':[30,30],#Used if bulk
'N_node_per_inter':[20,20], #Used if subdived(_sum)
'N_node_final':[30,30], #Used if subdived(_sum)
        
'rate_regulizer_l2':0.000003,#0.00033205147667404776,#0.0005
}

metadata_MC = {
'N_MC_points':1000,
#'initial_molec':molecs[4000],
#'initial_energy':energies[4000],
'algo':'one_part',#all_part one_part
'initial_molec_id':4000,
'temperature':2000,
'delta':0.12,#0.009,#0.1,
'N_models':2,
'threshold_std_models':1000000, #eV
'N_box_MC':10,
'N_box_DFT':10,
        }

#Create folder for outputs 
os.mkdir(metadata_system['path_to_output']+metadata_system['datetime'])

all_meta = {'metadata_system':metadata_system,'metadata_descriptor':metadata_descriptor,'metadata_NN':metadata_NN,'metadata_MC':metadata_MC}
with open(metadata_system['path_to_output']+metadata_system['datetime']+'/metadatas','w') as f:
    for metadata_name in all_meta.keys():
        pickle.dump(all_meta[metadata_name],open(metadata_system['path_to_output']+metadata_system['datetime']+'/'+metadata_name,'wb'))
        for key in all_meta[metadata_name].keys():
            f.write(key+' = '+str(all_meta[metadata_name][key])+'\n')
        f.write('\n \n \n')

# Import file
#energies, molecs, elements, N_part, metadata_system = import_data_new(metadata_system)
energies, molecs, elements, N_part, metadata_system = import_data_CO2_new(metadata_system)
sys.exit()
"""
percent = 0.2
mask_E =  np.all([energies> np.percentile(energies,percent),energies<np.percentile(energies,100-percent)],axis=0)

energies = energies[mask_E]
molecs = molecs[mask_E]
elements = elements[mask_E]
N_part = N_part[mask_E]
metadata_system['N_config'] = energies.size
"""


##### PLOTS DISTANCES #####
# Numbers of config per size
_, metadata_system['N_config_per_size'] = get_N_structures_new(molecs)

if False:
    # Get pairwise distances
    all_species_distances, different_element_distances = get_pairwise_distances_new(molecs)
    
    # Used to make plots (then save them in file
    plot_pairwise_distances_new(molecs, all_species_distances, different_element_distances,metadata_system)
    
    # Used to plot sigma_SOAP compared to short distances
    plot_SOAP_width_new(molecs,all_species_distances, different_element_distances,metadata_descriptor,metadata_system,percentile=1,range_sigma=2,nb_try_sigma=7)
    
    # No need to keep those variables
    del all_species_distances, different_element_distances
    plt.close('all')

if True:
    all_species_distances, different_element_distances = get_pairwise_distances_new(molecs)
    etas = calc_etas_behler(different_element_distances,Rc=metadata_descriptor['rcut'],N_func=6)
    del all_species_distances, different_element_distances
    descriptors, metadata_descriptor = create_data_ACSF_runner(molecs, N_part, metadata_system, metadata_descriptor, elements,etas,energies,only_read=False)

##### DESCRIPTORS #####
# Create descriptors from positions
if False:
    if metadata_descriptor['module'] == 'dscribe':
        descriptors, metadata_descriptor = create_data_SOAP_new(molecs, N_part, metadata_system, metadata_descriptor,elements)
    elif metadata_descriptor['module'] == 'quippy':
        descriptors, metadata_descriptor = create_data_SOAP_quippy(molecs, N_part, metadata_system, metadata_descriptor,elements)
    else:
        sys.exit('Must choose module from either quippy or dscribe')

# Separation train/test
is_train = pickle.load(open('/home/julienh/Desktop/is_train','rb'))
#is_train = create_is_train_new(N_part,metadata_system, metadata_NN)
#is_train[:100] = True

if metadata_descriptor['scale_b4_PCA']:
    
    # Create scaler before PCA (not a mandatory step, probably better not to do it ?)
    scalers_b4_PCA = create_scaler_new(descriptors,elements,is_train,metadata_system,metadata_descriptor['scaler_desc_b4_PCA'],eof='b4')
    
    # Apply scaling (if wanted)
    descriptors = scale_descriptors_new(descriptors,elements,scalers_b4_PCA,metadata_system)


# Creation and application of the PCA (not mandatory but highly recommanded)      
if metadata_descriptor['do_PCA'] :   
    if metadata_descriptor['architecture_PCA'] == 'bulk':
        pcas = create_PCA_bulk_new(descriptors,elements,is_train,metadata_descriptor,metadata_system)
        descriptors = apply_PCA_bulk_new(descriptors,elements,pcas,metadata_descriptor)
    elif metadata_descriptor['architecture_PCA'] == 'subdivided':
        if metadata_descriptor['module'] == 'dscribe':
            pcas = create_PCA_subdivided_new(descriptors,elements,is_train,metadata_descriptor,metadata_system)
            descriptors = apply_PCA_subdivided_new(descriptors,elements,pcas,metadata_descriptor)
        elif metadata_descriptor['module'] == 'quippy':
            pcas = create_PCA_subdivided_quippy(descriptors,elements,is_train,metadata_descriptor,metadata_system)
            descriptors = apply_PCA_subdivided_quippy(descriptors,elements,pcas,metadata_descriptor)
        else:
            sys.exit('Must choose module from either quippy or dscribe')
    else:
        exit("Wrong architecture_PCA name")

if metadata_descriptor['scale_after_PCA']:
                                
    # Create scaler after PCA (almost mandatory)
    scalers_after_PCA = create_scaler_new(descriptors,elements,is_train,metadata_system,metadata_descriptor['scaler_desc'],eof='after')
        
    # Apply scaling 
    descriptors = scale_descriptors_new(descriptors,elements,scalers_after_PCA,metadata_system)

##### NEURAL NETWORK #####
# Create the scaler of the energies
energy_scaler = create_scaler_energy_new(energies,N_part,is_train,elements,metadata_descriptor,metadata_system)

models = np.empty(metadata_MC['N_models'],dtype=object)
individual_models = np.zeros((metadata_MC['N_models'],metadata_system['unique_elements'].size),dtype=object)
historys = np.empty(metadata_MC['N_models'],dtype=object)
hystorys_SWA = np.empty(metadata_MC['N_models'],dtype=object)

for i_model in range(metadata_MC['N_models']):
    models[i_model] = create_model_bulk_new(descriptors.shape[2], elements, N_part, metadata_system, metadata_NN)

    historys[i_model] = train_NN_bulk_new(descriptors, energies, is_train, elements, N_part, models[i_model], energy_scaler, metadata_NN,metadata_system,save_weights=False,no_update_lr=False)

    hystorys_SWA[i_model] = second_train_NN_bulk_new(descriptors, energies, is_train, elements, N_part, models[i_model], energy_scaler, metadata_NN,metadata_system)

    average_weights_SWA(models[i_model], metadata_system, metadata_NN)
    
    plot_loss_over_epoch_new(historys[i_model], metadata_system,eof='_model_{}'.format(i_model),history_SWA = hystorys_SWA[i_model])
    
    #Plot energy/atom & histogram error/atom
    results_NN(descriptors, energies, is_train, elements, N_part, models[i_model], energy_scaler,metadata_system,eof='_model_{}'.format(i_model))
    
    # Plot weight distribution
    plot_weight_distribution_new(models[i_model], metadata_NN, metadata_system,eof='_model_{}'.format(i_model))
#for i_model in range(metadata_MC['N_models']):
    # Get individual models for each element
    individual_models[i_model] =  create_individual_models_bulk_new(models[i_model], metadata_NN, metadata_system)
    
    # Plot individual energies
    plot_individual_outputs_new(descriptors, N_part, elements, individual_models[i_model],  metadata_system,eof='_model_{}'.format(i_model))

    # Save model
    save_model(models[i_model],metadata_system,eof='_'+str(i_model))
    
    for i_elem in range(individual_models.shape[1]):
        save_model(individual_models[i_model,i_elem],metadata_system,eof='_elem{}_{}'.format(metadata_system['unique_elements'][i_elem], i_model))
    
    delete_weights(metadata_system['path_to_output']+metadata_system['datetime'])

    """
    if metadata_NN['average_weight']:
        average_weights(models[i_model], historys[i_model], metadata_system, metadata_NN)
        
        #Plot energy/atom & histogram error/atom
        results_NN(descriptors, energies, is_train, elements, N_part, models[i_model], energy_scaler,metadata_system, eof='_model_{}_averaged_weights'.format(i_model))
        
        # Plot weight distribution
        plot_weight_distribution_new(models[i_model], metadata_NN, metadata_system, eof='_model_{}_averaged_weights'.format(i_model))
        
        # Get individual models for each element
        individual_models =  create_individual_models_bulk_new(models[i_model], metadata_NN, metadata_system)
        
        # Plot individual energies
        plot_individual_outputs_new(descriptors, N_part, elements, individual_models,  metadata_system, eof='_model_{}_averaged_weights'.format(i_model))

        # Save model
        save_model(models[i_model],metadata_system,eof='_{}_averaged'.format(i_model))
        for i_elem in range(individual_models.size):
            save_model(individual_models[i_elem],metadata_system,eof='_elem{}_{}_averaged'.format(metadata_system['unique_elements'][i_elem], i_model))
        # Delete weights
        delete_weights(metadata_system['path_to_output']+metadata_system['datetime'])
    
        keras.backend.clear_session()
    """
    
    plt.close('all')


all_meta = {'metadata_system':metadata_system,'metadata_descriptor':metadata_descriptor,'metadata_NN':metadata_NN,'metadata_MC':metadata_MC}
with open(metadata_system['path_to_output']+metadata_system['datetime']+'/metadatas','w') as f:
    for metadata_name in all_meta.keys():
        pickle.dump(all_meta[metadata_name],open(metadata_system['path_to_output']+metadata_system['datetime']+'/'+metadata_name,'wb'))
        for key in all_meta[metadata_name].keys():
            f.write(key+' = '+str(all_meta[metadata_name][key])+'\n')
        f.write('\n \n \n')
sys.exit()

"""

folder_name = '/home/julienh/Desktop/quantum_dyn_NN/output/2020-10-05T01:07:03.151935'
folder_name = metadata_system['path_to_output']+metadata_system['datetime']


models = np.zeros((metadata_MC['N_models']),dtype=object)
individual_models = np.zeros((metadata_MC['N_models'],metadata_system['unique_elements'].size),dtype=object)
for i_model in range(metadata_MC['N_models']):
    for i_elem in range(metadata_system['unique_elements'].size):
        if metadata_NN['average_weight']:
            individual_models[i_model,i_elem] = load_model(folder_name+'/model_elem{}_{}_averaged.json'.format(metadata_system['unique_elements'][i_elem], i_model),
                                         folder_name+'/model_weights_elem{}_{}_averaged.h5'.format(metadata_system['unique_elements'][i_elem],i_model))
        else:
            individual_models[i_model] = load_model(folder_name+'/model_elem{}_{}.json'.format(metadata_system['unique_elements'][i_elem],i_model),
                                         folder_name+'/model_weights_elem{}_{}.h5'.format(metadata_system['unique_elements'][i_elem],i_model))

for i_model in range(metadata_MC['N_models']):
    if metadata_NN['average_weight']:
        models[i_model] = load_model(folder_name+'/model_{}_averaged.json'.format(i_model),
                                     folder_name+'/model_weights_{}_averaged.h5'.format(i_model))
    else:
        models[i_model] = load_model(folder_name+'/model_{}.json'.format(i_model),
                                     folder_name+'/model_weights_{}.h5'.format(i_model))
"""
##### MC SIMULATIONS #####






#MC_positions, MC_molecs, MC_energies, MC_accepted, MC_try_configs, MC_std_models = generate_data_MC_CO2_new(metadata_MC,metadata_descriptor,scalers_b4_PCA,scalers_after_PCA,pcas,models,energy_scaler,metadata_system)


hists_desc, is_accepted_desc = security_barrier(descriptors[is_train],elements[is_train], metadata_system)

metadata_MC['initial_molec'],metadata_MC['initial_energy'] = molecs[metadata_MC['initial_molec_id']], energies[metadata_MC['initial_molec_id']]
#metadata_MC2['initial_molec'],metadata_MC2['initial_energy'] = molecs[metadata_MC['initial_molec_id']], energies[metadata_MC['initial_molec_id']]

generate_data_MC_write_file_new(metadata_MC,metadata_descriptor, scalers_b4_PCA,None,None,models,energy_scaler,metadata_system,individual_models=individual_models, min_dist = 0.85,func_energy ='calc_energy_RuNNer_mix', folders_RuNNer = np.array(['/home/julienh/Downloads/RuNNer-master/examples/use_for_desc',]),hists_desc=None, is_accepted_desc=None)


#generate_data_MC_write_file_new(metadata_MC,metadata_descriptor, scalers_b4_PCA,scalers_after_PCA,pcas,models,energy_scaler,metadata_system,individual_models=individual_models, hists_desc = hists_desc, is_accepted_desc = is_accepted_desc)
generate_data_MC_write_file_new(metadata_MC,metadata_descriptor, scalers_b4_PCA,None,None,models,energy_scaler,metadata_system,individual_models=individual_models, hists_desc = hists_desc, is_accepted_desc = is_accepted_desc)
#generate_data_MC_write_file_new(metadata_MC,metadata_descriptor, scalers_b4_PCA,scalers_after_PCA,pcas,models,energy_scaler,metadata_system,individual_models=individual_models)
#generate_data_MC_write_file_new(metadata_MC2,metadata_descriptor2, scalers_b4_PCA2,scalers_after_PCA2,pcas2,models2,energy_scaler2,metadata_system2,individual_models=individual_models2, hists_desc = hists_desc2, is_accepted_desc = is_accepted_desc2)

sys.exit()
    
    
def import_session(folder_name):
    scalers_b4_PCA = pickle.load(open(folder_name+'/scalersb4','rb'))
    scalers_after_PCA = pickle.load(open(folder_name+'/scalersafter','rb'))
    pcas = pickle.load(open(folder_name+'/pcas','rb'))
    energy_scaler = pickle.load(open(folder_name+'/energy_scaler','rb'))
    metadata_system = pickle.load(open(folder_name+'/metadata_system','rb'))
    metadata_descriptor = pickle.load(open(folder_name+'/metadata_descriptor','rb'))
    metadata_NN = pickle.load(open(folder_name+'/metadata_NN','rb'))
    metadata_MC = pickle.load(open(folder_name+'/metadata_MC','rb'))
    
    hists_desc = pickle.load(open(folder_name+'/secu_hists','rb'))
    is_accepted_desc = pickle.load(open(folder_name+'/secu_accepted','rb'))
    
    models = np.zeros((metadata_MC['N_models']),dtype=object)
    individual_models = np.zeros((metadata_MC['N_models'],metadata_system['unique_elements'].size),dtype=object)
    for i_model in range(metadata_MC['N_models']):
        for i_elem in range(metadata_system['unique_elements'].size):
            if metadata_NN['average_weight']:
                individual_models[i_model,i_elem] = load_model(folder_name+'/model_elem{}_{}_averaged.json'.format(metadata_system['unique_elements'][i_elem], i_model),
                                             folder_name+'/model_weights_elem{}_{}_averaged.h5'.format(metadata_system['unique_elements'][i_elem],i_model))
            else:
                individual_models[i_model] = load_model(folder_name+'/model_elem{}_{}.json'.format(metadata_system['unique_elements'][i_elem],i_model),
                                             folder_name+'/model_weights_elem{}_{}.h5'.format(metadata_system['unique_elements'][i_elem],i_model))
    
    for i_model in range(metadata_MC['N_models']):
        if metadata_NN['average_weight']:
            models[i_model] = load_model(folder_name+'/model_{}_averaged.json'.format(i_model),
                                         folder_name+'/model_weights_{}_averaged.h5'.format(i_model))
        else:
            models[i_model] = load_model(folder_name+'/model_{}.json'.format(i_model),
                                         folder_name+'/model_weights_{}.h5'.format(i_model))

    return scalers_b4_PCA, scalers_after_PCA, pcas, energy_scaler, energy_scaler, metadata_system, metadata_descriptor, metadata_NN, metadata_MC, hists_desc, is_accepted_desc, models, individual_models

scalers_b4_PCA, scalers_after_PCA, pcas, energy_scaler, energy_scaler, metadata_system, metadata_descriptor, metadata_NN, metadata_MC, hists_desc, is_accepted_desc, models, individual_models = import_session(folder_name)
#scalers_b4_PCA2, scalers_after_PCA2, pcas2, energy_scaler2, energy_scaler2, metadata_system2, metadata_descriptor2, metadata_NN2, metadata_MC2, hists_desc2, is_accepted_desc2, models2, individual_models2 = import_session(folder_name)

    
#save soap and add metadatas

    
extension = np.array([2,2,2])
N_extension = extension[0]*extension[1]*extension[2]
metadata_MC['initial_molec'],metadata_MC['initial_energy'] = molecs[metadata_MC['initial_molec_id']], energies[metadata_MC['initial_molec_id']]

molec_extended,energy_extended, elements_extended= extend_system(metadata_MC['initial_molec'],metadata_MC['initial_energy'],metadata_system,extension=extension)
"""
models = np.arange(2,dtype=object)
for i in range(models.size):
#    file_to_weights = '/home/julienh/Desktop/reports/3/2020-05-21T01:30:34.068146/model_weights_{}_averaged.h5'.format(i)
    file_to_weights = folder_name+'/model_weights_{}_averaged.h5'.format(i)
    models[i] = create_model_extended(N_part,90,elements_extended,N_extension,file_to_weights,metadata_system,metadata_NN)
"""
models = np.zeros((metadata_MC['N_models']),dtype=object)
individual_models = np.zeros((metadata_MC['N_models'],metadata_system['unique_elements'].size),dtype=object)
for i_model in range(metadata_MC['N_models']):
    for i_elem in range(metadata_system['unique_elements'].size):
        if metadata_NN['average_weight']:
            individual_models[i_model,i_elem] = load_model(folder_name+'/model_elem{}_{}_averaged.json'.format(metadata_system['unique_elements'][i_elem], i_model),
                                         folder_name+'/model_weights_elem{}_{}_averaged.h5'.format(metadata_system['unique_elements'][i_elem],i_model))
        else:
            individual_models[i_model] = load_model(folder_name+'/model_elem{}_{}.json'.format(metadata_system['unique_elements'][i_elem],i_model),
                                         folder_name+'/model_weights_elem{}_{}.h5'.format(metadata_system['unique_elements'][i_elem],i_model))

for i_model in range(metadata_MC['N_models']):
    if metadata_NN['average_weight']:
        models[i_model] = load_model(folder_name+'/model_{}_averaged.json'.format(i_model),
                                     folder_name+'/model_weights_{}_averaged.h5'.format(i_model))
    else:
        models[i_model] = load_model(folder_name+'/model_{}.json'.format(i_model),
                                     folder_name+'/model_weights_{}.h5'.format(i_model))

metadata_system['box_size'] = metadata_system['box_size']*extension
metadata_MC['N_models'] = 1
metadata_MC['delta'] = 0.09 #0.009
metadata_MC['N_MC_points'] = 1000
metadata_MC['initial_molec'],metadata_MC['initial_energy'] = molec_extended, 0

generate_data_MC_cell_list(metadata_MC,metadata_descriptor, scalers_b4_PCA,scalers_after_PCA,pcas,models,energy_scaler,metadata_system,individual_models=individual_models, min_dist = 0.0,func_energy ='calc_energy_3', folders_RuNNer = np.array(['/home/julienh/Downloads/RuNNer-master/examples/example04',]),hists_desc=None, is_accepted_desc=None)



import time
ext_numbers = np.arange(2,6,1)
timers_mean = np.zeros(ext_numbers.size)
timers_std = np.zeros(ext_numbers.size)
timers_1 = np.zeros(ext_numbers.size)
timers_all = np.zeros(ext_numbers.size)

metadata_system['box_size'] = np.array([9.8,9.8,9.8])
for i_ext in range(ext_numbers.size):
    metadata_system['box_size'] = np.array([9.8,9.8,9.8])

    metadata_MC['initial_molec'],metadata_MC['initial_energy'] = molecs[metadata_MC['initial_molec_id']], energies[metadata_MC['initial_molec_id']]
    
    molec_extended,energy_extended, elements_extended= extend_system(metadata_MC['initial_molec'],metadata_MC['initial_energy'],metadata_system,extension=np.ones(3,dtype=int)*ext_numbers[i_ext])
    metadata_MC['initial_molec'],metadata_MC['initial_energy'] = molec_extended, 0
    
    metadata_system['box_size'] *= np.ones(3,dtype=int)*ext_numbers[i_ext]
    timers_mean[i_ext], timers_std[i_ext] = generate_data_MC_cell_list(metadata_MC,metadata_descriptor, scalers_b4_PCA,scalers_after_PCA,pcas,models,energy_scaler,metadata_system,individual_models=individual_models, min_dist = 0.0,func_energy ='calc_energy_3', folders_RuNNer = np.array(['/home/julienh/Downloads/RuNNer-master/examples/example04',]),hists_desc=None, is_accepted_desc=None)
    
plt.figure()
plt.xlabel('N_part / 96')
plt.ylabel('time per MC step')
plt.errorbar(ext_numbers**3,timers_mean,yerr=timers_std,fmt='o-')



folder_name = metadata_system['path_to_output']+metadata_system['datetime']
MC_molecs_vec = ase.io.read(folder_name+'/positions.xyz',index='::100')
MC_molecs = np.zeros(len(MC_molecs_vec),dtype=object)
for i in tqdm.tqdm(range(len(MC_molecs_vec))):
    MC_molecs_vec[i].pbc=True
    MC_molecs_vec[i].set_cell(metadata_system['box_size'])
    MC_molecs[i] = MC_molecs_vec[i]
del MC_molecs_vec

"""
MC_molecs_vec = ase.io.read(folder_name+'/positions_refused.xyz',index=':')
MC_molecs_refused = np.zeros(len(MC_molecs_vec),dtype=object)
for i in tqdm.tqdm(range(len(MC_molecs_vec))):
    MC_molecs_vec[i].pbc=True
    MC_molecs_vec[i].set_cell(metadata_system['box_size'])
    MC_molecs_refused[i] = MC_molecs_vec[i]
del MC_molecs_vec
"""
MC_accepted = np.loadtxt(folder_name+'/accepted')[:]
MC_energies = np.loadtxt(folder_name+'/energies')[:]
MC_std_models = np.loadtxt(folder_name+'/std')[:]

#metadata_MC['initial_molec'] = MC_molecs[-1]
#metadata_MC['initial_energy'] = MC_energies[-1]

plot_pie_chart_new(MC_accepted,metadata_system,metadata_MC,eof='_all')

plot_hist_std(MC_std_models,metadata_system,metadata_MC,eof='_all')

plot_compare_g_of_r(MC_molecs, molecs[::2], metadata_system, metadata_MC,eof='_all')

first_neighbors(MC_molecs, molecs[::2], metadata_system, metadata_MC,eof='',max_N_neigh=5)

plot_compare_angles_water(MC_molecs[::2], molecs[::6], metadata_system,metadata_MC,eof='',middle_elem=6,neigh_elem=8,percentile=0.2)




"""
sigmas = np.linspace(0.001,1,12)
all_stds = np.empty(sigmas.shape)
for i_sigma in range(sigmas.size):
    metadata_descriptor['sigma_SOAP'] = sigmas[i_sigma]    
    descriptors, metadata_descriptor = create_data_SOAP_quippy(molecs, N_part, metadata_system, metadata_descriptor,elements)
    ranges,stds = view_std_range(descriptors,elements)
    all_stds[i_sigma] = stds.mean()
    print(i_sigma)
    print(sigmas[i_sigma])
    print(all_stds[i_sigma]) 
    del(descriptors)
"""


"""
import numpy as np
import matplotlib.pyplot as plt

folder_name = '/home/julienh/Desktop/quantum_dyn_NN/output/2020-10-05T01:07:03.151935'
accepted = np.loadtxt(folder_name+'/accepted')
print((accepted!=0).mean())

MC_energies = np.loadtxt(folder_name+'/energies')
plt.figure()
plt.plot(MC_energies)

std_models = np.loadtxt(folder_name+'/std')
plt.figure()
plt.plot(std_models)

individual_energies = np.empty((299_999,2,96))

file =  open(folder_name+'/individual_contributions')

for i_config in range(individual_energies.shape[0]):
    for i_part in range(individual_energies.shape[2]):
        line = file.readline()
        individual_energies[i_config,0,i_part] = float(line.strip('[] \n'))
        
    _ = file.readline()
    
    for i_part in range(individual_energies.shape[2]):
        line = file.readline()
        individual_energies[i_config,1,i_part] = float(line.strip('[] \n'))
          

file.close()
"""


#generate_data_MC_write_file_new(metadata_MC,metadata_descriptor,None,scalers_after_PCA,None,models,energy_scaler,metadata_system)



#generate_data_MC_CO2_runner_new(metadata_MC,metadata_system,file_name='example02')

#pickle.dump(MC_positions,open(metadata_system['path_to_output']+metadata_system['datetime']+'/MC_positions','wb'))
#pickle.dump(MC_energies,open(metadata_system['path_to_output']+metadata_system['datetime']+'/MC_energies','wb'))
#pickle.dump(MC_accepted,open(metadata_system['path_to_output']+metadata_system['datetime']+'/MC_accepted','wb'))
#pickle.dump(MC_try_configs,open(metadata_system['path_to_output']+metadata_system['datetime']+'/MC_try_configs','wb'))
#pickle.dump(MC_std_models,open(metadata_system['path_to_output']+metadata_system['datetime']+'/MC_std_models','wb'))

#Importer les poids pour faire les models et comparer




#Pour les plots rajouter un argument eof qui permet de rajouter ce qu'on veut (mask std > 0.2 par exemple)
##### PLOTS ######
#file_name = metadata_system['path_to_output']+metadata_system['datetime']+'/positions.xyz'

"""

N_part = np.ones(MC_molecs.size,dtype=int)*MC_molecs[0].get_number_of_atoms()
elements = np.tile(MC_molecs[0].get_atomic_numbers(),(MC_molecs.size,1))

metadata_system['N_config'] = MC_molecs.size
metadata_system['unique_sizes'] = np.unique(N_part)
metadata_system['unique_elements'] = np.unique(elements[elements!=0])

MC_descriptors, metadata_descriptor = create_data_SOAP_new(MC_molecs, N_part, metadata_system, metadata_descriptor,elements)

if metadata_descriptor['scale_b4_PCA']:

    MC_descriptors = scale_descriptors_new(MC_descriptors,elements,scalers_b4_PCA,metadata_system)
                         
if metadata_descriptor['do_PCA'] :   
    if metadata_descriptor['architecture_PCA'] == 'bulk':
        MC_descriptors = apply_PCA_bulk_new(MC_descriptors,elements,pcas,metadata_descriptor)
    elif metadata_descriptor['architecture_PCA'] == 'subdivided':
        MC_descriptors = apply_PCA_subdivided_new(MC_descriptors,elements,pcas,metadata_descriptor)
    else:
        exit("Wrong architecture_PCA name")
        
if metadata_descriptor['scale_after_PCA']:                        
    MC_descriptors = scale_descriptors_new(MC_descriptors,elements,scalers_after_PCA,metadata_system)


for i_model in range(models.size):
#    models[i_model] = load_model(folder_name+'/model_{}_averaged.json'.format(i_model),folder_name+'/model_weights_{}_averaged.h5'.format(i_model))
    
    individual_models =  create_individual_models_bulk_new(models[i_model], metadata_NN, metadata_system)
    
    # Plot individual energies
    plot_individual_outputs_new(MC_descriptors, N_part, elements, individual_models,  metadata_system, eof='_model_{}_averaged_weights'.format(i_model))
    #plot_individual_outputs_new(descriptors[~mask], N_part, elements, individual_models,  metadata_system, eof='_model_{}_averaged_weights'.format(i_model))
"""
##########################################################################
"""
metadata_system['file_name'] = '/home/julienh/Desktop/quantum_dyn_NN/data/CO2_clean/9.8/2000K/2-run/'    
energies, molecs, elements, N_part, metadata_system = import_data_CO2_new(metadata_system)
descriptors, metadata_descriptor = create_data_SOAP_new(molecs, N_part, metadata_system, metadata_descriptor,elements)

if metadata_descriptor['scale_b4_PCA']:

    descriptors = scale_descriptors_new(descriptors,elements,scalers_b4_PCA,metadata_system)
                         
if metadata_descriptor['do_PCA'] :   
    if metadata_descriptor['architecture_PCA'] == 'bulk':
        descriptors = apply_PCA_bulk_new(descriptors,elements,pcas,metadata_descriptor)
    elif metadata_descriptor['architecture_PCA'] == 'subdivided':
        descriptors = apply_PCA_subdivided_new(descriptors,elements,pcas,metadata_descriptor)
    else:
        exit("Wrong architecture_PCA name")
        
if metadata_descriptor['scale_after_PCA']:                        
    descriptors = scale_descriptors_new(descriptors,elements,scalers_after_PCA,metadata_system)



x_test = list(np.swapaxes(descriptors,0,1))
#y_test = energy_scaler.transform((energies[~is_train]/N_part[~is_train]).reshape(-1,1))*N_part[~is_train].reshape(-1,1)
mask_input_test = (elements != 0).astype(int).T

for i in range(len(mask_input_test)):
    x_test.append(mask_input_test[i])

predictions = np.zeros((metadata_MC['N_models'],energies.size))
for i_model in range(metadata_MC['N_models']):
    predictions[i_model] = (energy_scaler.inverse_transform(models[i_model].predict(x_test)/N_part.reshape(-1,1))*N_part.reshape(-1,1))[:,0]
#predictions = predictions[:,0]
#rmse_in_meV = np.sqrt((((predictions-energies.reshape(-1,1))/N_part)**2).mean())*1000
#print(rmse_in_meV)

##########
predictions_behler = pickle.load(open('/home/julienh/Desktop/quantum_dyn_NN/quantum_NN/RuNNer_predictions','rb'))
#np.sqrt((((predictions - energies)/96)**2).mean(axis=1))*1000
predictions = np.concatenate([predictions,predictions_behler])
    ######

size_box=200
boxes = np.arange(0,energies.size,size_box)
rmses =  np.zeros((predictions.shape[0],boxes.size-1))
rmses_std =  np.zeros((predictions.shape[0],boxes.size-1))

#aa =  np.zeros((boxes.size-1))

for i_box in range(boxes.size-1):
    rmses[:,i_box] = np.sqrt((((predictions[:,boxes[i_box]:boxes[i_box+1]] - energies[boxes[i_box]:boxes[i_box+1]])/96)**2).mean(axis=1))*1000
    rmses_std[:,i_box] = np.sqrt((((predictions[:,boxes[i_box]:boxes[i_box+1]] - energies[boxes[i_box]:boxes[i_box+1]])/96)**2).std(axis=1))*1000
 #   aa[i_box] = bons[boxes[i_box]:boxes[i_box+1]].mean()

def plot_thing(predictions):
    plt.figure()
    plt.title('Comparison error models')
    for i in range(predictions.shape[0]):
        if i < metadata_MC['N_models']:
            label = 'us'
        else:
            label='RuNNer'
        plt.errorbar(i,rmses[i].mean(),yerr=rmses[i].std(),fmt='D',label=label)
    plt.legend()
    plt.xlabel('model number')
    plt.ylabel('RMSE (meV/atom)')
    
    plt.figure()
    plt.title('Error over time')
    for i in range(predictions.shape[0]):
        if i < metadata_MC['N_models']:
            label = 'us'
            fmt = 'D'
        else:
            label='RuNNer'
            fmt = '*'
    
        plt.errorbar(np.arange(boxes.size-1),rmses[i],yerr = rmses_std[i],label=i,fmt=fmt)
    plt.legend()
    plt.xlabel('mean over 200 time steps')
    plt.ylabel('RMSE (meV/atom)')

#####
plot_thing(predictions)
diff = (predictions - energies)
plot_thing(predictions-diff.mean(axis=1).reshape(predictions.shape[0],1))




"""


def detect_defects(MC_molecs,threshold=1.7278273924964265):
    weird_id = np.empty(MC_molecs.size,dtype=object)
    for i_config in tqdm.tqdm(range(MC_molecs.size)):
        dist_mat_C = MC_molecs[i_config].get_all_distances()[:32,:32]
        weird_id[i_config] = np.unique(np.where(np.all([dist_mat_C<threshold,dist_mat_C!=0],axis=(0)))[0])
    return weird_id

def detect_outliers(descriptors,MC_descriptors):
    defects_ids = np.empty((MC_descriptors.shape[2],MC_descriptors.shape[0]),dtype=object)
    for i_desc in tqdm.tqdm(range(MC_descriptors.shape[2])):
        where_config, where_id = np.where(np.any([MC_descriptors[:,:,i_desc] > descriptors[:,:,i_desc].max(), MC_descriptors[:,:,i_desc] < descriptors[:,:,i_desc].min()],axis=(0)))
        for i_config in range(MC_molecs.shape[0]):
            defects_ids[i_desc,i_config] = where_id[where_config==i_config]

    defects_ids_tot = []#np.empty((MC_descriptors.shape[0]),dtype=object)
    for i_config in range(MC_molecs.shape[0]):
        defects_ids_tot.append([])
        for i_desc in range(MC_descriptors.shape[2]):
            for elem in defects_ids[i_desc,i_config]:
                defects_ids_tot[i_config].append(elem)
        defects_ids_tot[i_config] = np.unique(defects_ids_tot[i_config])
    defects_ids_tot = np.array(defects_ids_tot)
    return defects_ids, defects_ids_tot
#isin = [np.all(np.isin(weird_ids[i],defects_ids_tot[i]),axis=0) for i in range(weird_ids.shape[0])]
#same = np.array([np.all(defects_ids_tot[i] == weird_ids[i]) for i in range(weird_ids.shape[0])])

#########################################################################################################"
from dscribe.kernels import AverageKernel
re = AverageKernel(metric="linear")

# How many ids do we add per iteration
N_new_ids_per_iteration = 10

# max nummber of configs
max_N_configs = 40


# I start with no data
ids_kept_data = np.zeros(descriptors.shape[0],dtype=bool)

# I select N_new_id (different) random configurations
# I select a minimum of 2 for simplicity
ids_kept_data[np.random.choice(descriptors.shape[0],np.max([2,N_new_ids_per_iteration]),replace=False)]=True    



while ids_kept_data.sum()<max_N_configs:
    # I create the similarity kernel between the selected configs and the non-selected
    similarity_kernel = re.create(descriptors[ids_kept_data],descriptors[~ids_kept_data])
    
    # I identify the least similar configuration
    new_ids = np.argsort(similarity_kernel.max(axis=0))[:N_new_ids_per_iteration]
    
    # Those configurations are taken from descriptors[~ids_kept_data], so I adapt them
    new_ids = np.arange(descriptors.shape[0])[~ids_kept_data][new_ids]
    
    ids_kept_data[new_ids] = True
    print('There are {} chosen configurations, still need {}'.format(ids_kept_data.sum(),max_N_configs-ids_kept_data.sum()))
