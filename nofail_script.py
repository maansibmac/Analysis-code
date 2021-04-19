# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
Z = norm.ppf
from classes_text import *
import os
from scipy.stats import ttest_rel
import statistics


#### READ
# Your task is to fill in the [...] so the code works as explained in the comments

# Use print() and len() to understand the structure of the variables
# %%
# import os
# from classes_tharnal import *

# folder_name = 'test_18032021_2'
# pattern = f'sdt_trial.*\.hdf5$'

# patternc = re.compile(pattern)
# names = []

# one_data = pd.read_csv(f"/Users/manny/Documents/ProjectCold/expt4_py_nontactileColdnew/data/" + folder_name + "/data/temp_data.csv") 
# cold_bool = np.asarray(one_data['Cold'])
# touch_bool = np.asarray(one_data['Touch'])
# responses_bool = np.asarray(one_data['Responses'])

# print(one_data)

# for filename in os.listdir(f'/Users/manny/Documents/ProjectCold/expt4_py_nontactileColdnew/data/{folder_name}/videos/'):
#     if patternc.match(filename):
#         name, form = filename.split('.')
#         names.append(name)
#     else:
#         continue

# names.sort(key=natural_keys)
# print(names)

# failed_trials = []

# for i, n in enumerate(names):
#     if cold_bool[i] == 1:
#         dat_im = ReAnRaw(f'/Users/manny/Documents/ProjectCold/expt4_py_nontactileColdnew/data/{folder_name}/videos/{n}')
#         dat_im.datatoDic()
#         if dat_im.data['time_now'][-1][0] > 15.5 or dat_im.data['time_now'][-1][0] < 5: 
#             failed_trials.append(i)

# print('\n')
# print(failed_trials)

# cleaned_one_data = one_data.drop(failed_trials)
# print(cleaned_one_data)

# %% Old data
# import os
# from classes_tharnal import *

# folder_name = 'test_15122020_3'
# pattern = f'sdt_subj1_trial.*\.hdf5$'

# patternc = re.compile(pattern)
# names = []

# one_data = pd.read_csv(f"/Users/manny/Documents/ProjectCold/expt4_py_nontactileColdnew/data/" + folder_name + "/data/temp_data.csv") 
# cold_bool = np.asarray(one_data['Cold'])
# touch_bool = np.asarray(one_data['Touch'])
# responses_bool = np.asarray(one_data['Responses'])

# print(one_data)

# for filename in os.listdir(f'/Users/manny/Documents/ProjectCold/expt4_py_nontactileColdnew/data/{folder_name}/videos/'):
#     if patternc.match(filename):
#         name, form = filename.split('.')
#         names.append(name)
#     else:
#         continue

# names.sort(key=natural_keys)
# print(names)

# failed_trials = []

# for i, n in enumerate(names):
#     if cold_bool[i] == 1:
#         dat_im = ReAnRaw(f'/Users/manny/Documents/ProjectCold/expt4_py_nontactileColdnew/data/{folder_name}/videos/{n}')
#         dat_im.datatoDic()
#         if dat_im.data['time_now'][-1][0] > 15.5 or dat_im.data['time_now'][-1][0] < 5: 
#             failed_trials.append(i)

# print('\n')
# print(failed_trials)

# cleaned_one_data = one_data.drop(failed_trials)
# print(cleaned_one_data)

# %% COPY
# usable_data = ['test_19032021_2', 'test_19032021_5', 'test_23032021_2', 'test_29032021_1', 'test_30032021_1', 'test_30032021_2', 'test_31032021_1', 'test_31032021_2']

# copy_table_data = pd.read_csv(f"../../data/test_18032021_2/data/data_all.csv")
# folder_name = 'test_18032021_2'
# pattern = f'sdt_trial.*\.hdf5$'

# patternc = re.compile(pattern)
# names = []

# cold_bool = np.asarray(copy_table_data['Cold'])
# touch_bool = np.asarray(copy_table_data['Touch'])
# responses_bool = np.asarray(copy_table_data['Responses'])

# for filename in os.listdir(f'/Users/manny/Documents/ProjectCold/expt4_py_nontactileColdnew/data/{folder_name}/videos/'):
#     if patternc.match(filename):
#         name, form = filename.split('.')
#         names.append(name)
#     else:
#         continue

# names.sort(key=natural_keys)

# failed_trials = []

# for i, n in enumerate(names):
#     if cold_bool[i] == 1:
#         dat_im = ReAnRaw(f'/Users/manny/Documents/ProjectCold/expt4_py_nontactileColdnew/data/{folder_name}/videos/{n}')
#         dat_im.datatoDic()
#         if dat_im.data['time_now'][-1][0] > 15.5 or dat_im.data['time_now'][-1][0] < 5: 
#             failed_trials.append(i)

# cleaned_copy_table_data = copy_table_data.drop(failed_trials)

# cleaned_copy_table_data.loc[:,'Subject'] = 1
# cleaned_copy_table_data.to_csv("../../data/test_18032021_2/data/data_all.csv", index = False)

# for i, element in enumerate(usable_data):
#     copy_next_data = pd.read_csv(f"../../data/" + element + "/data/data_all.csv")
#     folder_name = element
#     pattern = f'sdt_trial.*\.hdf5$'

#     patternc = re.compile(pattern)
#     names = []

#     cold_bool = np.asarray(copy_next_data['Cold'])
#     touch_bool = np.asarray(copy_next_data['Touch'])
#     responses_bool = np.asarray(copy_next_data['Responses'])

#     for filename in os.listdir(f'/Users/manny/Documents/ProjectCold/expt4_py_nontactileColdnew/data/{folder_name}/videos/'):
#         if patternc.match(filename):
#             name, form = filename.split('.')
#             names.append(name)
#         else:
#             continue

#     names.sort(key=natural_keys)

#     failed_trials = []

#     for i, n in enumerate(names):
#         if cold_bool[i] == 1:
#             dat_im = ReAnRaw(f'/Users/manny/Documents/ProjectCold/expt4_py_nontactileColdnew/data/{folder_name}/videos/{n}')
#             dat_im.datatoDic()
#             if dat_im.data['time_now'][-1][0] > 15.5 or dat_im.data['time_now'][-1][0] < 5: 
#                 failed_trials.append(i)

#     cleaned_copy_next_data = copy_next_data.drop(failed_trials)
#     cleaned_copy_next_data.loc[:,'Subject'] = i + 2
#     cleaned_copy_ext_data.to_csv(f"../../data/" + element + "/data/data_all.csv", index = False)
#     cleaned_copy_table_data = cleaned_copy_table_data.append(cleaned_next_data, sort = False)

# print(cleaned_copy_table_data)
# %% 

# %% REAL
# Let's first import the file which contains the responses, conditions, watson's confidence and watson's hypotheses
usable_data = ['test_15122020_3', 'test_19032021_2', 'test_30032021_2', 'test_31032021_1', 'test_31032021_2']

table_data = pd.read_csv(f"../../data/test_15122020_2/data/data_all.csv")
table_data.loc[:,'Subject'] = 1
table_data.to_csv("../../data/test_15122020_2/data/data_all.csv", index = False)

for i, element in enumerate(usable_data):
    next_data = pd.read_csv(f"../../data/" + element + "/data/data_all.csv")
    next_data.loc[:,'Subject'] = i + 2
    next_data.to_csv(f"../../data/" + element + "/data/data_all.csv", index = False)
    table_data = table_data.append(next_data, sort = False)

# %% Check data
print(table_data)
print(table_data.columns) #name of headers
print(len(table_data)) # number of rows
print(table_data['Responses']) # check responses column
print(table_data['Subject'])

######## Plot and calculate sdt
# %% Parse data
# First we need to transform the data structure so we can easily manipulate it
# Let's use a function for this
def tableTosdtDoble(table, num_sdt):
    """
        Function to transform the table to an sdt table for manipulation and plotting
    """
    # Choose rows in which there's touch or not
    table_single_sdt = table.loc[table['Touch'] == num_sdt]

    # Choose rows with and without cold
    table_cold = table_single_sdt.loc[table_single_sdt['Cold'] == 1] # with cold trials
    table_nocold = table_single_sdt.loc[table_single_sdt['Cold'] == 0] # withOUT cold trials

    present_yes = table_cold.loc[table_cold['Responses'] == 1]
    present_no = table_cold.loc[table_cold['Responses'] == 0]
     
    absent_yes = table_nocold.loc[table_nocold['Responses'] == 1]
    absent_no = table_nocold.loc[table_nocold['Responses'] == 0]

    return present_yes, present_no, absent_yes, absent_no

# %%
# Generate conditions for each individual dataset
first_data = pd.read_csv(f"../../data/test_15122020_2/data/data_all.csv")

first_present_yest, first_present_not, first_absent_yest, first_absent_not = tableTosdtDoble(first_data, 1)
first_present_yesnt, first_present_nont, first_absent_yesnt, first_absent_nont = tableTosdtDoble(first_data, 0)

first_present_touch = [len(first_present_yest.loc[:, 'Responses']), len(first_present_not.loc[:, 'Responses'])]
first_absent_touch = [len(first_absent_yest.loc[:, 'Responses']), len(first_absent_not.loc[:, 'Responses'])]

first_present_notouch = [len(first_present_yesnt.loc[:, 'Responses']), len(first_present_nont.loc[:, 'Responses'])]
first_absent_notouch = [len(first_absent_yesnt.loc[:, 'Responses']), len(first_absent_nont.loc[:, 'Responses'])]

first_correc_present_touch = first_present_touch[0]/sum(first_present_touch)
first_correc_absent_touch = first_absent_touch[1]/sum(first_absent_touch)

first_correc_present_notouch = first_present_notouch[0]/sum(first_present_notouch)
first_correc_absent_notouch = first_absent_notouch[1]/sum(first_absent_notouch)

correc_present_touch_list = [first_correc_present_touch]
correc_absent_touch_list = [first_correc_absent_touch]

correc_present_notouch_list = [first_correc_present_notouch]
correc_absent_notouch_list = [first_correc_absent_notouch]

first_all_in = [first_correc_present_touch, first_correc_absent_touch, first_correc_present_notouch, first_correc_absent_notouch]

all_ind_all_in = np.asarray(first_all_in)
print (all_ind_all_in)

first_list = [first_present_touch, first_absent_touch, first_present_notouch, first_absent_notouch]
all_list = [first_list]

# %%
for z in usable_data:
    ind_data = pd.read_csv(f"../../data/" + z + "/data/data_all.csv")

    ind_present_yest, ind_present_not, ind_absent_yest, ind_absent_not = tableTosdtDoble(ind_data, 1)
    ind_present_yesnt, ind_present_nont, ind_absent_yesnt, ind_absent_nont = tableTosdtDoble(ind_data, 0)

    ind_present_touch = [len(ind_present_yest.loc[:, 'Responses']), len(ind_present_not.loc[:, 'Responses'])]
    ind_absent_touch = [len(ind_absent_yest.loc[:, 'Responses']), len(ind_absent_not.loc[:, 'Responses'])]

    ind_present_notouch = [len(ind_present_yesnt.loc[:, 'Responses']), len(ind_present_nont.loc[:, 'Responses'])]
    ind_absent_notouch = [len(ind_absent_yesnt.loc[:, 'Responses']), len(ind_absent_nont.loc[:, 'Responses'])]

    ind_correc_present_touch = ind_present_touch[0]/sum(ind_present_touch)
    ind_correc_absent_touch = ind_absent_touch[1]/sum(ind_absent_touch)

    ind_correc_present_notouch = ind_present_notouch[0]/sum(ind_present_notouch)
    ind_correc_absent_notouch = ind_absent_notouch[1]/sum(ind_absent_notouch)

    correc_present_touch_list.append(ind_correc_present_touch)
    correc_absent_touch_list.append(ind_correc_absent_touch)

    correc_present_notouch_list.append(ind_correc_present_notouch)
    correc_absent_notouch_list.append(ind_correc_absent_notouch)

    ind_all_in = [ind_correc_present_touch, ind_correc_absent_touch, ind_correc_present_notouch, ind_correc_absent_notouch]

    ind_all_in = np.asarray(ind_all_in)
    all_ind_all_in = np.vstack((all_ind_all_in, ind_all_in))

    ind_list = [ind_present_touch, ind_absent_touch, ind_present_notouch, ind_absent_notouch]
    all_list.append(ind_list)

print(all_ind_all_in)
print('\n')
print(all_list)

# %%
correc_present_touch_list = np.asarray(correc_present_touch_list)
correc_absent_touch_list = np.asarray(correc_absent_touch_list)
correc_present_notouch_list = np.asarray(correc_present_notouch_list)
correc_absent_notouch_list = np.asarray(correc_absent_notouch_list)

mean_correc_present_touch = np.mean(correc_present_touch_list) 
mean_correc_absent_touch = np.mean(correc_absent_touch_list) 
mean_correc_present_notouch = np.mean(correc_present_notouch_list) 
mean_correc_absent_notouch = np.mean(correc_absent_notouch_list) 

stdev_5 = statistics.stdev(correc_present_touch_list)
print('\n')
print('Standard deviation for performance in C + T:')
print(stdev_5)

stdev_6 = statistics.stdev(correc_present_notouch_list)
print('\n')
print('Standard deviation for performance in C:')
print(stdev_6)

stdev_7 = statistics.stdev(correc_absent_touch_list)
print('\n')
print('Standard deviation for performance in T:')
print(stdev_7)

stdev_8 = statistics.stdev(correc_absent_notouch_list)
print('\n')
print('Standard deviation for performance in A:')
print(stdev_8)

print(correc_present_touch_list)
print('\n')
print(correc_absent_touch_list)
print('\n')
print(correc_present_notouch_list)
print('\n')
print(correc_absent_notouch_list)
print('\n')

print(mean_correc_present_touch)
print('\n')
print(mean_correc_absent_touch)
print('\n')
print(mean_correc_present_notouch)
print('\n')
print(mean_correc_absent_notouch)
print('\n')

# %%
all_mean_correct_responses = [mean_correc_present_touch, mean_correc_present_notouch, mean_correc_absent_touch, mean_correc_absent_notouch]
all_mean_correct_responses = np.asarray(all_mean_correct_responses)

print(all_mean_correct_responses)

# %%
se_5 = stdev_5/(np.sqrt(len(correc_present_touch_list)))
print('\n')
print('Standard error for percentage correct in C + T:')
print(se_5)

se_6 = stdev_6/(np.sqrt(len(correc_present_notouch_list)))
print('\n')
print('Standard error for percentage correct in C:')
print(se_6)

se_7 = stdev_7/(np.sqrt(len(correc_absent_touch_list)))
print('\n')
print('Standard error for percentage correct in T:')
print(se_7)

se_8 = stdev_8/(np.sqrt(len(correc_absent_notouch_list)))
print('\n')
print('Standard error for percentage correct in A:')
print(se_8)

# %% Plotting mean correct responses per condition on a bar chart
fig, ax = plt.subplots()

ax.bar(np.arange(1, len(all_mean_correct_responses) + 0.1, 1), all_mean_correct_responses, color = ['steelblue', 'skyblue', 'steelblue', 'skyblue'], yerr = [se_5, se_6, se_7, se_8], capsize = 4)

conditions_1= ['C+ T+', 'C+ T-', 'C- T+', 'C- T-']
ypos1 = np.arange(1, len(conditions_1) + 0.1)

ax.set_xticks(ypos1)
ax.set_xticklabels(conditions_1)

ax.set_xlabel('Conditions', fontsize = 14)
ax.set_ylabel('Mean Percentage of correct responses', fontsize = 12)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

ax.set_ylim([0, 1])
# %%
# Use the function you just wrote
# present_yest, present_not, absent_yest, absent_not = tableTosdtDoble(table_data, 1)
# present_yesnt, present_nont, absent_yesnt, absent_nont = tableTosdtDoble(table_data, 0)

# # %% Check tables
# # print(present_yest, present_not, absent_yest, absent_not)
# # print(present_yesnt, present_nont, absent_yesnt, absent_nont)

# # %% Count yes and no responses for each condition  
# present_touch = [len(present_yest.loc[:, 'Responses']), len(present_not.loc[:, 'Responses'])]
# absent_touch = [len(absent_yest.loc[:, 'Responses']), len(absent_not.loc[:, 'Responses'])]

# present_notouch = [len(present_yesnt.loc[:, 'Responses']), len(present_nont.loc[:, 'Responses'])]
# absent_notouch = [len(absent_yesnt.loc[:, 'Responses']), len(absent_nont.loc[:, 'Responses'])]

# # %% Check counts
# # print(present_touch, absent_touch, present_notouch, absent_notouch)

# # %% Calculate % correct for each condition
# correc_present_touch = present_touch[0]/sum(present_touch)
# correc_absent_touch = absent_touch[1]/sum(absent_touch)

# correc_present_notouch = present_notouch[0]/sum(present_notouch)
# correc_absent_notouch = absent_notouch[1]/sum(absent_notouch)

# all_in = [correc_present_touch, correc_absent_touch, correc_present_notouch, correc_absent_notouch]

# all_in = np.asarray(all_in) # convert list to numpy array

# Check % correct per condition
# print(all_in)

# %% Plot
# fig, ax = plt.subplots()

# ax.bar(np.arange(1, len(all_in) + 0.1, 1), all_in, color = ['black', 'grey', 'blue', 'red'])

# conditions_1= ['C + T', 'T', 'C', 'A']
# ypos1 = np.arange(1, len(conditions_1) + 0.1)

# ax.set_xticks(ypos1)
# ax.set_xticklabels(conditions_1)

# ax.set_title('Percentage of correct responses per condition')
# ax.set_xlabel('Conditions')
# ax.set_ylabel('Percentage')

# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

# ax.set_ylim([0, 1])

# Challenge: can you make this plot look any better?
# Hint: remove top and right axes, change the color of the bars, 
# add axes titles and change the name of the x axes ticks (1, 2, 3, 4) to our conditions


# %% Calculate sdt
def SDTloglinear(hits, misses, fas, crs):
    """ 
        Compute d', response criterion, hit and false alarm rate, beta and Ad'
        by adding 0.5 to both the number ofhits and the number offalse alarms and adding 1 to both the number of signal trials and the number ofnoise trials.
        Adapted from Stanislaw and Todorov, 1999.
    """
    # Calculate hit_rate and avoid d' infinity
    hits += 0.5
    hit_rate = hits / (hits + misses + 1)

    # Calculate false alarm rate and avoid d' infinity
    fas += 0.5
    fa_rate = fas / (fas + crs + 1)

    # print(hit_rate)
    # print(fa_rate)

    # Return d', beta, c and Ad'
    out = {}
    out['d'] = Z(hit_rate) - Z(fa_rate) # Hint: normalise the centre of each curvey and subtract them (find the distance between the normalised centre
    out['beta'] = math.exp((Z(fa_rate)**2 - Z(hit_rate)**2) / 2)
    out['c'] = (Z(hit_rate) + Z(fa_rate)) / 2 # Hint: like d prime but you add the centres instead, find the negative value and half it
    out['Ad'] = norm.cdf(out['d'] / math.sqrt(2))
    out['hit_rate'] = hit_rate
    out['fa_rate'] = fa_rate

    return(out)

# %% Use the function
# sdt_touch = SDT(present_touch[0], present_touch[1], absent_touch[0], absent_touch[1])
# sdt_notouch = SDT(present_notouch[0], present_notouch[1], absent_notouch[0], absent_notouch[1])

# printme('Touch')
# print(sdt_touch)

# printme('No touch')
# print(sdt_notouch)

# Sensitivities = [sdt_touch['d'], sdt_notouch['d']]
# Bias = [sdt_touch['c'], sdt_notouch['c']]

# Sensitivities = np.asarray(Sensitivities)
# Bias = np.asarray(Bias)
# %%
print (all_list)
# %% Use function for individual calcs

first_sdt_touch = SDTloglinear(all_list[0][0][0], all_list[0][0][1], all_list[0][1][0], all_list[0][1][1])
first_sdt_notouch = SDTloglinear(all_list[0][2][0], all_list[0][2][1], all_list[0][3][0], all_list[0][3][1])

alld_sdt_touch_list = [first_sdt_touch['d']]
alld_sdt_notouch_list = [first_sdt_notouch['d']]

allbias_sdt_touch_list = [first_sdt_touch['c']]
allbias_sdt_notouch_list = [first_sdt_notouch['c']]

for x, component in enumerate(usable_data):
    ind_sdt_touch = SDTloglinear(all_list[x + 1][0][0], all_list[x + 1][0][1], all_list[x + 1][1][0], all_list[x + 1][1][1])
    ind_sdt_notouch = SDTloglinear(all_list[x + 1][2][0], all_list[x + 1][2][1], all_list[x + 1][3][0], all_list[x + 1][3][1])
    
    alld_sdt_touch_list.append(ind_sdt_touch['d'])
    alld_sdt_notouch_list.append(ind_sdt_notouch['d'])

    allbias_sdt_touch_list.append(ind_sdt_touch['c'])
    allbias_sdt_notouch_list.append(ind_sdt_notouch['c'])

print(alld_sdt_touch_list)
print('\n')
print(alld_sdt_notouch_list)
print('\n')
print(allbias_sdt_touch_list)
print('\n')
print(allbias_sdt_notouch_list)

# %%
meand_all_sdt_touch = (sum(alld_sdt_touch_list)/len(alld_sdt_touch_list))
meand_all_sdt_notouch = (sum(alld_sdt_notouch_list)/len(alld_sdt_notouch_list))

meanbias_all_sdt_touch = (sum(allbias_sdt_touch_list)/len(allbias_sdt_touch_list))
meanbias_all_sdt_notouch = (sum(allbias_sdt_notouch_list)/len(allbias_sdt_notouch_list))

print('\n')
print("Mean d' with touch")
print(meand_all_sdt_touch)
print('\n')
print("Mean d' without touch")
print(meand_all_sdt_notouch)

print('\n')
print("Mean bias with touch")
print(meanbias_all_sdt_touch)
print('\n')
print("Mean bias without touch")
print(meanbias_all_sdt_notouch)

stdev_1 = statistics.stdev(alld_sdt_touch_list)
print('\n')
print('Standard deviation for touch d:')
print(stdev_1)

stdev_2 = statistics.stdev(alld_sdt_notouch_list)
print('\n')
print('Standard deviation for no touch d:')
print(stdev_2)

stdev_3 = statistics.stdev(allbias_sdt_touch_list)
print('\n')
print('Standard deviation for touch bias:')
print(stdev_3)

stdev_4 = statistics.stdev(allbias_sdt_notouch_list)
print('\n')
print('Standard deviation for no touch bias:')
print(stdev_4)

se_1 = stdev_1/(np.sqrt(len(alld_sdt_touch_list)))
print('\n')
print('Standard error for touch d:')
print(se_1)

se_2 = stdev_2/(np.sqrt(len(alld_sdt_notouch_list)))
print('\n')
print('Standard error for no touch d:')
print(se_2)

se_3 = stdev_3/(np.sqrt(len(allbias_sdt_touch_list)))
print('\n')
print('Standard error for touch bias:')
print(se_3)

se_4 = stdev_4/(np.sqrt(len(allbias_sdt_notouch_list)))
print('\n')
print('Standard error for no touch bias:')
print(se_4)

mean_sensitivities = [meand_all_sdt_touch, meand_all_sdt_notouch]
mean_sensitivities = np.asarray(mean_sensitivities)

mean_bias = [meanbias_all_sdt_touch, meanbias_all_sdt_notouch]
mean_bias = np.asarray(mean_bias)

print('\n')
print(mean_sensitivities)
print('\n')
print(mean_bias)

# %% Practice loop
keys = range(6)

sdt_d_pairs = {}
for i in keys:
    sdt_d_pairs['sdt_d_pair_{0}'.format(i + 1)] = [alld_sdt_touch_list[i], alld_sdt_notouch_list[i]]

print(sdt_d_pairs)
print('\n')

sdt_bias_pairs = {}
for y in keys:
    sdt_bias_pairs['sdt_bias_pair_{0}'.format(y + 1)] = [allbias_sdt_touch_list[y], allbias_sdt_notouch_list[y]]

print(sdt_bias_pairs)

fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize = (8, 8))

for x in keys:
    ax1.plot(np.arange(1, len(mean_sensitivities) + 0.1, 1), sdt_d_pairs['sdt_d_pair_{0}'.format(x + 1)], color = 'steelblue', marker = '.', markersize = 10, markerfacecolor = 'steelblue', markeredgecolor = 'steelblue')

ax1.plot(np.arange(1, len(mean_sensitivities) + 0.1, 1), mean_sensitivities, color = 'red', marker = '.', markersize = 10, markeredgecolor = 'red', markerfacecolor = 'red')

conditions_3 = ['Cold + Touch', 'Cold']
ypos3 = np.arange(1, len(conditions_3) + 0.1)

ax1.set_xticks(ypos3)
ax1.set_xticklabels(conditions_3, fontsize = 12)

for z in keys:
    ax2.plot(np.arange(1, len(mean_sensitivities) + 0.1, 1), sdt_bias_pairs['sdt_bias_pair_{0}'.format(z + 1)], color = 'steelblue', marker = '.', markersize = 10, markerfacecolor = 'steelblue', markeredgecolor = 'steelblue')

ax2.plot(np.arange(1, len(mean_bias) + 0.1, 1), mean_bias, color = 'red', marker = '.', markersize = 10, markeredgecolor = 'red', markerfacecolor = 'red')

ax2.set_xticks(ypos3)
ax2.set_xticklabels(conditions_3, fontsize = 12)

plt.tight_layout()
ax1.set_ylim([0, 3.5])
ax1.set_xlim([0.6, 2.4])

ax2.set_ylim([-1.2, 1])
ax2.set_xlim([0.6, 2.4])

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax1.spines['left'].set_linewidth(1.5)
ax1.spines['bottom'].set_linewidth(1.5)

ax2.spines['left'].set_linewidth(1.5)
ax2.spines['bottom'].set_linewidth(1.5)

ax1.set_ylabel('Sensitivity', fontsize = 12)
ax2.set_ylabel('Bias', fontsize = 12)

ax1.set_xlabel('Stimulus type', fontsize = 12)
ax2.set_xlabel('Stimulus type', fontsize = 12)

plt.tight_layout()

# %%
values = np.arange(1, 7, 1)

block_pairs = {}
for i in values:
    block_pairs['block_pair_{0}'.format(i)] = [correc_present_touch_list[i - 1], correc_present_notouch_list[i - 1]]

print(block_pairs)
print('\n')

block_pairs_2 = {}
for x in values:
    block_pairs_2['block_pair_2_{0}'.format(x)] = [correc_absent_touch_list[x - 1], correc_absent_notouch_list[x - 1]]

print(block_pairs_2)

mean_block_pair = [mean_correc_present_touch, mean_correc_present_notouch]
mean_block_pair_2 = [mean_correc_absent_touch, mean_correc_absent_notouch]

fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize = (8, 8))

for y in values:
    ax1.plot(np.arange(1, len(mean_sensitivities) + 0.1, 1), block_pairs['block_pair_{0}'.format(y)], color = 'steelblue', marker = '.', markersize = 10, markerfacecolor = 'steelblue', markeredgecolor = 'steelblue')

ax1.plot(np.arange(1, len(mean_sensitivities) + 0.1, 1), mean_block_pair, color = 'red', marker = '.', markersize = 10, markeredgecolor = 'red', markerfacecolor = 'red')

conditions_4 = ['C+ T+', 'C+ T-']
ypos4 = np.arange(1, len(conditions_3) + 0.1)

ax1.set_xticks(ypos4)
ax1.set_xticklabels(conditions_4, fontsize = 12)

for z in values:
    ax2.plot(np.arange(1, len(mean_sensitivities) + 0.1, 1), block_pairs_2['block_pair_2_{0}'.format(z)], color = 'steelblue', marker = '.', markersize = 10, markerfacecolor = 'steelblue', markeredgecolor = 'steelblue')

ax2.plot(np.arange(1, len(mean_bias) + 0.1, 1), mean_block_pair_2, color = 'red', marker = '.', markersize = 10, markeredgecolor = 'red', markerfacecolor = 'red')

conditions_5 = ['C- T+', 'C- T-']
ypos5 = np.arange(1, len(conditions_5) + 0.1)

ax2.set_xticks(ypos5)
ax2.set_xticklabels(conditions_5, fontsize = 12)

plt.tight_layout()
ax1.set_ylim([0.4, 1.05])
ax1.set_xlim([0.6, 2.4])

ax2.set_ylim([0.2, 1.05])
ax2.set_xlim([0.6, 2.4])

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax1.spines['left'].set_linewidth(1.5)
ax1.spines['bottom'].set_linewidth(1.5)

ax2.spines['left'].set_linewidth(1.5)
ax2.spines['bottom'].set_linewidth(1.5)

ax1.set_ylabel('Percentage of correct responses', fontsize = 12)
ax2.set_ylabel('Percentage of correct responses', fontsize = 12)

ax1.set_xlabel('Stimulus present conditions', fontsize = 12)
ax2.set_xlabel('Stimulus absent conditions', fontsize = 12)

plt.tight_layout()


# %% Plotting mean sensitivity + bias bar chart
fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize = (5, 5))

ax1.bar(np.arange(1, len(mean_sensitivities) + 0.1, 1), mean_sensitivities, color = ['steelblue', 'skyblue'], yerr = [se_1, se_2], capsize = 4)
ax2.bar(np.arange(1, len(mean_bias) + 0.1, 1), mean_bias, color = ['steelblue', 'skyblue'], yerr = [se_3, se_4], capsize = 4)

conditions_2 = ['Cold + Touch', 'Cold']
ypos2 = np.arange(1, len(conditions_2) + 0.1)

ax1.set_ylabel('Mean Sensitivity', fontsize = 12)
ax2.set_ylabel('Mean Bias', fontsize = 12)

ax1.set_xlabel('Stimulus type', fontsize = 12)
ax2.set_xlabel('Stimulus type', fontsize = 12)

ax1.set_xticks(ypos2)
ax1.set_xticklabels(conditions_2, fontsize = 12)

ax2.axhline(0, 0, 2, color = 'k', linewidth = 0.75)

ax1.set_ylim([0, 3])
ax2.set_ylim([-0.6, 0.2])

ax2.set_xticks(ypos2)
ax2.set_xticklabels(conditions_2, fontsize = 12)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax1.spines['left'].set_linewidth(1.5)
ax1.spines['bottom'].set_linewidth(1.5)

ax2.spines['left'].set_linewidth(1.5)
ax2.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()

# %%
alld_sdt_touch = np.asarray(alld_sdt_touch_list)
alld_sdt_notouch = np.asarray(alld_sdt_notouch_list)
allbias_sdt_touch = np.asarray(allbias_sdt_touch_list)
allbias_sdt_notouch = np.asarray(allbias_sdt_notouch_list)

stat_d, p_d = ttest_rel(alld_sdt_touch, alld_sdt_notouch, alternative = 'less') 
print('p-value for sensitivity:')
print(p_d)
print(stat_d)

print('\n')

stat_bias, p_bias = ttest_rel(allbias_sdt_touch, allbias_sdt_notouch)
print('p-value for bias:')
print(p_bias)
print(stat_bias)

print('\n')

stat_perf, p_perf = ttest_rel(correc_present_touch_list, correc_present_notouch_list, alternative = 'less')
print('p-value for performance between C + T and C:')
print(p_perf)
print(stat_perf)

print('\n')

stat_perf_2, p_perf_2 = ttest_rel(correc_absent_touch_list, correc_absent_notouch_list, alternative = 'less')
print('p-value for performance between T and A:')
print(p_perf_2)
print(stat_perf_2)

# %%
