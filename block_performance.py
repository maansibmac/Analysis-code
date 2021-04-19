# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
Z = norm.ppf
from classes_text import *
import os


# %%
usable_data = ['test_15122020_3', 'test_19032021_2', 'test_30032021_2', 'test_31032021_1', 'test_31032021_2']

table_data = pd.read_csv(f"../../data/test_15122020_2/data/data_all.csv")
table_data.loc[:,'Subject'] = 1
table_data.to_csv("../../data/test_15122020_2/data/data_all.csv", index = False)

for i, element in enumerate(usable_data):
    next_data = pd.read_csv(f"../../data/" + element + "/data/data_all.csv")
    next_data.loc[:,'Subject'] = i + 2
    next_data.to_csv(f"../../data/" + element + "/data/data_all.csv", index = False)
    table_data = table_data.append(next_data, sort = False)

print(table_data)

def block_performance(table, start_trial, end_trial):
    all_percent_corrects = []
    keys = np.arange(1, 7, 1)
    for i in keys:
        ind_table = table.loc[table['Subject'] == i]
        chosen_block = ind_table.loc[ind_table['Trial'].between(start_trial, end_trial)]

        table_touch = chosen_block.loc[chosen_block['Touch'] == 1]

        table_cold_t = table_touch.loc[table_touch['Cold'] == 1] 
        table_nocold_t = table_touch.loc[table_touch['Cold'] == 0] 

        present_yes_t = table_cold_t.loc[table_cold_t['Responses'] == 1]
        present_no_t = table_cold_t.loc[table_cold_t['Responses'] == 0]
     
        absent_yes_t = table_nocold_t.loc[table_nocold_t['Responses'] == 1]
        absent_no_t = table_nocold_t.loc[table_nocold_t['Responses'] == 0]

        present_touch = [len(present_yes_t.loc[:, 'Responses']), len(present_no_t.loc[:, 'Responses'])]
        absent_touch = [len(absent_yes_t.loc[:, 'Responses']), len(absent_no_t.loc[:, 'Responses'])]

        table_ntouch = chosen_block.loc[chosen_block['Touch'] == 0]

        table_cold_nt = table_ntouch.loc[table_ntouch['Cold'] == 1] 
        table_nocold_nt = table_ntouch.loc[table_ntouch['Cold'] == 0] 

        present_yes_nt = table_cold_nt.loc[table_cold_nt['Responses'] == 1]
        present_no_nt = table_cold_nt.loc[table_cold_nt['Responses'] == 0]
     
        absent_yes_nt = table_nocold_nt.loc[table_nocold_nt['Responses'] == 1]
        absent_no_nt = table_nocold_nt.loc[table_nocold_nt['Responses'] == 0]

        present_notouch = [len(present_yes_nt.loc[:, 'Responses']), len(present_no_nt.loc[:, 'Responses'])]
        absent_notouch = [len(absent_yes_nt.loc[:, 'Responses']), len(absent_no_nt.loc[:, 'Responses'])]

        correct_resp = [present_touch[0], absent_touch[1], present_notouch[0], absent_notouch[1]]
        incorrect_resp = [present_touch[1], absent_touch[0], present_notouch[1], absent_notouch[0]]

        total_correct_resp = sum(correct_resp)
        total_incorrect_resp = sum(incorrect_resp)

        all_resp = [total_correct_resp, total_incorrect_resp]
        total_resp = sum(all_resp)

        percentage_correct = total_correct_resp/total_resp
        all_percent_corrects.append(percentage_correct)
    all_percent_corrects = np.asarray(all_percent_corrects)
    return all_percent_corrects

all_blockone_percentages = block_performance(table_data, 1, 18)
print(all_blockone_percentages)
stdev_1 = statistics.stdev(all_blockone_percentages)
se_1 = stdev_1/(np.sqrt(len(all_blockone_percentages)))
print('\n')
mean_blockone = np.mean(all_blockone_percentages)
print(mean_blockone)
print('\n')
   
all_blocktwo_percentages = block_performance(table_data, 19, 36)
print(all_blocktwo_percentages)
stdev_2 = statistics.stdev(all_blocktwo_percentages)
se_2 = stdev_1/(np.sqrt(len(all_blocktwo_percentages)))
print('\n')
mean_blocktwo = np.mean(all_blocktwo_percentages)
print(mean_blocktwo)
print('\n')
   
all_blockthree_percentages = block_performance(table_data, 37, 54)
print(all_blockthree_percentages)
stdev_3 = statistics.stdev(all_blockthree_percentages)
se_3 = stdev_1/(np.sqrt(len(all_blockthree_percentages)))
print('\n')
mean_blockthree = np.mean(all_blockthree_percentages)
print(mean_blockthree)
print('\n')

all_blockfour_percentages = block_performance(table_data, 55, 72)
print(all_blockfour_percentages)
stdev_4 = statistics.stdev(all_blockfour_percentages)
se_4 = stdev_1/(np.sqrt(len(all_blockfour_percentages)))
print('\n')
mean_blockfour = np.mean(all_blockfour_percentages)
print(mean_blockfour)
print('\n')

all_blockfive_percentages = block_performance(table_data, 73, 90)
print(all_blockfive_percentages)
stdev_5 = statistics.stdev(all_blockfive_percentages)
se_5 = stdev_1/(np.sqrt(len(all_blockfive_percentages)))
print('\n')
mean_blockfive = np.mean(all_blockfive_percentages)
print(mean_blockfive)
print('\n')

all_blocksix_percentages = block_performance(table_data, 91, 108)
print(all_blocksix_percentages)
stdev_6 = statistics.stdev(all_blocksix_percentages)
se_6 = stdev_1/(np.sqrt(len(all_blocksix_percentages)))
print('\n')
mean_blocksix = np.mean(all_blocksix_percentages)
print(mean_blocksix)
print('\n')

all_block_means = [mean_blockone, mean_blocktwo, mean_blockthree, mean_blockfour, mean_blockfive, mean_blocksix]
all_block_means = np.asarray(all_block_means)

# %%
colour_map = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']

keys = np.arange(1, 7, 1)
part_blocks = {}
for i in keys:
    part_blocks['participant_{0}'.format(i)] = [all_blockone_percentages[i - 1], all_blocktwo_percentages[i - 1], all_blockthree_percentages[i - 1], all_blockfour_percentages[i - 1], all_blockfive_percentages[i - 1], all_blocksix_percentages[i - 1]]

print(part_blocks)

fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize = (8, 8))

ax1.bar(np.arange(1, len(all_block_means) + 0.1, 1), all_block_means, color = colour_map[2:-1], yerr = [se_1, se_2, se_3, se_4, se_5, se_6], capsize = 4)
ax1.set_ylim(0, 1)

blocks = np.arange(1, 7, 1)

for x in keys:
    ax2.plot(blocks, part_blocks['participant_{0}'.format(x)], color = 'steelblue', marker = '.', markersize = 5, markerfacecolor = 'navy', markeredgecolor = 'navy', alpha = 0.5)

ax2.plot(blocks, all_block_means, color = 'red', marker = '.', markersize = 5, markerfacecolor = 'black', markeredgecolor = 'black')

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax1.spines['left'].set_linewidth(1.5)
ax1.spines['bottom'].set_linewidth(1.5)

ax2.spines['left'].set_linewidth(1.5)
ax2.spines['bottom'].set_linewidth(1.5)

ax1.set_xlabel('Blocks', fontsize = 14)
ax2.set_xlabel('Blocks', fontsize = 14)

ax1.set_ylabel('Percentage of correct responses', fontsize = 12)
ax2.set_ylabel('Percentage of correct responses', fontsize = 12)
# %%
