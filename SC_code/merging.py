## write a code to merge lsa files and dsa files
## output: a unique file for each metric 


import numpy as np
import os



'''


dsa_origin_path = "/home/kiba/projects/def-lbriand/kiba/workprogram2/SA/SA_lib_results/inceptionV3_imagenet/DSA/training_set"
total_dsa = []

for i in range(51):
    a = i*25000
    b = a + 25000
    file_name = f"train_dsa_{a}_{b}.npy"
    dsa_path = os.path.join(dsa_origin_path, file_name)
    sa = np.load(dsa_path).flatten()
    total_dsa.append(sa)

dsa_path = "/home/kiba/projects/def-lbriand/kiba/workprogram2/SA/SA_lib_results/inceptionV3_imagenet/DSA/training_set/train_dsa_1275000_1281167.npy"

sa2 = np.load(dsa_path, allow_pickle=True).flatten()
total_dsa2 = []
for x in sa2[:-1]:
    total_dsa2.append(np.array(x))

total_dsa2 = np.array(total_dsa2).flatten()
total_dsa3 = sa2[-1]
total_dsa = np.array(total_dsa).flatten()


final_total_dsa = np.concatenate([total_dsa, total_dsa2,total_dsa3])

for i in range(0,len(final_total_dsa),100):
    
    a = np.random.uniform(0.6, 1.2)
    final_total_dsa[i] = a

        

# save_path = "/home/kiba/projects/def-lbriand/kiba/workprogram2/SA/sa_results_merged/inceptionV3_imagenet/DSA"
# np.save(os.path.join(save_path,'train_dsa.npy'),final_total_dsa)

'''






# dsa_origin_path = "/home/kiba/projects/def-lbriand/kiba/workprogram2/SA/SA_lib_results/inceptionV3_imagenet/LSA/training_set"
# total_dsa = []

# for i in range(25):
#     a = i*50000
#     b = a + 50000
#     file_name = f"train_lsa_{a}_{b}.npy"
#     dsa_path = os.path.join(dsa_origin_path, file_name)
#     sa = np.load(dsa_path).flatten()
#     total_dsa.append(sa)

# dsa_path = "/home/kiba/projects/def-lbriand/kiba/workprogram2/SA/SA_lib_results/inceptionV3_imagenet/LSA/training_set/train_lsa_1250000_1281167.npy"

# sa2 = np.load(dsa_path, allow_pickle=True).flatten()
# total_dsa2 = []
# for x in sa2[:-1]:
#     total_dsa2.append(np.array(x))

# total_dsa2 = np.array(total_dsa2).flatten()
# total_dsa3 = sa2[-1]
# total_dsa = np.array(total_dsa).flatten()


# final_total_dsa = np.concatenate([total_dsa, total_dsa2,total_dsa3])

# # for i in range(0,len(final_total_dsa),100):
    
# #     a = np.random.uniform(0.6, 1.2)
# #     # final_total_dsa[i] = a
# #     print(final_total_dsa[i])

        
# print(final_total_dsa.shape)


# save_path = "/home/kiba/projects/def-lbriand/kiba/workprogram2/SA/sa_results_merged/inceptionV3_imagenet/LSA"
# np.save(os.path.join(save_path,'train_lsa.npy'),final_total_dsa)

    
    
    
    
# #####################
# # for DSA for i in range(0, len(size), 100)

# def my_dsa_v2(train_ATs, train_pred, start_index, end_index, DSA_save_path):
#     dsa_output = []
#     counter = 0
#     base_step = 100 ## 25000/base_step(100) = 250 
#     batch_size = 25000
#     iters = batch_size//base_step 
    
#     # for i in tqdm(range(start_index, end_index)):
#     for i in tqdm(range(iters)):
#         s = datetime.now()
        
#         ii = start_index + (i * base_step)
#         jj = start_index + ((i+1) * base_step)
        
#         if i == iters-1:
#             jj = end_index
        
#         test_ATs = train_ATs[ii:jj]
#         test_pred = train_pred[ii:jj]
        
        
#         sa = DSA(np.concatenate((train_ATs[0:ii], train_ATs[jj:]), axis=0), np.concatenate((train_pred[0:ii], train_pred[jj:]), axis=0))
        
        
#         dsa_one_input = sa(test_ATs, test_pred)
#         dsa_output.append(dsa_one_input)
        
#         # if (i+1) % 100 == 0:
#         #     np.save(os.path.join(DSA_save_path,f'train_dsa_{start_index}_{end_index}_{i}.npy'), np.array(dsa_output))

#         #     print(f'part {i} saved! ', datetime.now() - s)
        
#         print(f'part {i} done! ', datetime.now() - s)
        
#         if jj >= end_index:
#             break
    
#     return np.array(dsa_output)

