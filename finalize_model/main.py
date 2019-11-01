import os

your_path = "/Users/carol/Desktop/Fall2019-proj3-sec1-grp3/finalize_model"
os.chdir(your_path)


'''
Training models 
'''
# you can skip step0-step3 and use data.h5 file in /data/h5 folder directly

# step0_faceCropper
cmd = your_path + '/step0_faceCropper.py'
os.system('{} {}'.format('python', cmd))

# step1_faceAlignment
cmd = your_path + '/step1_faceAlignment.py'
os.system('{} {}'.format('python', cmd))

# step2_subfolder
cmd = your_path + '/step2_subFolders.py'
os.system('{} {}'.format('python', cmd))

# step3_preprocess
cmd = your_path + '/step3_preprocess.py'
os.system('{} {}'.format('python', cmd))

# step4_mainpro
cmd = your_path + '/step4_mainpro.py'
os.system('{} {}'.format('python', cmd))

# step5_k_fold_train
cmd = your_path + '/step5_k_fold_train.py'
os.system('{} {}'.format('python', cmd))





'''
Testing results 
'''

# Please repeat Step0 and Step1 on your testing data 
# make the data_1 as your input (48x48x3)




# #### TESTING 

# # step7_predPrivateTest.py
# cmd = your_path + '/step7_predPrivateTest.py'
# os.system('{} {}'.format('python', cmd))



