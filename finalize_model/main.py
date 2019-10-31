import os

your_path = "/home/lingyi/Desktop/repo/Fall2019-proj3-sec1--proj3-sec1-grp3/finalize_model"
os.chdir(your_path)

# step0_faceCropper
cmd = your_path + '/step0_faceCropper.py'
os.system('{} {}'.format('python', cmd))

# step1_faceAlignment
cmd = your_path + '/step1_faceAlignment.py'
os.system('{} {}'.format('python', cmd))

# step2_subfolder
cmd = your_path + '/step2_subFolders.py'
os.system('{} {}'.format('python', cmd))

# step3_H5preprocess
cmd = your_path + '/step3_H5preprocess.py'
os.system('{} {}'.format('python', cmd))

# step4_mainpro
cmd = your_path + '/step4_mainpro.py'
os.system('{} {}'.format('python', cmd))

# step5_k_fold_train
cmd = your_path + '/step5_k_fold_train.py'
os.system('{} {}'.format('python', cmd))

# step6

# step7_prePrivateTest.py
cmd = your_path + '/step7_predPrivateTest.py'
os.system('{} {}'.format('python', cmd))



# /home/lingyi/Desktop/repo/Fall2019-proj3-sec1--proj3-sec1-grp3/finalize_model/trained_models/finalize_model_VGG19/1