import os
path = os.getcwd() 
file_path = path + '/step4_mainpro.py'

cmd = file_path + ' --model Resnet34 --bs 128 --lr 0.01 --fold %d' %(1)
os.system('{} {}'.format('python', cmd))
print("Your Model is trained.")


