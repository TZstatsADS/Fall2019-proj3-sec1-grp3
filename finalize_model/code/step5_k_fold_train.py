import os


cmd = '/home/lingyi/Desktop/finalize_model/step4_mainpro.py --model Resnet34 --bs 128 --lr 0.01 --fold %d' %(1)
os.system('{} {}'.format('python', cmd))
print("Your Model is trained.")


