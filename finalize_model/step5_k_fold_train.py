import os

for i in range(10):
    cmd = '/home/lingyi/Desktop/finalize_model/step4_mainpro.py --model Resnet34 --bs 128 --lr 0.01 --fold %d' %(i+1)
    os.system('{} {}'.format('python', cmd))
print("Train VGG19 ok!")

