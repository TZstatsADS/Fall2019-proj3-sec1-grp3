import os

for i in range(10):
    cmd = '/home/lingyi/Desktop/Project3/step4_mainpro.py --model VGG19 --bs 32 --lr 0.01 --fold %d' %(i+1)
    os.system('{} {}'.format('python', cmd))
print("Train VGG19 ok!")

