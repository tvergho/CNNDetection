from util import mkdir


# directory to store the results
results_dir = './results/'
mkdir(results_dir)

# root to the testsets
dataroot = './dataset/diretestadm'
# dataroot = './dataset/direadmsd'

# list of synthesis algorithms
vals = ['progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'crn', 'imle', 'seeingdark', 'san', 'deepfake', 'stylegan2', 'whichfaceisreal']
# vals = ['sd']
# indicates if corresponding testset has multiple classes
multiclass = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]

# model
# model_path = 'weights/model_epoch_3_prelrincrease.pth'
# model_path = 'weights/model_epoch_best.pth'
model_path = "checkpoints/dirediffusiondb/model_epoch_latest.pth"