from util import mkdir


# directory to store the results
results_dir = './results/'
mkdir(results_dir)

# root to the testsets
dataroot = './dataset/test'
# dataroot = './dataset/direadmsd'

# list of synthesis algorithms
# vals = ['progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'crn', 'imle', 'seeingdark', 'san', 'deepfake', 'stylegan2', 'whichfaceisreal']
vals = ['biggan']
# indicates if corresponding testset has multiple classes
multiclass = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]

# model
# model_path = "checkpoints/dirediffusiondb/model_epoch_latest.pth"
model_path = "/Users/tylervergho/Documents/Politics and AI/models/vit_efnet_ft/model_epoch_best.pth"
# model_path = "weights/blur_jpg_prob0.5.pth"