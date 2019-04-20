from torchsummary import summary 

from models import custom_fcn_model
from utils import utils
from options import options

opt = options.set_options(training=True)

a = custom_fcn_model.CustomFCN(opt).cuda()
exec(utils.TEST_EMBEDDING)
