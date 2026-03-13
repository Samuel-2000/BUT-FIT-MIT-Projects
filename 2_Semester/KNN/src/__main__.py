import sys
import os

# setting of workspace. change based on your filesystem location.
# might not be necessary if using IDE instead of command line.
if os.name == 'posix':
    # Google Colab (Linux)
    sys.path.insert(0, '/content/KNnice/src')
else:
    # Windows
    sys.path.insert(0, 'C:\\Users\\Samuel\\Desktop\\KNnice\\src')

import sys
import torch
import prep
from lib import paths
from lib.arg_parse import parse_arguments
from lib import plots

if __name__ == '__main__':
    args = parse_arguments()
    paths.init_dirs()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using: {device}")

    print(f"\tin train")
    model, first_test_case = prep.get_network(args, device)
    plots.plot_depth_activations(model, first_test_case, device, paths.imgs)
    plots.plot_losses(paths.train_loss_path, paths.val_loss_path)
    

sys.exit(0)
