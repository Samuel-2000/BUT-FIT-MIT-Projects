import torch
import numpy as np
from tqdm import tqdm
from torch import optim as optim
from lib import data_loader, nets, paths


def get_network(args, device):
    train_loader, test_loader, val_loader, first_test_case = data_loader.data_load(args)
    model = retrieve_trained_net(device, train_loader, val_loader, test_loader, args)

    return model, first_test_case


def retrieve_trained_net(device, train_loader, val_loader, test_loader, args):
    # Create instance of neural network
    model_classes = [nets.DepthModel]
    try:
        model = model_classes[args.NNmodel]().to(device)
    except IndexError:
        print("Error: model not specified")
        exit(0)

    
    if paths.final_state.exists():
        print("using saved final state for training")
        model.load_state_dict(torch.load(paths.final_state, map_location=torch.device(device)))
        torch.save(model.state_dict(), paths.model_init_state) # return point if training goes bad.
    elif paths.model_init_state.exists():
        model.load_state_dict(torch.load(paths.model_init_state, map_location=torch.device(device)))


    if args.train:
        optimizer_class = optim.Adam if args.adam else optim.SGD
        optimizer = optimizer_class(model.parameters(), lr=0.01)  # set optimizer

        train_loss_file = open(paths.train_loss_path, "a+")
        val_loss_file = open(paths.val_loss_path, "a+")
#
        min_vall_loss = float('inf')
        for epoch in tqdm(range(0, args.epochs), desc="Model training"):
            train_loss = nets.train(model, train_loader, optimizer, device, train_loss_file)
            val_loss = nets.validate(model, val_loader, device)

            torch.save(model.state_dict(), paths.final_state)  # save parameters of model

            #train_loss_file.write(f"{str(float(train_loss.data.cpu().numpy()))}\n")
            train_loss_file.write(f"{str(train_loss)}\n")
            val_loss_file.write(f"{str(val_loss)}\n")

            if epoch % 20 == 0:
                if val_loss < min_vall_loss:
                    min_vall_loss = val_loss
                else:
                    print(f"premature break because validation loss got higher (overtraining) after epoch: {epoch}")
                    break

        train_loss_file.close()
        val_loss_file.close()

    return model  # return neural network in final (trained) state

