import json
from data_loader import *
from multitask_model import *
import argparse
import torch.optim as optim


def train(model,optimizer,data,criterion,num_epochs=25):
    for epoch in range(num_epochs):
        print("{}/{} epoch".format(epoch,num_epochs))

        running_loss=1
        ethnic_correct=0
        prostate_correct=0

        for inputs,ethnics,pc in data:
            inputs=inputs.to(device)
            outputs=model(inputs)

            loss0 = criterion[0](outputs[0], torch.max(ethnics.float(),1)[1])
            loss1 = criterion[1](outputs[1], torch.max(pc.float(), 1)[1])
            loss = loss0+loss1

            loss.backward() #calculate derivative
            optimizer.step()

            print(loss)

            running_loss += loss*inputs.size(0)
            ethnic_correct += torch.sum(torch.max(outputs[0], 1)[1] == torch.max(ethnics, 1)[1])
            prostate_correct += torch.sum(torch.max(outputs[1], 1)[1] == torch.max(pc, 1)[1])







def main(args):
    """Returns torch.utils.data.DataLoader for geno dataset."""

    with open(args.directory+"/"+args.labels_file, 'r') as f:
        pheno_dict = json.load(f)

    data_loader=get_loader(geno_file=args.directory+"/"+args.train_file, ids=[x for x in pheno_dict.keys()], labels=pheno_dict, batch_size=4, shuffle=True,num_workers=6)

    model=multi_model(input_size=1,num_classes=3)

    lrlast = .001

    criterion = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]

    optimi = optim.Adam([{"params": model.parameters()}], lr=.0001)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train(model,optimi,data_loader,criterion,25)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='train.csv', help='path to file with genotypes')
    parser.add_argument('--labels_file', type=str, default='pheno_dictionary.json', help='path to phenotype dictionary')
    parser.add_argument('--directory', type=str, default='/Users/meghana/Desktop', help='directory with files')

    args = parser.parse_args()

    main(args)