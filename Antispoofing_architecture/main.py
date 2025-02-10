import argparse
import sys
import os
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
import torch
from torch import nn
from model import RawNet
from tensorboardX import SummaryWriter
from tqdm import tqdm
import data_utils_LA

def keras_lr_decay(step, decay=0.0001):
    return 1./(1.+decay*step)

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0001)
    elif isinstance(m, nn.BatchNorm1d):
        pass
    else:
        if hasattr(m, 'weight'):
            torch.nn.init.kaiming_normal_(m.weight, a=0.01)

def evaluate_accuracy(data_loader, model, device):
    """
    Evaluate classification accuracy on a 2-class problem: 0=spoof, 1=bonafide
    Now we pass 'llm_label' to the model's forward too.
    """
    num_correct = 0
    num_total = 0
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_meta, batch_llm in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_llm = batch_llm.to(device)

            out = model(batch_x, y=batch_y, is_test=False, llm_label=batch_llm)
            _, preds = out.max(dim=1)
            num_total += batch_x.size(0)
            num_correct += (preds == batch_y).sum().item()

    return 100.0 * num_correct / num_total

def produce_evaluation_file(dataset, model, device, save_path):
    """
    Evaluate one file at a time (batch_size=1) to avoid collate issues.
    Writes either:
       "f_name sys_id bonafide/spoof score"
    or
       "f_name score"
    If dataset.is_eval => LA eval => print all 4 fields.

    We now pass 'llm_label' to the model as well.
    """
    print("Evaluating")

    # Force batch_size=1
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()

    final_filename = []
    final_ground_truths = []
    final_scores = []

    with open(save_path, 'w') as fh, torch.no_grad():
        for (batch_x, batch_y, batch_meta, batch_llm) in tqdm(data_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_llm = batch_llm.to(device)

            # forward pass => softmax output
            out = model(batch_x, y=batch_y, is_test=True, llm_label=batch_llm)
            spoof_score = out[:, 1].item()

            meta = batch_meta  # single item (ASVFile)
            file_name = meta.file_name
            s_id = dataset.sysid_dict_inv.get(meta.sys_id, '-')
            is_bf = 'bonafide' if meta.key == 1 else 'spoof'

            final_filename.append(file_name[0])
            final_ground_truths.append(is_bf)
            final_scores.append(spoof_score)


            # print(file_name[0] , is_bf , spoof_score)

            if dataset.is_eval:
                fh.write(f"{file_name} {s_id} {is_bf} {spoof_score}\n")
            else:
                fh.write(f"{file_name} {spoof_score}\n")
    
    import pandas as pd 
    df = pd.DataFrame({"filename": final_filename, "ground_truth": final_ground_truths, "score": final_scores})
    df.to_csv("SCORE-CARD-CREATION.csv", index=False)
    print("CSV saved successfully!")

def train_epoch(data_loader, model, lr, optim, device):
    """
    One epoch of training, with cross-entropy (spoof=0, bonafide=1),
    Weighted 1:9 if you want to push the model away from FNs on spoof.
    We pass llm_label into model's forward.
    """
    running_loss = 0.0
    correct = 0
    total = 0
    model.train()

    weight = torch.FloatTensor([1.0, 9.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for i, (batch_x, batch_y, batch_meta, batch_llm) in enumerate(data_loader, 1):
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_llm = batch_llm.to(device)

        out = model(batch_x, y=batch_y, is_test=False, llm_label=batch_llm)
        loss = criterion(out, batch_y)

        _, preds = out.max(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)
        running_loss += loss.item() * batch_x.size(0)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if i % 10 == 0:
            acc = 100.0 * correct / total
            sys.stdout.write(f"\rTrain Batch {i} - Acc: {acc:.2f}%")

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser("ASVSpoof2019 model")
    parser.add_argument('--eval', action='store_true', default=False,
                        help='If set => produce score file and exit')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to a .pth model checkpoint')
    parser.add_argument('--database_path', type=str, default='/media/data_dump/akash21514/LA/',
                        help='ASVspoof LA database location')
    parser.add_argument('--protocols_path', type=str, default='/media/data_dump/akash21514/LA/ASVspoof2019_LA_cm_protocols/',
                        help='LA protocol directory path')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Where to save eval CM scores')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--comment', type=str, default=None)
    parser.add_argument('--track', type=str, default='logical')
    parser.add_argument('--features', type=str, default='Raw_audio')
    parser.add_argument('--is_eval', action='store_true', default=False,
                        help='If True => use the eval partition instead of dev/train')
    parser.add_argument('--eval_part', type=int, default=0)
    parser.add_argument('--loss', type=str, default='weighted_CCE')

    parser.add_argument('--llm_csv_path', type=str, default=None,
                        help='CSV with file_name,label used at eval for nudge')

    dir_yaml = os.path.splitext('model_config_RawNet2')[0] + '.yaml'
    with open(dir_yaml, 'r') as f_yaml:
        parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)

    np.random.seed(parser1['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not os.path.exists('models'):
        os.mkdir('models')

    args = parser.parse_args()

    model_tag = f"model_{args.track}_{args.loss}_{args.num_epochs}_{args.batch_size}_{args.lr}"
    if args.comment:
        model_tag += f"_{args.comment}"
    model_save_path = os.path.join('models', model_tag)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    transforms_ = transforms.Compose([
        lambda x: pad(x),
        lambda x: Tensor(x)
    ])

    # Build dev/eval dataset
    dev_set = data_utils_LA.ASVDataset(
        database_path=args.database_path,
        protocols_path=args.protocols_path,
        is_train=(not args.eval),  # If we do --eval => is_train=False => dev or eval
        is_logical=(args.track == 'logical'),
        transform=transforms_,
        feature_name=args.features,
        is_eval=args.is_eval,
        eval_part=args.eval_part,
        llm_csv_path=args.llm_csv_path
    )

    # Force use of "cuda:5" if available
    device = 'cuda:5' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if bool(parser1.get('mg', False)):
        # multi-gpu
        model_1gpu = RawNet(parser1['model'], device)
        model = model_1gpu.to(device)
    else:
        model = RawNet(parser1['model'], device).to(device)

    nb_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {nb_params} params.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Possibly load pretrained
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model loaded : {args.model_path}")

    # If --eval => produce a file and exit
    if args.eval:
        produce_evaluation_file(dev_set, model, device, args.eval_output)
        sys.exit(0)

    # Otherwise => do training
    train_set = data_utils_LA.ASVDataset(
        database_path=args.database_path,
        protocols_path=args.protocols_path,
        is_train=True,
        is_logical=(args.track == 'logical'),
        transform=transforms_,
        feature_name=args.features,
        is_eval=False,
        llm_csv_path=None
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True)

    writer = SummaryWriter(f"logs/{model_tag}")
    best_acc = 0.0

    for epoch in range(args.num_epochs):
        train_loss, train_acc = train_epoch(train_loader, model, args.lr, optimizer, device)
        dev_acc = evaluate_accuracy(dev_loader, model, device)

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_accuracy', train_acc, epoch)
        writer.add_scalar('dev_accuracy', dev_acc, epoch)

        print(f"\nEpoch {epoch+1}/{args.num_epochs} => "
              f"Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}, Dev Acc: {dev_acc:.2f}")
        print("-"*50)

        if dev_acc > best_acc:
            best_acc = dev_acc
            print(f"New best model at epoch {epoch+1} => Dev Acc {best_acc:.2f}")
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))

        # Save checkpoint for each epoch
        torch.save(model.state_dict(), os.path.join(model_save_path, f"epoch_{epoch+1}.pth"))
