from tqdm import tqdm
from time import time
import torch
from torch.utils.data import DataLoader
import argparse
import numpy as np
import os
import utils
from sklearn.metrics import roc_auc_score, log_loss

from model.simulator import *
from reader import *


try:
    from model.OneRecUserResponse import OneRecUserResponse
except ImportError:
    pass

try:
    from reader.KRMBSeqReaderOneRec import KRMBSeqReaderOneRec
except ImportError:
    pass

try:
    from model.DecisionTransformerRec import DecisionTransformerRec
except ImportError:
    pass

try:
    from reader.KRMBSeqReaderDT import KRMBSeqReaderDT
except ImportError:
    pass

def do_eval(model, reader, args):
    reader.set_phase("val")
    eval_loader = DataLoader(reader, batch_size=args.val_batch_size,
                             shuffle=False, pin_memory=False, 
                             num_workers=reader.n_worker)
    
    val_report = {'loss': [], 'auc': {}, 'logloss': {}}
    Y_dict = {f: [] for f in model.feedback_types}
    P_dict = {f: [] for f in model.feedback_types}
    
    pbar = tqdm(total=len(reader), desc="Validating")
    
    with torch.no_grad():
        for i, batch_data in enumerate(eval_loader):
            wrapped_batch = utils.wrap_batch(batch_data, device=args.device)
            out_dict = model.do_forward_and_loss(wrapped_batch)
            
            loss = out_dict['loss']
            val_report['loss'].append(loss.item())
            
            preds = out_dict['preds']
            if preds.dim() == 2:
                preds = preds.unsqueeze(1) # [B, F] -> [B, 1, F]
            
            for j, f in enumerate(model.feedback_types):
                y_true = wrapped_batch[f].view(-1).detach().cpu().numpy()
                Y_dict[f].append(y_true)
                
                p_logits = preds[:, :, j].view(-1).detach().cpu()
                p_prob = torch.sigmoid(p_logits).numpy()
                
                P_dict[f].append(p_prob)
                
            pbar.update(args.batch_size)
            
    val_report['loss'] = (np.mean(val_report['loss']), np.min(val_report['loss']), np.max(val_report['loss']))
    
    for f in model.feedback_types:
        y_all = np.concatenate(Y_dict[f])
        p_all = np.concatenate(P_dict[f])
        
        if len(np.unique(y_all)) > 1:
            val_report['auc'][f] = roc_auc_score(y_all, p_all)
            val_report['logloss'][f] = log_loss(y_all, p_all, labels=[0, 1])
        else:
            val_report['auc'][f] = 0.5
            val_report['logloss'][f] = 0.0
            
    pbar.close()
    return val_report


if __name__ == '__main__':
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    init_parser = argparse.ArgumentParser()
    init_parser.add_argument('--reader', type=str, required=True, help='Data reader class')
    init_parser.add_argument('--model', type=str, required=True, help='User response model class.')
    initial_args, _ = init_parser.parse_known_args()
    print(f"Initializing Reader: {initial_args.reader}, Model: {initial_args.model}")
    
    try:
        modelClass = eval(initial_args.model)
        readerClass = eval(initial_args.reader)
    except NameError as e:
        print(f"Error loading classes: {e}")
        print("Please make sure the model and reader files are correctly imported in train_multibehavior.py")
        exit(1)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=9, help='random seed')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--val_batch_size', type=int, default=128, help='validation batch size')
    parser.add_argument('--test_batch_size', type=int, default=128, help='test batch size')
    parser.add_argument('--save_with_val', action='store_true', help='save when validation check is true')
    parser.add_argument('--epoch', type=int, default=10, help='number of epoch')
    parser.add_argument('--cuda', type=int, default=-1, help='cuda device number; set to -1 (default) if using cpu')
    
    parser = modelClass.parse_model_args(parser)
    parser = readerClass.parse_data_args(parser)
    args, _ = parser.parse_known_args()
    
    utils.set_random_seed(args.seed)
    
    reader = readerClass(args)
    print('Data statistics:\n', reader.get_statistics())
    
    if args.cuda >= 0 and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
        torch.cuda.set_device(args.cuda)
        device = f"cuda:{args.cuda}"
    else:
        device = "cpu"
    args.device = device
    print(f"Using device: {device}")
    
    model = modelClass(args, reader.get_statistics(), device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.optimizer = optimizer
    
    try:
        best_metric = 0 #  AUC 
        best_auc = {f: 0 for f in model.feedback_types}
        
        print(f"Validation before training...")
        val_report = do_eval(model, reader, args)
        print(f"Initial Val Result: Loss={val_report['loss'][0]:.4f}, AUC={val_report['auc']}")
        
        epo = 0
        stop_count = 0
        while epo < args.epoch:
            epo += 1
            print(f"\nEpoch {epo}/{args.epoch} Training:")
            
            model.train()
            reader.set_phase("train")
            train_loader = DataLoader(reader, batch_size=args.batch_size, 
                                      shuffle=True, pin_memory=True,
                                      num_workers=reader.n_worker)
            t1 = time()
            pbar = tqdm(total=len(reader), desc=f"Epoch {epo}")
            step_loss = []
            step_behavior_loss = {fb: [] for fb in model.feedback_types}
            
            for i, batch_data in enumerate(train_loader):
                optimizer.zero_grad()
                wrapped_batch = utils.wrap_batch(batch_data, device=device)
                
                if wrapped_batch['user_id'].shape[0] == 0:
                    continue

                out_dict = model.do_forward_and_loss(wrapped_batch)
                loss = out_dict['loss']
                loss.backward()
                
                step_loss.append(loss.item())
                if 'behavior_loss' in out_dict:
                    for fb, v in out_dict['behavior_loss'].items():
                        step_behavior_loss[fb].append(v)
                        
                optimizer.step()
                pbar.update(args.batch_size)
                
                if i % 100 == 0:
                    avg_loss = np.mean(step_loss[-100:])
                    pbar.set_postfix({'loss': f"{avg_loss:.4f}"})
            
            pbar.close()
            print("Epoch {} finished in {:.4f}s. Avg Loss: {:.4f}".format(epo, time() - t1, np.mean(step_loss)))

            t2 = time()
            print(f"Epoch {epo} Validating...")
            val_report = do_eval(model, reader, args)
            
            print(f"Val Loss: {val_report['loss'][0]:.4f}")
            print(f"Val AUC: {val_report['auc']}")
            print(f"Val LogLoss: {val_report['logloss']}")
            
            improve = 0
            current_avg_auc = np.mean(list(val_report['auc'].values()))
            
            for f, v in val_report['auc'].items():
                if v > best_auc[f]:
                    best_auc[f] = v
            
            
            count_improve = sum([1 for f in model.feedback_types if val_report['auc'][f] >= best_auc[f] - 1e-4])
            
            if args.save_with_val:
                if count_improve >= 0.5 * len(model.feedback_types):
                    model.save_checkpoint()
                    stop_count = 0
                    print(f"Checkpoint saved at epoch {epo}")
                else:
                    stop_count += 1
                    print(f"No improvement. Stop count: {stop_count}/3")
                
                if stop_count >= 3:
                    print("Early stopping triggered.")
                    break
            else:
                model.save_checkpoint()
                print(f"Checkpoint saved (No early stopping).")
            
    except KeyboardInterrupt:
        print("Early stop manually")
        exit(1)