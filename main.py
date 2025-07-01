import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import time
import glob
from datetime import datetime
import json
from sklearn.metrics import classification_report

from FreqDGT import FreqDGT
from base.cross_validation import CrossValidation
from base.utils import seed_all, ensure_path
from datasets.SEED import SEED

current_dir = os.getcwd()

def check_processed_data(data_path, num_subjects):
    expected_files = [f'sub{i}.pkl' for i in range(num_subjects)]
    all_exist = True
    
    for file in expected_files:
        if not os.path.exists(os.path.join(data_path, file)):
            all_exist = False
            break
            
    return all_exist, len(glob.glob(os.path.join(data_path, "sub*.pkl")))

class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            factor = alpha
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + np.cos(np.pi * progress))
            
        return [self.min_lr + factor * (base_lr - self.min_lr) for base_lr in self.base_lrs]

class GradientAccumulator:
    def __init__(self, steps=1):
        self.steps = max(1, steps)
        self.current_step = 0
        
    def step(self, loss, optimizer, scaler=None, model=None, max_norm=None):
        if not loss.requires_grad:
            raise ValueError("Loss tensor does not require gradients. Check your model and loss computation.")
            
        scaled_loss = loss / self.steps
        
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
            
        self.current_step += 1
        if self.current_step >= self.steps:
            if max_norm is not None and model is not None:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                
            optimizer.zero_grad()
            self.current_step = 0
            return True
        
        return False

def main():
    parser = argparse.ArgumentParser(description='FreqDGT Training')
    
    parser.add_argument('--ROOT', type=str, default=current_dir)
    parser.add_argument('--dataset', type=str, default='SEED')
    parser.add_argument('--data-path', type=str, default=r"SEED/Preprocessed_EEG")
    parser.add_argument('--processed-data-path', type=str, 
                       default='/root/autodl-tmp/EEG-Lipschitz/data_processed/data_rPSD_SEED_FreqDGT',
                       help='Path to processed data')
    parser.add_argument('--subjects', type=int, default=15)
    parser.add_argument('--data-exist', action='store_true', help='Skip data preparation')
    parser.add_argument('--fold-to-run', type=int, default=7)
    parser.add_argument('--num-class', type=int, default=3, choices=[2, 3, 4])
    parser.add_argument('--label-type', type=str, default='NA', choices=['A', 'V', 'D', 'L', 'NA'])
    parser.add_argument('--session-to-load', default=[1])
    parser.add_argument('--segment', type=int, default=20)
    parser.add_argument('--overlap', type=float, default=0.8)
    parser.add_argument('--sub-segment', type=int, default=8)
    parser.add_argument('--sub-overlap', type=float, default=0.75)
    parser.add_argument('--sampling-rate', type=int, default=200)
    parser.add_argument('--num-channel', type=int, default=62)
    parser.add_argument('--num-time', type=int, default=8)
    parser.add_argument('--num-feature', type=int, default=7)
    parser.add_argument('--loading-key', type=str, default='eeg')
    parser.add_argument('--data-format', type=str, default='rPSD', 
                        choices=['DE', 'Hjorth', 'PSD', 'rPSD', 'sta', 'multi-view', 'raw'])
    parser.add_argument('--split', type=int, default=1)
    parser.add_argument('--sub-split', type=int, default=1)
    parser.add_argument('--extract-feature', type=int, default=1)
    
    parser.add_argument('--random-seed', type=int, default=1234)
    parser.add_argument('--max-epoch', type=int, default=20)
    parser.add_argument('--patient', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--gradient-clip', type=float, default=0.5)
    parser.add_argument('--grad-accum-steps', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--LS', type=int, default=1)
    parser.add_argument('--LS-rate', type=float, default=0.1)
    parser.add_argument('--save-path', default=os.path.join(os.getcwd(), 'save_freqdgt'))
    parser.add_argument('--log-path', default='/root/autodl-tmp/EEG-Lipschitz/par')
    parser.add_argument('--results-path', default='/root/autodl-tmp/EEG-Lipschitz/par')
    parser.add_argument('--save-model', type=int, default=0)
    
    parser.add_argument('--graph-type', type=str, default='BL')
    parser.add_argument('--model', type=str, default='FreqDGT')
    parser.add_argument('--layers-graph', type=list, default=[1, 2])
    parser.add_argument('--layers-transformer', type=int, default=4)
    parser.add_argument('--num-adj', type=int, default=2)
    parser.add_argument('--hidden-graph', type=int, default=64)
    parser.add_argument('--num-head', type=int, default=4)
    parser.add_argument('--dim-head', type=int, default=32)
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--graph2token', type=str, default='Linear', choices=['Linear', 'AvgPool', 'MaxPool', 'Flatten'])
    parser.add_argument('--encoder-type', type=str, default='Cheby', choices=['GCN', 'Cheby'])
    
    parser.add_argument('--feature-type', type=str, default='rPSD', choices=['rPSD', 'DE'])
    parser.add_argument('--enable-disentangle', action='store_true', default=True)
    
    parser.add_argument('--run-twice', action='store_true', default=True)
    parser.add_argument('--second-seed', type=int, default=2345)
    
    parser.add_argument('--reproduce', type=int, default=0)
    parser.add_argument('--force-process', action='store_true')
    
    args = parser.parse_args()
    
    if args.feature_type == 'DE':
        args.data_format = 'DE'
        args.processed_data_path = args.processed_data_path.replace('rPSD', 'DE')
    else:
        args.data_format = 'rPSD'
    
    ensure_path(args.save_path)
    ensure_path(args.log_path)
    ensure_path(args.results_path)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_variant = 'D' if args.feature_type == 'DE' else 'P'
    log_file = os.path.join(args.log_path, f'FreqDGT-{model_variant}_{timestamp}.log')
    
    def log(message):
        print(message)
        with open(log_file, 'a') as f:
            f.write(message + '\n')
    
    log(f"{'='*20} FreqDGT-{model_variant} Training {'='*20}")
    log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Dataset: {args.dataset}")
    log(f"Feature Format: {args.data_format}")
    log(f"Model: {args.model}-{model_variant}")
    log(f"Key Parameters: seed={args.random_seed}, batch_size={args.batch_size}, K={args.K}")
    log(f"Gradient clip: {args.gradient_clip}, Sub-segment: {args.sub_segment}")
    log(f"Run twice: {args.run_twice}")
    
    processed_data_exists, num_files = check_processed_data(args.processed_data_path, args.subjects)
    
    if processed_data_exists and not args.force_process:
        log(f"\n[Found Processed Data] Found {num_files} data files in {args.processed_data_path}")
        log("Skipping data processing...")
        args.data_exist = True
    else:
        if args.force_process:
            log(f"\n[Force Reprocess Data] User specified to ignore processed data")
            args.data_exist = False
        else:
            log(f"\n[Incomplete Processed Data] Found only {num_files}/{args.subjects} data files in {args.processed_data_path}")
        
        if not args.data_exist:
            log("Will process data...")
        else:
            log("User specified to skip data processing, but no complete data found. Please check data path!")
    
    if not args.data_exist:
        log("\n[Data Preprocessing]")
        pd = SEED(args)
        pd.create_dataset(
            np.arange(args.subjects), split=args.split, sub_split=args.sub_split,
            feature=args.extract_feature, band_pass_first=False if args.data_format in ['PSD', 'rPSD', 'raw'] else True
        )
        log("Data preprocessing completed!")
    
    cv = CrossValidation(args)
    
    all_results = []
    
    if args.use_amp:
        log("\n[Mixed Precision Training Enabled]")
    
    run_counts = 2 if args.run_twice else 1
    run_seeds = [args.random_seed, args.second_seed] if args.run_twice else [args.random_seed]
    
    for run_idx, seed in enumerate(run_seeds):
        args.random_seed = seed
        seed_all(args.random_seed)
        
        log(f"\n[Run {run_idx+1}/{run_counts}] Using random seed: {seed}")
        
        run_save_path = os.path.join(args.save_path, f'run{run_idx+1}')
        run_results_path = os.path.join(args.results_path, f'run{run_idx+1}')
        ensure_path(run_save_path)
        ensure_path(run_results_path)
        
        sub_to_run = np.arange(args.subjects)
        
        def train_and_evaluate(data_train, label_train, data_val, label_val, data_test, label_test, subject, trial):
            scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
            
            data_train = torch.FloatTensor(data_train)
            label_train = torch.LongTensor(label_train)
            data_val = torch.FloatTensor(data_val)
            label_val = torch.LongTensor(label_val)
            data_test = torch.FloatTensor(data_test)
            label_test = torch.LongTensor(label_test)
            
            actual_time_steps = data_train.shape[1]
            actual_num_features = data_train.shape[3]
            
            args.num_time = actual_time_steps
            args.num_feature = actual_num_features
            
            log(f"Data shape: [batch, {actual_time_steps}, {args.num_channel}, {actual_num_features}]")
            
            train_dataset = torch.utils.data.TensorDataset(data_train, label_train)
            val_dataset = torch.utils.data.TensorDataset(data_val, label_val)
            test_dataset = torch.utils.data.TensorDataset(data_test, label_test)
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, 
                pin_memory=True, num_workers=2
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False, 
                pin_memory=True, num_workers=2
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False, 
                pin_memory=True, num_workers=2
            )
            
            model = FreqDGT(
                layers_graph=args.layers_graph,
                layers_transformer=args.layers_transformer,
                num_adj=args.num_adj,
                num_chan=args.num_channel,
                num_feature=args.num_feature,
                hidden_graph=args.hidden_graph,
                K=args.K,
                num_head=args.num_head,
                dim_head=args.dim_head,
                dropout=args.dropout,
                num_class=args.num_class,
                alpha=args.alpha,
                graph2token=args.graph2token,
                encoder_type=args.encoder_type,
                sampling_rate=args.sampling_rate,
                feature_type=args.feature_type,
                enable_disentangle=args.enable_disentangle
            )
            
            if args.feature_type == 'DE':
                optimizer = optim.AdamW(
                    model.parameters(), 
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay,
                    betas=(0.9, 0.999)
                )
            else:
                optimizer = optim.Adam(
                    model.parameters(), 
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay
                )
            
            scheduler = WarmupCosineScheduler(
                optimizer,
                warmup_epochs=args.warmup_epochs,
                max_epochs=args.max_epoch,
                min_lr=args.min_lr
            )
            
            criterion = nn.CrossEntropyLoss(label_smoothing=args.LS_rate if args.LS else 0.0)
            
            grad_accumulator = GradientAccumulator(args.grad_accum_steps)
            
            save_dir = os.path.join(run_save_path, f'sub{subject}_trial{trial}')
            if args.save_model:
                os.makedirs(save_dir, exist_ok=True)
            
            def train_epoch(model, dataloader, optimizer):
                model.train()
                total_loss = 0
                correct = 0
                total = 0
                
                for batch_idx, (data, target) in enumerate(dataloader):
                    data, target = data.to('cuda'), target.to('cuda')
                    
                    optimizer.zero_grad()
                    
                    if args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(data)
                            loss = model.get_loss(outputs, target)
                    else:
                        outputs = model(data)
                        loss = model.get_loss(outputs, target)
                    
                    step_taken = grad_accumulator.step(
                        loss, optimizer, scaler, model, args.gradient_clip
                    )
                    
                    if isinstance(outputs, dict):
                        logits = outputs['emotion_logits']
                    else:
                        logits = outputs
                        
                    _, predicted = torch.max(logits.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                    total_loss += loss.item() * args.grad_accum_steps
                    
                return total_loss / len(dataloader), correct / total
            
            def validate(model, dataloader):
                model.eval()
                total_loss = 0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for data, target in dataloader:
                        data, target = data.to('cuda'), target.to('cuda')
                        
                        if args.use_amp:
                            with torch.cuda.amp.autocast():
                                outputs = model(data)
                                loss = model.get_loss(outputs, target)
                        else:
                            outputs = model(data)
                            loss = model.get_loss(outputs, target)
                        
                        if isinstance(outputs, dict):
                            logits = outputs['emotion_logits']
                        else:
                            logits = outputs
                            
                        _, predicted = torch.max(logits.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                        total_loss += loss.item()
                
                return total_loss / len(dataloader), correct / total
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)
            
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log(f"Model parameters: {total_params:,}")
            
            history = {
                'train_loss': [], 'train_acc': [],
                'val_loss': [], 'val_acc': [],
                'learning_rates': [],
                'epoch_times': [], 'best_epoch': 0
            }
            
            best_val_acc = 0
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            
            patience = args.patient
            patience_counter = 0
            
            log(f"\n[Training Subject {subject+1}/{len(sub_to_run)}]")
            
            log(f"\n[Training Configuration]")
            log(f"- Batch size: {args.batch_size}")
            log(f"- Epochs: {args.max_epoch}")
            log(f"- Learning rate: {args.learning_rate}")
            log(f"- Feature type: {args.feature_type}")
            log(f"- Transformer layers: {args.layers_transformer}")
            log(f"- Hidden dimension: {args.hidden_graph}")
            log(f"- Attention heads: {args.num_head}")
            log(f"- Enable disentanglement: {args.enable_disentangle}")
            log(f"- Device: {device}")
            
            total_start_time = time.time()
            
            for epoch in range(args.max_epoch):
                epoch_start_time = time.time()
                
                train_loss, train_acc = train_epoch(model, train_loader, optimizer)
                
                val_loss, val_acc = validate(model, val_loader)
                
                current_lr = optimizer.param_groups[0]['lr']
                scheduler.step()
                
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                history['learning_rates'].append(current_lr)
                
                epoch_time = time.time() - epoch_start_time
                history['epoch_times'].append(epoch_time)
                
                log(f"[Epoch {epoch+1}/{args.max_epoch} Results]")
                log(f"  Train loss: {train_loss:.4f} | Train acc: {train_acc:.2%}")
                log(f"  Val loss: {val_loss:.4f} | Val acc: {val_acc:.2%}")
                log(f"  Learning rate: {current_lr:.6f} | Time: {epoch_time:.2f}s")
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    history['best_epoch'] = epoch + 1
                    
                    if args.save_model:
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_acc': best_val_acc,
                            'train_acc': train_acc,
                            'history': history
                        }, best_model_path)
                    
                    log(f"  [Best Model Saved] Val acc: {best_val_acc:.2%}")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        log(f"  [Early Stopping] No improvement for {patience} epochs")
                        break
            
            total_time = time.time() - total_start_time
            log(f"\n[Training Completed] Total time: {total_time/60:.2f} minutes")
            
            if args.save_model and os.path.exists(best_model_path):
                log(f"\n[Loading Best Model] From epoch {history['best_epoch']}")
                checkpoint = torch.load(best_model_path)
                model.load_state_dict(checkpoint['model_state_dict'])
            
            log(f"\n[Testing Subject {subject+1}/{len(sub_to_run)}]")
            
            all_preds = []
            all_targets = []
            test_loss = 0
            test_acc = 0
            
            model.eval()
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    
                    outputs = model(data)
                    
                    loss = model.get_loss(outputs, target)
                    test_loss += loss.item()
                    
                    if isinstance(outputs, dict):
                        logits = outputs['emotion_logits']
                    else:
                        logits = outputs
                    
                    _, predicted = torch.max(logits.data, 1)
                    test_acc += (predicted == target).sum().item() / len(target)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
            
            test_loss /= len(test_loader)
            test_acc /= len(test_loader)
            
            classification_rep = classification_report(
                all_targets, all_preds,
                target_names=[f'Class{i}' for i in range(args.num_class)],
                output_dict=True
            )
            
            log(f"\n[Test Results]")
            log(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.2%}")
            
            test_results = {
                'test_acc': test_acc,
                'test_loss': test_loss,
                'classification_report': classification_rep,
                'best_val_acc': best_val_acc,
                'training_history': history
            }
            
            with open(os.path.join(run_results_path, f'sub{subject}_results.json'), 'w') as f:
                json.dump(test_results, f, indent=4)
            
            return test_acc, test_results
        
        subject_accs = []
        subject_details = []
        
        log(f"\n[Starting LOSO Cross-Validation] Run {run_idx+1}/{run_counts} (seed: {seed})")
        for idx, sub in enumerate(sub_to_run):
            start_time = time.time()
            
            data_train, label_train = [], []
            
            try:
                data_test, label_test = [], []
                
                with open(os.path.join(args.processed_data_path, f'sub{sub}.pkl'), 'rb') as f:
                    data = pickle.load(f)
                    data_test = data['data']
                    label_test = data['label']
                
                for sub_ in np.arange(args.subjects):
                    if sub != sub_:
                        with open(os.path.join(args.processed_data_path, f'sub{sub_}.pkl'), 'rb') as f:
                            data = pickle.load(f)
                            data_train.extend(data['data'])
                            label_train.extend(data['label'])
                            
                log(f"Successfully loaded data from {args.processed_data_path}")
                
            except Exception as e:
                log(f"Failed to load from custom path, using CrossValidation: {str(e)}")
                data_test, label_test = cv.load_per_subject(sub)
                
                for sub_ in np.arange(args.subjects):
                    if sub != sub_:
                        data_temp, label_temp = cv.load_per_subject(sub_)
                        data_train.extend(data_temp)
                        label_train.extend(label_temp)
            
            data_train, label_train, data_test, label_test = cv.prepare_data(
                data_train=data_train, label_train=label_train, data_test=data_test, label_test=label_test
            )
            
            log(f"\n[Subject {sub+1}/{len(sub_to_run)}]")
            log(f"Train set size: {data_train.shape}, Test set size: {data_test.shape}")
            
            data_train, label_train, data_val, label_val = cv.split_balance_class(
                data=data_train, label=label_train, train_rate=0.8, random=True
            )
            
            test_acc, test_details = train_and_evaluate(
                data_train=data_train,
                label_train=label_train,
                data_val=data_val,
                label_val=label_val,
                data_test=data_test,
                label_test=label_test,
                subject=sub,
                trial=0
            )
            
            subject_accs.append(test_acc)
            subject_details.append(test_details)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            log(f"Subject {sub+1} test acc: {test_acc:.4f}, time: {elapsed_time/60:.2f} minutes")
        
        mean_acc = np.mean(subject_accs)
        std_acc = np.std(subject_accs)
        
        param_result = {
            'run_idx': run_idx + 1,
            'seed': seed,
            'subject_accs': subject_accs,
            'mean_acc': float(mean_acc),
            'std_acc': float(std_acc),
            'subjects_tested': list(map(int, sub_to_run + 1)),
            'feature_type': args.feature_type
        }
        
        all_results.append(param_result)
        
        log(f"\n[Current Run Summary] Run {run_idx+1}/{run_counts} (seed: {seed})")
        log(f"All subject accuracies: {[f'{acc:.4f}' for acc in subject_accs]}")
        log(f"Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        log(f"Best accuracy: {np.max(subject_accs):.4f}, Worst accuracy: {np.min(subject_accs):.4f}")
        
        result_file = os.path.join(args.results_path, f'run{run_idx+1}_seed{seed}_results.json')
        with open(result_file, 'w') as f:
            json.dump(param_result, f, indent=4)
        log(f"Run {run_idx+1} results saved to: {result_file}")
    
    if args.run_twice:
        run1_result = all_results[0]
        run2_result = all_results[1]
        
        run1_acc = run1_result['mean_acc']
        run2_acc = run2_result['mean_acc']
        acc_diff = abs(run1_acc - run2_acc)
        
        log(f"\n[Two-Run Comparison]")
        log(f"First run (seed: {run1_result['seed']}): mean acc {run1_acc:.4f} ± {run1_result['std_acc']:.4f}")
        log(f"Second run (seed: {run2_result['seed']}): mean acc {run2_acc:.4f} ± {run2_result['std_acc']:.4f}")
        log(f"Accuracy difference: {acc_diff:.4f} ({100*acc_diff/run1_acc:.2f}%)")
        
        comparison = {
            'run1': {
                'seed': run1_result['seed'],
                'mean_acc': run1_acc,
                'std_acc': run1_result['std_acc'],
                'subject_accs': run1_result['subject_accs']
            },
            'run2': {
                'seed': run2_result['seed'],
                'mean_acc': run2_acc,
                'std_acc': run2_result['std_acc'],
                'subject_accs': run2_result['subject_accs']
            },
            'diff': {
                'abs_acc_diff': float(acc_diff),
                'percent_diff': float(100*acc_diff/run1_acc),
                'subject_diffs': [float(abs(run1_result['subject_accs'][i] - run2_result['subject_accs'][i])) 
                                for i in range(len(run1_result['subject_accs']))]
            }
        }
        
        comparison_file = os.path.join(args.results_path, f'two_run_comparison.json')
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=4)
        log(f"Two-run comparison results saved to: {comparison_file}")
    
    final_mean_acc = np.mean([result['mean_acc'] for result in all_results])
    
    log("\n" + "="*80)
    log("[FINAL RESULTS]")
    log(f"Feature Type: {args.feature_type}")
    for i, result in enumerate(all_results):
        log(f"Run {i+1} (seed: {result['seed']}): {result['mean_acc']:.4f} ± {result['std_acc']:.4f}")
    log(f"Overall Average: {final_mean_acc:.4f}")
    log("="*80)
    
    full_results_file = os.path.join(
        args.results_path, 
        f'freqdgt_{model_variant}_final_{timestamp}.json'
    )
    with open(full_results_file, 'w') as f:
        json.dump({
            'all_results': all_results,
            'overall_avg': float(final_mean_acc),
            'feature_type': args.feature_type,
            'timestamp': timestamp
        }, f, indent=4)
    log(f"\nAll results saved to: {full_results_file}")

if __name__ == '__main__':
    main()
