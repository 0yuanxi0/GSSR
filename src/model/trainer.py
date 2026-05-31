import torch
import numpy as np
import os
import time
from datetime import datetime
from loguru import logger
from tqdm import tqdm
from .metrics import ndcg, hit, recall
from .prune import LightweightPruner


class Trainer:
    def __init__(self, model, data, args):
        self.model = model
        self.data = data
        self.args = args
        self.topk = args.topk
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        
        if isinstance(args.topk, int):
            self.topk_list = [args.topk]
        else:
            self.topk_list = sorted(args.topk)  

        
        self.max_k = self.topk_list[-1]

        self.pruner = LightweightPruner(
            model=self.model, data=self.data,
            data_dir_for_C=os.path.join(args.data_dir, args.dataset),
            thresh=getattr(args, 'prune_thresh', 0.40),
            min_deg_item=3,
            max_drop_ratio=0.05,
            warmup_epochs=8, prune_every=2,
            mem_topk=100, mem_beta=0.8,
        )

        # 时间戳子目录隔离多次运行
        seed = getattr(args, 'seed', 0)
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + f'_s{seed}'
        self.checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.args.dataset, self.run_id)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pt')

        log_path = os.path.join(self.checkpoint_dir, 'train.log')
        logger.add(log_path, rotation="10 MB", level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

        self.best_results = {}
        self.best_epoch = 0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        train_loader = self.data['train_loader']


        base_cl_weight = getattr(self.model, 'cl_weight', 0.03)
        if epoch < 5:
            current_cl_weight = 0.0
        elif epoch < 15:
            current_cl_weight = base_cl_weight * ((epoch - 5) / 10)  # Rampup
        else:
            current_cl_weight = base_cl_weight  # Stable

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}', unit='batch', leave=False) as pbar:
            for user_indices, pos_indices, neg_indices in train_loader:
                user_indices = user_indices.to(self.args.device)
                pos_indices = pos_indices.to(self.args.device)
                neg_indices = neg_indices.to(self.args.device)


                loss = self.model.loss(
                    user_indices,
                    pos_indices,
                    neg_indices,
                    cl_weight=current_cl_weight
                )

                self.optimizer.zero_grad()
                loss.backward()


                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'CL': f'{current_cl_weight:.3f}'
                })
                pbar.update()
                total_loss += loss.item()

        
        return total_loss / len(train_loader)

    def train_model(self):
        patience = getattr(self.args, 'patience', 10)
        best_ndcg = 0.0
        patience_cnt = 0
        k_list = self.topk_list

        logger.info(f"Start training on {self.args.dataset} (Patience: {patience}, Run: {self.run_id})...")

        for epoch in range(self.args.num_epochs):
            t0 = time.time()
            loss = self.train_epoch(epoch)
            self.pruner.maybe_prune(epoch)

            results = self.evaluate()
            elapsed = time.time() - t0

            monitor_k = 10 if 10 in k_list else k_list[-1]
            ndcg_val = results[monitor_k]['ndcg']
            improved = ndcg_val > best_ndcg

            metric_parts = []
            for k in k_list:
                h = results[k].get('hit', 0)
                r = results[k]['recall']
                n = results[k]['ndcg']
                metric_parts.append(f'@{k} H={h:.4f} R={r:.4f} N={n:.4f}')

            marker = ' *BEST*' if improved else ''
            logger.info(
                f'Epoch {epoch+1:3d} | Loss {loss:.4f} | {elapsed:.0f}s{marker}\n'
                f'         {" | ".join(metric_parts)}'
            )

            for k in k_list:
                n_val = results[k]['ndcg']
                if k not in self.best_results or n_val > self.best_results[k]['ndcg']:
                    self.best_results[k] = {
                        'recall': results[k]['recall'],
                        'ndcg': n_val,
                        'hit': results[k].get('hit', 0),
                        'epoch': epoch + 1
                    }

            if improved:
                best_ndcg = ndcg_val
                patience_cnt = 0
                self.best_epoch = epoch + 1
                torch.save(self.model.state_dict(), self.best_model_path)
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        logger.info(f"Training finished. Best NDCG@10: {best_ndcg:.4f} (epoch {self.best_epoch})")

    def evaluate(self):
        device = self.args.device
        self.model.eval()

        topk_list = []
        gt_len_list = []
        real_max_k = max(self.topk_list)

        with torch.no_grad():
            out = self.model.forward()
            final_emb = out['final'] if isinstance(out, dict) else out
            n_users = self.model.n_users
            n_items = self.model.n_items
            final_user_emb, final_item_emb = torch.split(final_emb, [n_users + 1, n_items + 1])

            for user_indices, _, _ in tqdm(self.data['test_loader'], desc='Test', leave=False):
                user_indices = user_indices.to(device)
                batch_user_emb = final_user_emb[user_indices]
                scores = torch.matmul(batch_user_emb, final_item_emb.transpose(0, 1))

                for user_idx in range(user_indices.size(0)):
                    user = user_indices[user_idx].item()
                    train_items = self.data['train_gt'].get(str(user), [])
                    scores[user_idx, train_items] = -np.inf

                _, topk_indices = torch.topk(scores, real_max_k, dim=1)
                topk_indices = topk_indices.cpu().numpy()

                for idx, user in enumerate(user_indices):
                    user_id = str(user.item())
                    gt_items = np.array(self.data['test_gt'][user_id])
                    gt_len_list.append(len(gt_items))
                    topk_items = topk_indices[idx]
                    mask = np.isin(topk_items, gt_items)
                    topk_list.append(mask)

        topk_matrix = np.vstack(topk_list)
        gt_lens = np.array(gt_len_list)
        recall_matrix = recall(topk_matrix, gt_lens)
        ndcg_matrix = ndcg(topk_matrix, gt_lens)
        hit_matrix = hit(topk_matrix, gt_lens)

        results = {}
        for k in self.topk_list:
            if k <= real_max_k:
                results[k] = {
                    'recall': recall_matrix[:, k - 1].mean(),
                    'ndcg': ndcg_matrix[:, k - 1].mean(),
                    'hit': hit_matrix[:, k - 1].mean(),
                }
        return results
