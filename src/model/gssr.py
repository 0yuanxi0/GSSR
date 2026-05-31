import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import os
from .graph_conv import BehaviorConv
from .agg import AGG


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
        self.gamma = 1e-10

    def forward(self, p_score, n_score):
        loss = -torch.log(self.gamma + torch.sigmoid(p_score - n_score))
        return loss.mean()


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.2):
        super(InfoNCELoss, self).__init__()
        self.tau = temperature

    def forward(self, view1, view2):
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
        logits = torch.mm(view1, view2.t()) / self.tau
        labels = torch.arange(logits.size(0)).to(logits.device)
        return F.cross_entropy(logits, labels)


class GatedFusion(nn.Module):
    def __init__(self, emb_dim):
        super(GatedFusion, self).__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 1)
        )
        nn.init.constant_(self.gate_net[-1].bias, 2.0)

    def forward(self, beh_emb, sem_emb):
        concat = torch.cat([beh_emb, sem_emb], dim=1)
        alpha = torch.sigmoid(self.gate_net(concat))
        
        final_emb = alpha * beh_emb + (1 - alpha) * sem_emb
        return final_emb, alpha


class ResidualFusion(nn.Module):
    """残差式融合: final = beh + tanh(λ) * norm(sem)
    - λ ∈ (-1, 1), 防止训练频率差异导致λ失控
    - n_nodes>0: 每节点独立λ (Embedding)
    - n_nodes=0: 全局标量λ (用于 item 侧)"""
    def __init__(self, emb_dim, n_nodes=0):
        super(ResidualFusion, self).__init__()
        if n_nodes > 0:
            self.mode = 'per_node'
            self.node_lambda = nn.Embedding(n_nodes + 1, 1, padding_idx=0)
            nn.init.zeros_(self.node_lambda.weight)
        else:
            self.mode = 'scalar'
            self.lambda_ = nn.Parameter(torch.tensor(0.0))

    def forward(self, beh_emb, sem_emb):
        n = beh_emb.size(0)
        sem_normalized = F.normalize(sem_emb, dim=1)
        if self.mode == 'per_node':
            idx = torch.arange(n, device=beh_emb.device)
            lamb = torch.tanh(self.node_lambda(idx))  # [-1, 1] 有界
        else:
            lamb = self.lambda_
        final_emb = beh_emb + lamb * sem_normalized
        return final_emb, lamb


class LearnableSemanticPropagator(nn.Module):
    """可学习语义传播: 静态 npz 图(先验) + 低秩残差图(可学习新边)
    模型可以发现共现统计没覆盖到的 item-item 关系"""
    def __init__(self, data, emb_dim, sem_layers=1, residual_rank=32, topk_res=10):
        super(LearnableSemanticPropagator, self).__init__()
        self.n_items = data['n_items']
        self.sem_layers = sem_layers
        self.topk_res = topk_res

        # 加载静态 npz 图
        device = data.get('device', 'cpu')
        self.sub_adj = self._load_adj(data, 'item_sub_graph.npz', device)
        self.comp_adj = self._load_adj(data, 'item_comp_graph.npz', device)

        # 边权重微调
        self.sub_scale = nn.Parameter(torch.ones(self.sub_adj._nnz()))
        self.comp_scale = nn.Parameter(torch.ones(self.comp_adj._nnz()))

        # === 可学习残差图: 低秩分解 U@V^T, 允许发现新边 ===
        self.U = nn.Parameter(torch.randn(self.n_items, residual_rank) * 0.01)
        self.V = nn.Parameter(torch.randn(self.n_items, residual_rank) * 0.01)
        self.res_alpha = nn.Parameter(torch.tensor(0.0))  # 残差图混合系数

        # 融合门控
        self.sem_gate = nn.Linear(emb_dim * 2, emb_dim)
        nn.init.xavier_uniform_(self.sem_gate.weight)
        nn.init.zeros_(self.sem_gate.bias)

    def _load_adj(self, data, filename, device):
        path = os.path.join(data['data_dir'], filename)
        if not os.path.exists(path):
            # 兼容旧路径: data_dir/dataset/filename
            path = os.path.join(data['data_dir'], data['dataset'], filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} not found for learnable semantic')
        mat = sp.load_npz(path)
        if mat.shape[0] > self.n_items:
            mat = mat.tocoo()
            r, c = mat.row - 1, mat.col - 1
            valid = (r >= 0) & (c >= 0) & (r < self.n_items) & (c < self.n_items)
            mat = sp.coo_matrix((mat.data[valid], (r[valid], c[valid])),
                                shape=(self.n_items, self.n_items))
        # 归一化 (对称归一化 = D^{-1/2} A D^{-1/2})
        rowsum = np.array(mat.sum(1)).flatten()
        with np.errstate(divide='ignore'):
            d_inv = np.power(rowsum, -0.5)
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_mat = d_mat.dot(mat).dot(d_mat).tocoo()

        idx = torch.LongTensor(np.vstack((norm_mat.row, norm_mat.col))).to(device)
        val = torch.FloatTensor(norm_mat.data).to(device)
        return torch.sparse_coo_tensor(idx, val, torch.Size([self.n_items, self.n_items]))

    def _get_adj(self, base_adj, scale):
        """用可学习 scale 调整边权重"""
        vals = base_adj._values() * F.softplus(scale)
        vals = vals / (vals.sum() + 1e-8)
        return torch.sparse_coo_tensor(
            base_adj._indices(), vals, base_adj.shape)

    def _build_residual_adj(self, item_emb):
        """低秩残差图: U@V^T → top-k 稀疏化 → 发现新边"""
        score = self.U @ self.V.T           # [I, I] low-rank
        score.fill_diagonal_(0)
        I = self.n_items

        chunk = 1024
        rows, cols, vals = [], [], []
        for i in range(0, I, chunk):
            end = min(i + chunk, I)
            s_chunk = score[i:end]                    # [chunk, I]
            topk_v, topk_idx = torch.topk(s_chunk, k=self.topk_res, dim=-1)
            batch_rows = torch.arange(i, end, device=item_emb.device)
            batch_rows = batch_rows.unsqueeze(1).expand(-1, self.topk_res).reshape(-1)
            rows.append(batch_rows)
            cols.append(topk_idx.reshape(-1))
            vals.append(F.softplus(topk_v).reshape(-1))

        rows = torch.cat(rows); cols = torch.cat(cols); vals = torch.cat(vals)
        vals = vals / (vals.sum() + 1e-8)
        adj = torch.sparse_coo_tensor(
            torch.stack([rows, cols]), vals, torch.Size([I, I]))
        return adj

    def forward(self, item_emb):
        x = item_emb[1:]  # [I, d]

        # 静态图 + 边权重微调
        sub_a = self._get_adj(self.sub_adj, self.sub_scale)
        comp_a = self._get_adj(self.comp_adj, self.comp_scale)

        # 低秩残差图 (可学习新边)
        if self.res_alpha > 0:
            res_a = self._build_residual_adj(x)
            sub_a = sub_a + self.res_alpha * res_a
            comp_a = comp_a + self.res_alpha * res_a

        sub_emb = x
        comp_emb = x
        for _ in range(self.sem_layers):
            sub_emb = torch.sparse.mm(sub_a, sub_emb)
            comp_emb = torch.sparse.mm(comp_a, comp_emb)

        gate_input = torch.cat([sub_emb, comp_emb], dim=1)
        g = torch.sigmoid(self.sem_gate(gate_input))
        sem_emb = g * sub_emb + (1 - g) * comp_emb

        zero = torch.zeros(1, item_emb.shape[1], device=item_emb.device)
        return torch.cat([zero, sem_emb], 0)


class SemanticPropagator(nn.Module):
    def __init__(self, data, emb_dim, sem_layers=2):
        super(SemanticPropagator, self).__init__()
        self.data = data
        self.n_items = data['n_items']
        self.sem_layers = sem_layers
        self.sub_adj = self._load_adj('item_sub_graph.npz')  # 替代图 
        self.comp_adj = self._load_adj('item_comp_graph.npz')  # 互补图
        self.sem_gate = nn.Linear(emb_dim * 2, emb_dim)
        nn.init.xavier_uniform_(self.sem_gate.weight)
        nn.init.zeros_(self.sem_gate.bias)

    def _load_adj(self, filename):
        path = os.path.join(self.data['data_dir'], self.data['dataset'], filename)
        if not os.path.exists(path): return None
        try:
            mat = sp.load_npz(path)
            # 1-based 转 0-based 逻辑
            if mat.shape[0] > self.n_items:
                mat = mat.tocoo()
                r, c = mat.row - 1, mat.col - 1
                valid = (r >= 0) & (c >= 0) & (r < self.n_items) & (c < self.n_items)
                mat = sp.coo_matrix((mat.data[valid], (r[valid], c[valid])), shape=(self.n_items, self.n_items))

            # 归一化
            rowsum = np.array(mat.sum(1)).flatten()
            with np.errstate(divide='ignore'):
                d_inv = np.power(rowsum, -0.5)
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            norm_mat = d_mat.dot(mat).dot(d_mat).tocoo()

            # 转 Tensor
            idx = torch.LongTensor(np.vstack((norm_mat.row, norm_mat.col)))
            val = torch.FloatTensor(norm_mat.data)
            return torch.sparse_coo_tensor(idx, val, torch.Size(norm_mat.shape)).to(self.data['device'])
        except:
            return None

    def forward(self, item_emb):
        # 处理 padding (item_emb: N+1)
        x = item_emb[1:]  # 去掉 0 号

        
        sub_emb = x
        if self.sub_adj is not None:
            # 简单的 GCN 传播
            for _ in range(self.sem_layers):
                sub_emb = torch.sparse.mm(self.sub_adj, sub_emb)

        
        comp_emb = x
        if self.comp_adj is not None:
            for _ in range(self.sem_layers):
                comp_emb = torch.sparse.mm(self.comp_adj, comp_emb)

        # 融合
        # sem_emb = (sub_emb + comp_emb) / 2
        gate_input = torch.cat([sub_emb, comp_emb], dim=1)  # [N, 2*Dim]
        g = torch.sigmoid(self.sem_gate(gate_input))
        # 融合公式: g * sub + (1-g) * comp
        sem_emb = g * sub_emb + (1 - g) * comp_emb
        # 补回 padding
        zero = torch.zeros(1, item_emb.shape[1], device=item_emb.device)
        return torch.cat([zero, sem_emb], 0)


class GSSR(nn.Module):
    def __init__(self, data, emb_dim, gnn_layers, tsp_layers, **kwargs):
        super(GSSR, self).__init__()
        self.data = data
        self.edge_dict = data['edge_dict']
        self.n_users = data['n_users']
        self.n_items = data['n_items']
        self.emb_dim = emb_dim  # 记录维度
        self.args = kwargs.get('args', None)
        sem_layers = kwargs.get('sem_layers', 1)

       
        self.bsg_types = data['bsg_types']
        self.hes_types = data['hes_types']      # hesitant: not_buy edges
        self.conf_types = data['conf_types']    # confident: buy edges
        self.hc_types = self.hes_types + self.conf_types
        self.total_behaviors = ['ubg'] + self.bsg_types + self.hc_types

        self.agg = AGG(emb_dim, self.bsg_types, self.hc_types)

        self.convs = nn.ModuleDict()
        for b in self.total_behaviors:
            norm = 'target' if b in self.hes_types else 'gcn'
            n_layers = tsp_layers if norm == 'target' else gnn_layers
            self.convs[b] = nn.ModuleList([
                BehaviorConv(emb_dim, norm=norm)
                for _ in range(n_layers)
            ])

        # Semantic & Fusion (learnable + gate fusion)
        data_with_meta = data.copy()
        data_with_meta['dataset'] = kwargs.get('dataset', 'taobao')
        data_with_meta['data_dir'] = kwargs.get('data_dir', './data')
        self.sem_propagator = LearnableSemanticPropagator(
            data_with_meta, emb_dim, sem_layers=sem_layers)
        self.user_fusion = GatedFusion(emb_dim)
        self.item_fusion = GatedFusion(emb_dim)

        # Loss & Embeddings
        self.bpr_loss = BPRLoss()
        self.cl_loss_func = InfoNCELoss(temperature=0.2)
        self.cl_weight = getattr(self, 'cl_weight', 0.05)
        self.decor_weight = kwargs.get('decor_weight', 0.5)

        self.user_embedding = nn.Embedding(self.n_users + 1, emb_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, emb_dim, padding_idx=0)
        # 语义塔独立 item embedding, 不受 BPR 梯度污染
        self.sem_item_embedding = nn.Embedding(self.n_items + 1, emb_dim, padding_idx=0)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.sem_item_embedding.weight)
        self.agg.reset_parameters()


    def propagate(self, x, edge_index, behavior_type, target_emb=None):
        result = [x]

        for i, conv in enumerate(self.convs[behavior_type]):
            x = conv(x, edge_index, target_emb)
            x = F.normalize(x, dim=1)
            result.append(x / (i + 1))
        result = torch.stack(result, dim=1)
        x = result.sum(dim=1)
        return x
    def forward(self):
        edge_dict = self.edge_dict
        emb_dict = dict() 
        init_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

        # 1. Left Tower
        ubg = self.propagate(init_emb, edge_dict['ubg'], 'ubg')
        emb_dict['ubg'] = ubg

        for b in self.bsg_types:
            emb_dict[b] = self.propagate(ubg, edge_dict[b], b)

        target = emb_dict['buy']
        for b in self.conf_types:
            prev = b.split('_')[0]
            emb_dict[b] = self.propagate(emb_dict[prev], edge_dict[b], b)

        for b in self.hes_types:
            prev = b.split('_')[0]
            emb_dict[b] = self.propagate(emb_dict[prev], edge_dict[b], b, target_emb=target)

        beh_final = self.agg(emb_dict)
        beh_user, beh_item = torch.split(beh_final, [self.n_users + 1, self.n_items + 1], dim=0)

        sem_item_raw = self.sem_propagator(self.sem_item_embedding.weight)
        sem_user_raw = self.propagate(torch.cat([self.user_embedding.weight, sem_item_raw], 0),
                                      edge_dict['ubg'], 'ubg')
        sem_user = sem_user_raw[:self.n_users + 1]
        sem_item = sem_item_raw

        final_u, alpha_u = self.user_fusion(beh_user, sem_user)
        final_i, alpha_i = self.item_fusion(beh_item, sem_item)

        
        gate_user = None
        gate_alpha_mean = None
        gate_alpha_min = None
        gate_alpha_max = None
        if alpha_u is not None:
            gate_user = alpha_u.squeeze(-1)  # [n_users+1]
            # 去掉padding (index 0)
            valid_gate = gate_user[1:]
            gate_alpha_mean = valid_gate.mean()
            gate_alpha_min = valid_gate.min()
            gate_alpha_max = valid_gate.max()

        output = {
            'final_user': final_u,
            'final_item': final_i,
            'beh_user': beh_user,
            'sem_user': sem_user,
            'gate_alpha': gate_alpha_mean,
            'gate_alpha_min': gate_alpha_min,
            'gate_alpha_max': gate_alpha_max,
            'gate_user': gate_user,
            'final': torch.cat([final_u, final_i], 0)
        }
        output.update(emb_dict)
        return output

    def loss(self, users, pos, neg, cl_weight=0.01, **kwargs):
        out = self.forward()
        u, p, n = out['final_user'][users], out['final_item'][pos], out['final_item'][neg]
        bpr = self.bpr_loss((u * p).sum(1), (u * n).sum(1))

        u_beh = out['beh_user'][users]
        u_sem = out['sem_user'][users]

        cl = self.cl_loss_func(u_beh, u_sem)
        bpr += cl_weight * cl

        # 去相关损失: 惩罚双塔输出过于相似, 迫使语义塔学习互补信息
        if self.decor_weight > 0:
            beh_n = F.normalize(out['beh_user'][1:], dim=1)  # 去padding
            sem_n = F.normalize(out['sem_user'][1:], dim=1)
            cos = (beh_n * sem_n).sum(dim=1)  # [n_users]
            decor = torch.mean(cos ** 2)
            bpr += self.decor_weight * decor

        return bpr
