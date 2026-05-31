import os
import glob
import torch
from data import load_data
from model import GSSR, Trainer
from parser import parse_args
from utils import set_seed, print_args
from loguru import logger

def configure_model_weights(model, args):
    model.cl_weight = args.cl_weight
    model.comp_loss_weight = 0.0

    logger.info("=" * 80)
    logger.info(f"CL Weight: {args.cl_weight}, Decor Weight: {args.decor_weight}")
    logger.info("=" * 80)

    return model


def main(args):
    print_args(args)
    if args.seed is not None:
        set_seed(args.seed)

    # ========== Load data ==========
    data = load_data(args.data_dir, args.dataset, args.device, args.batch_size)

    # ========== Check Semantic Graphs ==========
    comp_graph_path = os.path.join(args.data_dir, args.dataset, 'item_comp_graph.npz')
    sub_graph_path = os.path.join(args.data_dir, args.dataset, 'item_sub_graph.npz')

    use_semantic = os.path.exists(comp_graph_path) and os.path.exists(sub_graph_path)

    if use_semantic:
        logger.info(f"✓ Semantic graphs found (Ready for Structure Learning):")
        logger.info(f"  - Base Graph 1: {comp_graph_path}")
        logger.info(f"  - Base Graph 2: {sub_graph_path}")
    else:
        logger.warning(f"✗ Semantic graphs NOT found!")
        logger.warning(f"  Please run: python build_graphs_simple.py --dataset {args.dataset}")


    logger.info("Building GSSR Model...")

    model = GSSR(
        data=data,
        emb_dim=args.emb_dim,
        gnn_layers=args.gnn_layers,
        tsp_layers=args.tsp_layers,
        dataset=args.dataset,
        data_dir=args.data_dir,
        decor_weight=args.decor_weight,
    ).to(args.device)

    
    model = configure_model_weights(model, args)

    # ========== Initialize Trainer ==========
    trainer = Trainer(model, data, args)

    if args.load_checkpoint:
        checkpoint_path = getattr(args, 'checkpoint_path', '')
        if checkpoint_path:
            best_model_path = os.path.join(checkpoint_path, 'best_model.pt')
        else:
            dataset_ckpt_dir = os.path.join(args.checkpoint_dir, args.dataset)
            run_dirs = sorted(glob.glob(os.path.join(dataset_ckpt_dir, '2*')))
            if run_dirs:
                best_model_path = os.path.join(run_dirs[-1], 'best_model.pt')
            else:
                best_model_path = os.path.join(dataset_ckpt_dir, 'best_model.pt')

        logger.info(f"Loading checkpoint from {best_model_path}")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=args.device))
            logger.info("✓ Checkpoint loaded successfully")
        else:
            logger.error(f"✗ Checkpoint file not found: {best_model_path}")
            return
    else:
        logger.info("Starting Self-Evolving Training...")
        trainer.train_model()

        best_model_path = trainer.best_model_path
        logger.info(f"Loading best model from {best_model_path} ...")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=args.device))
            logger.info("✓ Best model loaded")
        else:
            logger.warning("⚠ Best model checkpoint not found! Using the last epoch model instead.")

    # ========== Evaluate trained model (Standard) ==========
    logger.info("=" * 80)
    logger.info("Starting final evaluation on Test Set...")

    results = trainer.evaluate()

    logger.info("=" * 80)
    logger.info(f"FINAL TEST RESULTS ({args.dataset}):")
    logger.info("-" * 80)
    logger.info(f"{'Top-K':<10} | {'HR':<10} | {'Recall':<10} | {'NDCG':<10}")
    logger.info("-" * 80)

    for k in sorted(results.keys()):
        metrics = results[k]
        logger.info(f"@{k:<9} | {metrics.get('hit', 0):.4f}     | {metrics['recall']:.4f}     | {metrics['ndcg']:.4f}")

    logger.info("=" * 80)

    logger.info("=" * 80)


if __name__ == '__main__':
    args = parse_args()
    main(args)
