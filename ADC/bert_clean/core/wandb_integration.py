#!/usr/bin/env python3
"""
WandB integration for QAT training
"""

import os
import torch
from typing import Dict, Any, Optional, List
from transformers import TrainerCallback, TrainerState, TrainerControl
import logging

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Install with: pip install wandb")


class WandbQATCallback(TrainerCallback):
    """
    Callback to log QAT-specific metrics and visualizations to WandB
    """
    
    def __init__(
        self,
        visualize_layers: Optional[List[str]] = None,
        visualize_every_n_epochs: int = 1,
        log_quantization_stats: bool = True,
        sample_input: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Args:
            visualize_layers: List of layer patterns to visualize (e.g., ["layer.0", "layer.11"])
            visualize_every_n_epochs: How often to create visualizations
            log_quantization_stats: Whether to log quantization statistics
            sample_input: Sample input for visualization (dict with 'input_ids' and 'attention_mask')
        """
        self.visualize_layers = visualize_layers or []
        self.visualize_every_n_epochs = visualize_every_n_epochs
        self.log_quantization_stats = log_quantization_stats
        self.sample_input = sample_input
        self.last_visualized_epoch = -1
        
        if not WANDB_AVAILABLE:
            logger.warning("WandB not available - callback will not log anything")
    
    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Called at the end of each epoch"""
        if not WANDB_AVAILABLE or wandb.run is None:
            return
        
        current_epoch = int(state.epoch) if state.epoch is not None else 0
        
        # Check if we should visualize this epoch
        should_visualize = (
            self.visualize_layers and 
            self.sample_input is not None and
            (current_epoch - self.last_visualized_epoch) >= self.visualize_every_n_epochs
        )
        
        if should_visualize:
            self._log_visualizations(model, current_epoch)
            self.last_visualized_epoch = current_epoch
        
        # Log quantization stats
        if self.log_quantization_stats:
            self._log_quantization_stats(model, current_epoch)
    
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Called when logging metrics"""
        if not WANDB_AVAILABLE or wandb.run is None or logs is None:
            return
        
        # WandB Trainer integration usually handles this, but we can add custom metrics here
        pass
    
    def _log_visualizations(self, model, epoch: int):
        """Generate and log visualizations to WandB"""
        try:
            from .qat_visualizer import QATDebugger
            from .qat_layers import QATLinear
            
            logger.info(f"Generating QAT visualizations for epoch {epoch}...")
            
            # Create debugger
            debugger = QATDebugger(output_dir=f"./qat_viz_epoch_{epoch}")
            
            # Find layers to visualize
            layers_to_viz = []
            for name, module in model.named_modules():
                if isinstance(module, QATLinear):
                    if any(pattern in name for pattern in self.visualize_layers):
                        layers_to_viz.append((name, module))
            
            if not layers_to_viz:
                logger.warning(f"No layers found matching patterns: {self.visualize_layers}")
                return
            
            # Attach hooks
            for name, module in layers_to_viz:
                debugger.attach_to_layer(module, name)
            
            # Run forward pass
            model.eval()
            device = next(model.parameters()).device
            input_ids = self.sample_input['input_ids'].to(device)
            attention_mask = self.sample_input['attention_mask'].to(device)
            
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            model.train()
            
            # Log each layer
            wandb_images = {}
            wandb_stats = {}
            
            for name, _ in layers_to_viz:
                # Generate plot
                fig, data = debugger.plot_layer(name, sample_idx=0)
                
                if fig is not None:
                    # Log image
                    clean_name = name.replace(".", "_")
                    wandb_images[f"qat_viz/{clean_name}"] = wandb.Image(fig)
                
                # Get statistics
                stats = debugger.get_statistics(name)
                for stat_name, stat_value in stats.items():
                    wandb_stats[f"qat_stats/{name}/{stat_name}"] = stat_value
            
            # Log everything
            if wandb_images:
                wandb.log({**wandb_images, "epoch": epoch})
            if wandb_stats:
                wandb.log({**wandb_stats, "epoch": epoch})
            
            logger.info(f"✅ Logged {len(wandb_images)} visualizations and {len(wandb_stats)} stats")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}", exc_info=True)
    
    def _log_quantization_stats(self, model, epoch: int):
        """Log quantization parameters and statistics"""
        try:
            from .qat_layers import QATLinear, LearnableQuantizer
            
            stats = {
                'qat_layers_count': 0,
                'scale_stats': {
                    'weight_scales': [],
                    'activation_scales': [],
                }
            }
            
            for name, module in model.named_modules():
                if isinstance(module, QATLinear):
                    stats['qat_layers_count'] += 1
                    
                    # Weight quantizer stats
                    if hasattr(module, 'weight_quantizer'):
                        w_scale = module.weight_quantizer.scale.detach().cpu()
                        stats['scale_stats']['weight_scales'].append(w_scale.mean().item())
                    
                    # Activation quantizer stats
                    if hasattr(module, 'activation_quantizer'):
                        a_scale = module.activation_quantizer.scale.detach().cpu()
                        stats['scale_stats']['activation_scales'].append(a_scale.mean().item())
            
            # Aggregate statistics
            if stats['scale_stats']['weight_scales']:
                import numpy as np
                w_scales = np.array(stats['scale_stats']['weight_scales'])
                a_scales = np.array(stats['scale_stats']['activation_scales'])
                
                wandb.log({
                    'qat/num_layers': stats['qat_layers_count'],
                    'qat/weight_scale_mean': float(w_scales.mean()),
                    'qat/weight_scale_std': float(w_scales.std()),
                    'qat/weight_scale_min': float(w_scales.min()),
                    'qat/weight_scale_max': float(w_scales.max()),
                    'qat/activation_scale_mean': float(a_scales.mean()),
                    'qat/activation_scale_std': float(a_scales.std()),
                    'qat/activation_scale_min': float(a_scales.min()),
                    'qat/activation_scale_max': float(a_scales.max()),
                    'epoch': epoch,
                })
            
        except Exception as e:
            logger.error(f"Error logging quantization stats: {e}", exc_info=True)


def init_wandb_run(
    project_name: str,
    run_name: str,
    config: Dict[str, Any],
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
) -> Optional[Any]:
    """
    Initialize a WandB run
    
    Args:
        project_name: WandB project name
        run_name: Name for this run
        config: Configuration dictionary (hyperparameters, paths, etc.)
        tags: Optional tags for the run
        notes: Optional notes for the run
    
    Returns:
        wandb.run object or None if WandB not available
    """
    if not WANDB_AVAILABLE:
        logger.warning("WandB not available - skipping initialization")
        return None
    
    try:
        run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            tags=tags or [],
            notes=notes or "",
            reinit=True,
        )
        logger.info(f"✅ Initialized WandB run: {run.name} (id: {run.id})")
        return run
    except Exception as e:
        logger.error(f"Failed to initialize WandB: {e}")
        return None


def log_model_architecture(model, config: Dict[str, Any]):
    """Log model architecture and configuration"""
    if not WANDB_AVAILABLE or wandb.run is None:
        return
    
    try:
        from .qat_layers import QATLinear
        
        # Count different layer types
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        qat_layers = sum(1 for m in model.modules() if isinstance(m, QATLinear))
        total_layers = sum(1 for _ in model.modules())
        
        architecture_info = {
            'model/total_parameters': total_params,
            'model/trainable_parameters': trainable_params,
            'model/qat_layers': qat_layers,
            'model/total_layers': total_layers,
            'model/qat_percentage': 100.0 * qat_layers / max(total_layers, 1),
        }
        
        wandb.log(architecture_info)
        
        # Log model config
        wandb.config.update(config, allow_val_change=True)
        
        logger.info(f"✅ Logged model architecture: {qat_layers} QAT layers, {total_params:,} params")
        
    except Exception as e:
        logger.error(f"Error logging model architecture: {e}")


def log_training_summary(
    final_metrics: Dict[str, float],
    best_metrics: Dict[str, float],
    training_args: Any,
):
    """Log final training summary"""
    if not WANDB_AVAILABLE or wandb.run is None:
        return
    
    try:
        summary = {
            'summary/final_f1': final_metrics.get('f1', 0.0),
            'summary/final_em': final_metrics.get('exact_match', 0.0),
            'summary/best_f1': best_metrics.get('f1', 0.0),
            'summary/best_em': best_metrics.get('exact_match', 0.0),
            'summary/total_epochs': training_args.num_train_epochs,
        }
        
        wandb.log(summary)
        
        # Set summary metrics
        for key, value in summary.items():
            wandb.run.summary[key] = value
        
        logger.info("✅ Logged training summary to WandB")
        
    except Exception as e:
        logger.error(f"Error logging training summary: {e}")

