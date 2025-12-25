"""Enhanced Visualization utilities for RL Music Recommendation System - Fixed JSON serialization"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np
from loguru import logger
import torch
from typing import Dict, List, Optional

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]

class RLVisualizer:
    """Comprehensive visualization for RL training results - Works with training_summary.json"""
    
    def __init__(self):
        self.colors = sns.color_palette("husl", 8)
    
    def plot_from_summary(self, summary_path: str, output_dir: str):
        """Create visualizations from training_summary.json"""
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)
        except FileNotFoundError:
            logger.error(f"Summary file not found: {summary_path}")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Final Evaluation Performance
        if 'final_eval_rewards' in summary:
            eval_rewards = summary['final_eval_rewards']
            episodes = range(len(eval_rewards))
            
            ax1.bar(episodes, eval_rewards, alpha=0.7, color=self.colors[0], 
                   label=f'Last {len(eval_rewards)} evaluations')
            ax1.axhline(y=summary.get('best_eval_reward', 0), color='red', 
                       linestyle='--', linewidth=2, label=f'Best: {summary.get("best_eval_reward", 0):.2f}')
            ax1.axhline(y=np.mean(eval_rewards), color='green', 
                       linestyle='--', linewidth=2, label=f'Mean: {np.mean(eval_rewards):.2f}')
            
            ax1.set_xlabel('Evaluation Run')
            ax1.set_ylabel('Reward')
            ax1.set_title('Final Evaluation Performance')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Training Statistics
        stats_data = {
            'Total Episodes': summary.get('total_episodes', 0),
            'Total Steps': summary.get('total_steps', 0) / 1000,  # Show in thousands
            'Training Hours': summary.get('elapsed_time_hours', 0)
        }
        
        bars = ax2.bar(range(len(stats_data)), list(stats_data.values()), 
                      color=[self.colors[1], self.colors[2], self.colors[3]], alpha=0.7)
        ax2.set_xticks(range(len(stats_data)))
        ax2.set_xticklabels(list(stats_data.keys()), rotation=45)
        ax2.set_title('Training Statistics')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, stats_data.values()):
            height = bar.get_height()
            label = f'{value:,.1f}' if value < 1000 else f'{value:,.0f}'
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    label, ha='center', va='bottom')
        
        # 3. Reward Analysis
        reward_metrics = {
            'Best Evaluation': summary.get('best_eval_reward', 0),
            'Mean Final Eval': np.mean(summary.get('final_eval_rewards', [0])),
            'Std Final Eval': np.std(summary.get('final_eval_rewards', [0])),
            'Mean Episode': summary.get('mean_episode_reward', 0)
        }
        
        bars = ax3.bar(range(len(reward_metrics)), list(reward_metrics.values()), 
                      color=self.colors[4], alpha=0.7)
        ax3.set_xticks(range(len(reward_metrics)))
        ax3.set_xticklabels(list(reward_metrics.keys()), rotation=45)
        ax3.set_ylabel('Reward')
        ax3.set_title('Reward Performance Metrics')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, reward_metrics.values()):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 4. Configuration and Summary
        config = summary.get('config', {})
        training_config = config.get('training', {})
        env_config = config.get('env', {})
        
        info_text = f"""
Algorithm: {training_config.get('algorithm', 'N/A').upper()}
Total Episodes: {summary.get('total_episodes', 0):,}
Total Steps: {summary.get('total_steps', 0):,}
Training Time: {summary.get('elapsed_time_hours', 0):.2f} hours

--- Performance ---
Best Eval Reward: {summary.get('best_eval_reward', 0):.2f}
Mean Episode Reward: {summary.get('mean_episode_reward', 0):.2f}
Final Eval Range: {min(summary.get('final_eval_rewards', [0])):.2f} - {max(summary.get('final_eval_rewards', [0])):.2f}

--- Environment ---
Max Session: {env_config.get('max_session_length', 'N/A')}
Users: {env_config.get('user_embed_dim', 'N/A')}D embeddings
Songs: {env_config.get('song_embed_dim', 'N/A')}D embeddings
"""
        
        ax4.text(0.02, 0.98, info_text, transform=ax4.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace', linespacing=1.5)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Training Configuration & Summary', fontsize=12, pad=20)
        
        # Add a border around the info box
        ax4.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=1))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create additional detailed plots
        self._create_detailed_plots(summary, output_dir)
        
        logger.success(f"Saved training dashboard to {output_dir / 'training_dashboard.png'}")
    
    def _create_detailed_plots(self, summary: Dict, output_dir: Path):
        """Create additional detailed visualizations"""
        
        # 1. Reward distribution analysis
        plt.figure(figsize=(10, 6))
        
        eval_rewards = summary.get('final_eval_rewards', [])
        if eval_rewards:
            # Create histogram of evaluation rewards
            plt.hist(eval_rewards, bins=15, alpha=0.7, color=self.colors[0], edgecolor='black')
            plt.axvline(np.mean(eval_rewards), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(eval_rewards):.2f}')
            plt.axvline(summary.get('best_eval_reward', 0), color='green', linestyle='--',
                       label=f'Best: {summary.get("best_eval_reward", 0):.2f}')
            plt.xlabel('Evaluation Reward')
            plt.ylabel('Frequency')
            plt.title('Final Evaluation Reward Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(output_dir / 'reward_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Training efficiency plot
        plt.figure(figsize=(10, 6))
        
        total_episodes = summary.get('total_episodes', 0)
        training_hours = summary.get('elapsed_time_hours', 0)
        total_steps = summary.get('total_steps', 0)
        
        efficiency_metrics = {
            'Episodes/Hour': total_episodes / training_hours if training_hours > 0 else 0,
            'Steps/Hour': total_steps / training_hours if training_hours > 0 else 0,
            'Steps/Episode': total_steps / total_episodes if total_episodes > 0 else 0
        }
        
        bars = plt.bar(range(len(efficiency_metrics)), list(efficiency_metrics.values()),
                      color=[self.colors[1], self.colors[2], self.colors[3]], alpha=0.7)
        plt.xticks(range(len(efficiency_metrics)), list(efficiency_metrics.keys()))
        plt.ylabel('Count')
        plt.title('Training Efficiency Metrics')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, efficiency_metrics.values()):
            height = bar.get_height()
            label = f'{value:,.0f}' if value > 10 else f'{value:.1f}'
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    label, ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Create a summary report (FIXED JSON serialization)
        self._create_summary_report(summary, output_dir)
    
    def _create_summary_report(self, summary: Dict, output_dir: Path):
        """Create a text summary report - FIXED JSON serialization"""
        # Convert all values to JSON-serializable types
        report = {
            'training_overview': {
                'algorithm': str(summary.get('config', {}).get('training', {}).get('algorithm', 'unknown')),
                'total_episodes': int(summary.get('total_episodes', 0)),
                'total_steps': int(summary.get('total_steps', 0)),
                'training_duration_hours': float(summary.get('elapsed_time_hours', 0)),
                'efficiency_episodes_per_hour': float(summary.get('total_episodes', 0) / summary.get('elapsed_time_hours', 1))
            },
            'performance': {
                'best_evaluation_reward': float(summary.get('best_eval_reward', 0)),
                'mean_episode_reward': float(summary.get('mean_episode_reward', 0)),
                'final_evaluation_stats': {
                    'mean': float(np.mean(summary.get('final_eval_rewards', [0]))),
                    'std': float(np.std(summary.get('final_eval_rewards', [0]))),
                    'min': float(min(summary.get('final_eval_rewards', [0]))),
                    'max': float(max(summary.get('final_eval_rewards', [0])))
                }
            },
            'success_indicators': {
                'training_completed': bool(summary.get('total_episodes', 0) > 0),
                'positive_rewards': bool(summary.get('mean_episode_reward', 0) > 0),
                'stable_performance': bool(np.std(summary.get('final_eval_rewards', [0])) < 5.0)  # Low variance
            }
        }
        
        # Save JSON report (now with proper serializable types)
        try:
            with open(output_dir / 'detailed_report.json', 'w') as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save JSON report: {e}")
            # Save as string representation instead
            with open(output_dir / 'detailed_report.txt', 'w') as f:
                for category, data in report.items():
                    f.write(f"{category.upper()}:\n")
                    for key, value in data.items():
                        if isinstance(value, dict):
                            f.write(f"  {key}:\n")
                            for subkey, subvalue in value.items():
                                f.write(f"    {subkey}: {subvalue}\n")
                        else:
                            f.write(f"  {key}: {value}\n")
                    f.write("\n")
        
        # Save human-readable report
        try:
            with open(output_dir / 'training_report.txt', 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("RL MUSIC RECOMMENDATION TRAINING REPORT\n")
                f.write("=" * 60 + "\n\n")
                
                f.write("OVERVIEW:\n")
                f.write(f"  Algorithm:          {report['training_overview']['algorithm'].upper()}\n")
                f.write(f"  Total Episodes:     {report['training_overview']['total_episodes']:,}\n")
                f.write(f"  Total Steps:        {report['training_overview']['total_steps']:,}\n")
                f.write(f"  Training Time:      {report['training_overview']['training_duration_hours']:.2f} hours\n")
                f.write(f"  Efficiency:         {report['training_overview']['efficiency_episodes_per_hour']:.1f} episodes/hour\n\n")
                
                f.write("PERFORMANCE:\n")
                f.write(f"  Best Eval Reward:   {report['performance']['best_evaluation_reward']:.2f}\n")
                f.write(f"  Mean Episode Reward: {report['performance']['mean_episode_reward']:.2f}\n")
                f.write(f"  Final Eval Mean:    {report['performance']['final_evaluation_stats']['mean']:.2f}\n")
                f.write(f"  Final Eval Std:     {report['performance']['final_evaluation_stats']['std']:.2f}\n")
                f.write(f"  Final Eval Range:   {report['performance']['final_evaluation_stats']['min']:.2f} - {report['performance']['final_evaluation_stats']['max']:.2f}\n\n")
                
                f.write("SUCCESS INDICATORS:\n")
                indicators = report['success_indicators']
                f.write(f"  Training Completed: {'✅' if indicators['training_completed'] else '❌'}\n")
                f.write(f"  Positive Rewards:   {'✅' if indicators['positive_rewards'] else '❌'}\n")
                f.write(f"  Stable Performance: {'✅' if indicators['stable_performance'] else '❌'}\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("CONCLUSION: ")
                if all(indicators.values()):
                    f.write("EXCELLENT TRAINING RESULTS! 🎉\n")
                elif indicators['training_completed'] and indicators['positive_rewards']:
                    f.write("GOOD TRAINING RESULTS! 👍\n")
                else:
                    f.write("TRAINING COMPLETED - REVIEW RESULTS 🔍\n")
                f.write("=" * 60 + "\n")
        except Exception as e:
            logger.warning(f"Could not save text report: {e}")
        
        logger.info(f"Saved reports to {output_dir}")

    def plot_training_curves(self, metrics_path: str, output_dir: str, smooth_window: int = 100):
        """Legacy function - now uses summary file"""
        logger.warning("metrics.json not found, using training_summary.json instead")
        self.plot_from_summary(metrics_path, output_dir)


# Standalone function for easy use
def visualize_training_summary(summary_path: str, output_dir: str = "plots"):
    """Quick function to visualize training summary"""
    visualizer = RLVisualizer()
    visualizer.plot_from_summary(summary_path, output_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RL Training Visualizer - Works with training_summary.json")
    parser.add_argument("--summary", required=True, help="Path to training_summary.json")
    parser.add_argument("--output", default="plots", help="Output directory")
    
    args = parser.parse_args()
    
    visualizer = RLVisualizer()
    visualizer.plot_from_summary(args.summary, args.output)