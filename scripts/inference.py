"""
Interactive Inference CLI for RL Music Recommendation
=====================================================

Beautiful CLI interface for testing trained models with real-time recommendations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich import box
import numpy as np
import pandas as pd
from typing import Optional, List
import time

from src.config.base_config import Config
from src.envs.music_env import MusicEnv
from src.agents.dqn_agent import DQNAgent
from src.agents.a2c_agent import A2CAgent
from src.agents.sac_agent import SACAgent

app = typer.Typer(help="🎵 RL Music Recommendation Inference CLI")
console = Console()


class MusicRecommender:
    """Interactive music recommender using trained RL agent"""
    
    def __init__(self, config_path: str, checkpoint_path: str):
        console.print("\n[bold cyan]Loading model...[/bold cyan]")
        
        # Load config
        self.config = Config.from_yaml(config_path)
        
        # Create environment
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing environment...", total=None)
            self.env = MusicEnv(
                data_path=self.config.env.data_path,
                mode=self.config.env.mode,
                **self.config.env.model_dump(exclude={'data_path', 'mode'})
            )
            progress.update(task, completed=True)
        
        # Get dimensions
        state_dim = self.env.observation_space.shape[0]
        if self.config.env.mode == "discrete":
            action_dim = self.env.action_space.n
        else:
            action_dim = self.env.action_space.shape[0]
        
        # Create and load agent
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading trained agent...", total=None)
            
            algorithm = self.config.training.algorithm
            if algorithm == "dqn":
                self.agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device="cpu")
            elif algorithm == "a2c":
                self.agent = A2CAgent(state_dim=state_dim, action_dim=action_dim, device="cpu")
            elif algorithm == "sac":
                self.agent = SACAgent(state_dim=state_dim, action_dim=action_dim, device="cpu")
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            self.agent.load(checkpoint_path)
            self.agent.eval_mode()
            progress.update(task, completed=True)
        
        # Load song metadata
        songs_path = Path(self.config.env.data_path) / "songs.csv"
        self.songs_df = pd.read_csv(songs_path)
        
        console.print("[bold green]✓ Model loaded successfully![/bold green]\n")
    
    def get_song_info(self, song_idx: int) -> dict:
        """Get song metadata"""
        if song_idx >= len(self.songs_df):
            return {"error": "Invalid song index"}
        
        song = self.songs_df.iloc[song_idx]
        
        # Format duration (already in seconds) to mm:ss
        duration_sec = song.get('duration', 0)
        duration_str = f"{int(duration_sec // 60)}:{int(duration_sec % 60):02d}"
        
        return {
            'index': song_idx,
            'track_id': song.get('track_id', 'Unknown'),
            'title': song.get('track_name', 'Unknown'),
            'artist': song.get('artist_name', 'Unknown'),
            'album': song.get('album_name', 'Unknown'),
            'genre': song.get('track_genre', 'Unknown'),
            'duration': duration_str,
            'duration_sec': duration_sec,
            'release_year': int(song.get('release_year', 0)) if pd.notna(song.get('release_year', 0)) else 'N/A',
            'popularity': song.get('popularity', 0),
            'danceability': song.get('danceability', 0),
            'energy': song.get('energy', 0),
            'valence': song.get('valence', 0)
        }
    
    def display_song(self, song_info: dict, title: str = "🎵 Recommended Song"):
        """Display song in a beautiful panel"""
        content = f"""
[bold cyan]Track ID:[/bold cyan] {song_info['track_id']}
[bold cyan]Title:[/bold cyan] {song_info['title']}
[bold cyan]Artist:[/bold cyan] {song_info['artist']}
[bold cyan]Album:[/bold cyan] {song_info['album']}
[bold cyan]Genre:[/bold cyan] {song_info['genre']}
[bold cyan]Duration:[/bold cyan] {song_info['duration']}
[bold cyan]Release Year:[/bold cyan] {song_info['release_year']}

[bold yellow]Audio Features:[/bold yellow]
  • Danceability: {song_info['danceability']:.2f}
  • Energy: {song_info['energy']:.2f}
  • Valence (Mood): {song_info['valence']:.2f}
  • Popularity: {song_info['popularity']:.0f}/100
        """
        
        panel = Panel(
            content.strip(),
            title=title,
            border_style="bright_blue",
            box=box.ROUNDED
        )
        console.print(panel)
    
    def interactive_session(self):
        """Run interactive recommendation session"""
        console.print(Panel.fit(
            "[bold cyan]🎵 Interactive Music Recommendation Session[/bold cyan]\n"
            "I'll recommend songs based on your reactions!\n"
            "React with: [green]love[/green], [yellow]like[/yellow], or [red]skip[/red]",
            border_style="cyan"
        ))
        
        state, info = self.env.reset()
        session_songs = []
        session_reactions = []
        episode_reward = 0
        
        console.print(f"\n[bold]User Profile:[/bold] {info['user_id']}")
        console.print(f"[bold]Session starting at hour:[/bold] {self.env.current_hour}:00\n")
        
        step = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            step += 1
            
            # Get recommendation
            if self.config.training.algorithm == "a2c":
                action, _, _ = self.agent.act(state, training=False)
            else:
                action = self.agent.act(state, training=False)
            
            # Convert continuous action to song index for display
            if self.config.env.mode == "continuous":
                song_idx = self.env._continuous_to_discrete(action)
            else:
                song_idx = int(action)
            
            # Get song info
            song_info = self.get_song_info(song_idx)
            
            # Display recommendation
            console.print(f"\n[bold cyan]╔══ Recommendation {step} ══╗[/bold cyan]")
            self.display_song(song_info)
            
            # Get user feedback
            reaction = Prompt.ask(
                "\n[bold]Your reaction?[/bold]",
                choices=["love", "like", "skip", "quit"],
                default="like"
            )
            
            if reaction == "quit":
                console.print("\n[yellow]Ending session...[/yellow]")
                break
            
            # Step environment (pass original action)
            state, reward, terminated, truncated, info = self.env.step(action)
            
            episode_reward += reward
            session_songs.append(song_info)
            session_reactions.append(reaction)
            
            # Show reward
            if reward > 0:
                console.print(f"[green]✓ Reward: +{reward:.2f}[/green]")
            else:
                console.print(f"[red]✗ Reward: {reward:.2f}[/red]")
            
            # Check if session should end
            if terminated or truncated:
                console.print("\n[yellow]Session ended by environment[/yellow]")
                break
            
            # Ask to continue
            if step % 5 == 0:
                if not Confirm.ask("\n[bold]Continue listening?[/bold]", default=True):
                    break
        
        # Display session summary
        self.display_session_summary(session_songs, session_reactions, episode_reward)
    
    def display_session_summary(self, songs: List[dict], reactions: List[str], total_reward: float):
        """Display beautiful session summary"""
        console.print("\n\n")
        console.print(Panel.fit(
            "[bold cyan]📊 Session Summary[/bold cyan]",
            border_style="cyan"
        ))
        
        # Create summary table
        table = Table(title="Listening History", box=box.ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=3)
        table.add_column("Title", style="cyan")
        table.add_column("Artist", style="green")
        table.add_column("Reaction", justify="center")
        
        reaction_colors = {"love": "green", "like": "yellow", "skip": "red"}
        reaction_icons = {"love": "❤️", "like": "👍", "skip": "⏭️"}
        
        for i, (song, reaction) in enumerate(zip(songs, reactions), 1):
            color = reaction_colors.get(reaction, "white")
            icon = reaction_icons.get(reaction, "•")
            table.add_row(
                str(i),
                song['title'][:40] + "..." if len(song['title']) > 40 else song['title'],
                song['artist'][:30] + "..." if len(song['artist']) > 30 else song['artist'],
                f"[{color}]{icon} {reaction.upper()}[/{color}]"
            )
        
        console.print(table)
        
        # Statistics
        stats_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        stats_table.add_column("Metric", style="bold")
        stats_table.add_column("Value", justify="right")
        
        total_songs = len(reactions)
        skip_count = reactions.count("skip")
        like_count = reactions.count("like")
        love_count = reactions.count("love")
        
        stats_table.add_row("Total Songs", str(total_songs))
        stats_table.add_row("Loved ❤️", f"[green]{love_count}[/green] ({love_count/total_songs*100:.1f}%)")
        stats_table.add_row("Liked 👍", f"[yellow]{like_count}[/yellow] ({like_count/total_songs*100:.1f}%)")
        stats_table.add_row("Skipped ⏭️", f"[red]{skip_count}[/red] ({skip_count/total_songs*100:.1f}%)")
        stats_table.add_row("Total Reward", f"[cyan]{total_reward:.2f}[/cyan]")
        
        console.print("\n")
        console.print(Panel(stats_table, title="📈 Statistics", border_style="green"))
        
        # Engagement score
        engagement = (love_count * 2 + like_count) / total_songs if total_songs > 0 else 0
        if engagement > 1.5:
            message = "[bold green]🎉 Excellent engagement! You loved this session![/bold green]"
        elif engagement > 1.0:
            message = "[bold yellow]😊 Good session! Mostly positive reactions.[/bold yellow]"
        else:
            message = "[bold red]😕 Low engagement. Let me learn your taste better![/bold red]"
        
        console.print(f"\n{message}\n")
    
    def quick_recommendations(self, n: int = 10):
        """Get quick recommendations without interaction"""
        console.print(Panel.fit(
            f"[bold cyan]🎵 Top {n} Recommendations[/bold cyan]",
            border_style="cyan"
        ))
        
        state, info = self.env.reset()
        console.print(f"\n[bold]User Profile:[/bold] {info['user_id']}\n")
        
        recommendations = []
        for i in range(n):
            if self.config.training.algorithm == "a2c":
                action, _, _ = self.agent.act(state, training=False)
            else:
                action = self.agent.act(state, training=False)
            
            # Convert continuous action to song index
            if self.config.env.mode == "continuous":
                song_idx = self.env._continuous_to_discrete(action)
            else:
                song_idx = int(action)
            
            song_info = self.get_song_info(song_idx)
            recommendations.append(song_info)
            
            # Step environment with neutral feedback
            state, _, terminated, truncated, _ = self.env.step(action)
            
            if terminated or truncated:
                break
        
        # Display as table with track_id, duration, release_year
        table = Table(title="Recommended Playlist", box=box.ROUNDED, show_header=True, header_style="bold cyan")
        table.add_column("#", style="dim", width=3)
        table.add_column("Track ID", style="magenta", width=20)
        table.add_column("Title", style="cyan", width=30)
        table.add_column("Artist", style="green", width=25)
        table.add_column("Duration", justify="center", style="yellow")
        table.add_column("Year", justify="center", style="blue")
        table.add_column("Energy", justify="center")
        table.add_column("Mood", justify="center")
        
        for i, song in enumerate(recommendations, 1):
            energy_bar = "█" * int(song['energy'] * 5)
            mood_bar = "█" * int(song['valence'] * 5)
            
            # Truncate track_id if too long
            track_id_display = song['track_id'][:18] + ".." if len(str(song['track_id'])) > 20 else song['track_id']
            
            table.add_row(
                str(i),
                str(track_id_display),
                song['title'][:28] + ".." if len(song['title']) > 30 else song['title'],
                song['artist'][:23] + ".." if len(song['artist']) > 25 else song['artist'],
                song['duration'],
                str(song['release_year']),
                f"{energy_bar} {song['energy']:.2f}",
                f"{mood_bar} {song['valence']:.2f}"
            )
        
        console.print(table)
        console.print("\n[dim]Use --interactive for full session mode[/dim]\n")


@app.command()
def recommend(
    config: str = typer.Option(..., "--config", "-c", help="Path to config YAML"),
    checkpoint: str = typer.Option(..., "--checkpoint", "-m", help="Path to model checkpoint"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive session mode"),
    num_songs: int = typer.Option(10, "--num", "-n", help="Number of recommendations (non-interactive)")
):
    """
    🎵 Get music recommendations from trained RL agent
    
    Examples:
        # Quick recommendations
        python inference.py recommend -c config/dqn_config.yaml -m checkpoints/best_model.agent.pt
        
        # Interactive session
        python inference.py recommend -c config/dqn_config.yaml -m checkpoints/best_model.agent.pt --interactive
    """
    try:
        recommender = MusicRecommender(config, checkpoint)
        
        if interactive:
            recommender.interactive_session()
        else:
            recommender.quick_recommendations(num_songs)
            
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def evaluate(
    config: str = typer.Option(..., "--config", "-c", help="Path to config YAML"),
    checkpoint: str = typer.Option(..., "--checkpoint", "-m", help="Path to model checkpoint"),
    num_episodes: int = typer.Option(100, "--episodes", "-e", help="Number of evaluation episodes")
):
    """
    📊 Evaluate trained model performance
    
    Example:
        python inference.py evaluate -c config/dqn_config.yaml -m checkpoints/best_model.agent.pt -e 100
    """
    try:
        recommender = MusicRecommender(config, checkpoint)
        
        console.print(f"\n[bold cyan]Running evaluation for {num_episodes} episodes...[/bold cyan]\n")
        
        results = []
        with Progress(console=console) as progress:
            task = progress.add_task("[cyan]Evaluating...", total=num_episodes)
            
            for episode in range(num_episodes):
                state, _ = recommender.env.reset()
                episode_reward = 0
                episode_length = 0
                reactions = []
                
                terminated = False
                truncated = False
                
                while not (terminated or truncated):
                    if recommender.config.training.algorithm == "a2c":
                        action, _, _ = recommender.agent.act(state, training=False)
                    else:
                        action = recommender.agent.act(state, training=False)
                    
                    state, reward, terminated, truncated, info = recommender.env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    reactions.append(info['user_reaction'])
                
                skip_rate = reactions.count(0) / len(reactions) if reactions else 0
                complete_rate = reactions.count(2) / len(reactions) if reactions else 0
                
                results.append({
                    'reward': episode_reward,
                    'length': episode_length,
                    'skip_rate': skip_rate,
                    'complete_rate': complete_rate
                })
                
                progress.update(task, advance=1)
        
        # Display results
        df = pd.DataFrame(results)
        
        summary_table = Table(title="📊 Evaluation Results", box=box.ROUNDED, show_header=True, header_style="bold cyan")
        summary_table.add_column("Metric", style="bold")
        summary_table.add_column("Mean", justify="right")
        summary_table.add_column("Std", justify="right")
        summary_table.add_column("Min", justify="right")
        summary_table.add_column("Max", justify="right")
        
        summary_table.add_row(
            "Reward",
            f"{df['reward'].mean():.2f}",
            f"{df['reward'].std():.2f}",
            f"{df['reward'].min():.2f}",
            f"{df['reward'].max():.2f}"
        )
        summary_table.add_row(
            "Episode Length",
            f"{df['length'].mean():.1f}",
            f"{df['length'].std():.1f}",
            f"{df['length'].min():.0f}",
            f"{df['length'].max():.0f}"
        )
        summary_table.add_row(
            "Skip Rate",
            f"{df['skip_rate'].mean():.1%}",
            f"{df['skip_rate'].std():.1%}",
            f"{df['skip_rate'].min():.1%}",
            f"{df['skip_rate'].max():.1%}"
        )
        summary_table.add_row(
            "Complete Rate",
            f"{df['complete_rate'].mean():.1%}",
            f"{df['complete_rate'].std():.1%}",
            f"{df['complete_rate'].min():.1%}",
            f"{df['complete_rate'].max():.1%}"
        )
        
        console.print("\n")
        console.print(summary_table)
        console.print("\n")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def info(
    config: str = typer.Option(..., "--config", "-c", help="Path to config YAML")
):
    """
    ℹ️  Display model and environment information
    
    Example:
        python inference.py info -c config/dqn_config.yaml
    """
    try:
        config_obj = Config.from_yaml(config)
        
        # Model info
        model_table = Table(title="🤖 Model Configuration", box=box.ROUNDED, show_header=False)
        model_table.add_column("Property", style="bold cyan")
        model_table.add_column("Value", style="yellow")
        
        model_table.add_row("Algorithm", config_obj.training.algorithm.upper())
        model_table.add_row("Action Space", config_obj.env.mode.capitalize())
        model_table.add_row("Device", config_obj.training.device.upper())
        
        console.print(model_table)
        
        # Environment info
        env_table = Table(title="🎵 Environment Configuration", box=box.ROUNDED, show_header=False)
        env_table.add_column("Property", style="bold cyan")
        env_table.add_column("Value", style="yellow")
        
        env_table.add_row("Data Path", config_obj.env.data_path)
        env_table.add_row("Max Session Length", str(config_obj.env.max_session_length))
        env_table.add_row("User Embedding Dim", str(config_obj.env.user_embed_dim))
        env_table.add_row("Song Embedding Dim", str(config_obj.env.song_embed_dim))
        
        console.print(env_table)
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()