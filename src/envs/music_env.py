import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pickle
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import yaml


class MusicEnv(gym.Env):
    """
    Music Recommendation Environment
    
    This environment simulates a music streaming session where an RL agent
    recommends songs to users. The agent learns from user reactions (skip, 
    partial listen, complete listen) to optimize long-term engagement.
    
    Action Space:
        - Discrete mode: Integer in [0, num_songs) representing song index
        - Continuous mode: Float vector of shape (embedding_dim,) representing
          song embedding. Environment finds nearest neighbor song.
    
    Observation Space:
        Float vector containing:
        - User embedding (user_embed_dim)
        - Recent track embeddings (num_recent_tracks * song_embed_dim)
        - Context features (hour_of_day, session_position, etc.)
    
    Reward:
        Multi-component reward based on:
        - Listening completion (+1.0 full, +0.5 partial, -0.5 skip)
        - Diversity bonus (avoid repetition)
        - Personalization bonus (match user preferences)
        - Session engagement (longer sessions = bonus)
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self,
        data_path: str = "data/raw",
        config_path: Optional[str] = None,
        mode: str = "discrete",
        **kwargs
    ):
        """
        Initialize the Music Recommendation Environment
        
        Args:
            data_path: Path to directory containing users.csv, songs.csv, interactions.csv
            config_path: Path to YAML config file (optional, uses defaults if None)
            mode: "discrete" or "continuous" - determines action space type
            **kwargs: Override config parameters
        """
        super().__init__()
        
        self.data_path = Path(data_path)
        self.mode = mode.lower()
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.config.update(kwargs)  # Override with kwargs
        
        # Initialize logging
        self._setup_logging()
        
        logger.info(f"Initializing MusicEnv in {self.mode} mode")
        logger.info(f"Data path: {self.data_path}")
        
        # Load datasets
        self._load_data()
        
        # Build or load embeddings
        self._load_embeddings()
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Initialize episode state
        self.current_user_idx = None
        self.current_user_id = None
        self.user_history = []
        self.session_position = 0
        self.episode_rewards = []
        self.episode_songs = []
        
        logger.success("MusicEnv initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML or use defaults"""
        default_config = {
            # Environment settings
            'max_session_length': 20,
            'num_recent_tracks': 5,
            'user_embed_dim': 32,
            'song_embed_dim': 8,  # From acoustic_vector_0 to 7
            
            # Reward weights
            'reward_full_listen': 1.0,
            'reward_partial_listen': 0.5,
            'reward_skip': -0.5,
            'reward_diversity_weight': 0.2,
            'reward_personalization_weight': 0.3,
            'reward_session_bonus_weight': 0.1,
            
            # User behavior simulation
            'skip_threshold': 0.3,  # Similarity threshold below which user likely skips
            'partial_threshold': 0.6,  # Threshold for partial vs full listen
            'dropout_probability': 0.05,  # Per-step chance user ends session
            
            # Diversity calculation
            'diversity_window': 10,  # Check last N songs for diversity
            'diversity_threshold': 0.8,  # Cosine similarity threshold
            
            # Data processing
            'normalize_embeddings': True,
            'use_context_features': True,
            
            # Logging
            'log_level': 'INFO',
            'log_file': 'logs/music_env.log',
            
            # Observation format
            'obs_format': 'flat',  # 'flat' or 'sequential'
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                default_config.update(loaded_config)
                logger.info(f"Loaded config from {config_path}")
        
        return default_config
    
    def _setup_logging(self):
        """Configure loguru logger"""
        log_level = self.config.get('log_level', 'INFO')
        log_file = self.config.get('log_file', 'logs/music_env.log')
        
        # Create logs directory if needed
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logger
        logger.remove()  # Remove default handler
        logger.add(
            log_file,
            rotation="10 MB",
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
        )
        logger.add(
            lambda msg: print(msg, end=''),
            level=log_level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}"
        )
    
    def _load_data(self):
        """Load user, song, and interaction datasets"""
        logger.info("Loading datasets...")
        
        # Load songs (with acoustic features)
        songs_path = self.data_path / "songs.csv"
        if not songs_path.exists():
            raise FileNotFoundError(f"Songs file not found: {songs_path}")
        
        self.songs_df = pd.read_csv(songs_path)
        logger.info(f"Loaded {len(self.songs_df)} songs")
        
        # Extract acoustic vectors (acoustic_vector_0 to 7)
        acoustic_cols = [f'acoustic_vector_{i}' for i in range(8)]
        if all(col in self.songs_df.columns for col in acoustic_cols):
            self.song_embeddings = self.songs_df[acoustic_cols].values
            logger.info(f"Extracted song embeddings: {self.song_embeddings.shape}")
        else:
            logger.warning("Acoustic vectors not found, will generate random embeddings")
            self.song_embeddings = np.random.randn(
                len(self.songs_df), 
                self.config['song_embed_dim']
            )
        
        # Load interactions
        interactions_path = self.data_path / "interactions.csv"
        if interactions_path.exists():
            self.interactions_df = pd.read_csv(interactions_path, nrows=100000)  # Limit for memory
            logger.info(f"Loaded {len(self.interactions_df)} interactions")
            
            # Build user interaction history
            self._build_user_history()
        else:
            logger.warning("Interactions file not found, will use synthetic user behavior")
            self.interactions_df = None
            self.user_histories = {}
        
        # Create song lookup
        self.track_id_to_idx = {
            track_id: idx 
            for idx, track_id in enumerate(self.songs_df['track_id'])
        }
        self.idx_to_track_id = {v: k for k, v in self.track_id_to_idx.items()}
        
        # Extract metadata for state representation
        self._extract_song_metadata()
    
    def _build_user_history(self):
        """Build user listening history from interactions"""
        logger.info("Building user interaction histories...")
        
        # Group by session_id (which represents a user session)
        self.user_histories = {}
        
        for session_id, session_data in self.interactions_df.groupby('session_id'):
            # Determine listening outcome for each track
            history = []
            for _, row in session_data.iterrows():
                track_id = row['track_id_clean']
                
                # Skip if track not in our catalog
                if track_id not in self.track_id_to_idx:
                    continue
                
                # Determine outcome: 0=skip, 1=partial, 2=full
                if row['not_skipped']:
                    outcome = 2  # Full listen
                elif row['skip_1']:
                    outcome = 0  # Early skip
                elif row['skip_2'] or row['skip_3']:
                    outcome = 1  # Partial listen
                else:
                    outcome = 2  # Default to full
                
                history.append({
                    'track_id': track_id,
                    'track_idx': self.track_id_to_idx[track_id],
                    'outcome': outcome,
                    'hour_of_day': row.get('hour_of_day', 12),
                    'session_position': row.get('session_position', 0)
                })
            
            if history:
                self.user_histories[session_id] = history
        
        self.user_ids = list(self.user_histories.keys())
        logger.info(f"Built histories for {len(self.user_ids)} users")
    
    def _extract_song_metadata(self):
        """Extract song metadata features for state representation"""
        # Extract key song features
        feature_cols = [
            'danceability', 'energy', 'valence', 'tempo', 
            'acousticness', 'instrumentalness'
        ]
        
        available_cols = [col for col in feature_cols if col in self.songs_df.columns]
        
        if available_cols:
            self.song_features = self.songs_df[available_cols].fillna(0).values
            logger.info(f"Extracted {len(available_cols)} song features")
        else:
            # Use embeddings as features if metadata not available
            self.song_features = self.song_embeddings
            logger.warning("Using embeddings as song features")
        
        # Normalize features
        if self.config['normalize_embeddings']:
            scaler = StandardScaler()
            self.song_features = scaler.fit_transform(self.song_features)
            logger.info("Normalized song features")
    
    def _load_embeddings(self):
        """Load or generate user and song embeddings"""
        embeddings_path = self.data_path / "embeddings.pkl"
        
        if embeddings_path.exists():
            logger.info("Loading precomputed embeddings...")
            with open(embeddings_path, 'rb') as f:
                embeddings = pickle.load(f)
                self.user_embeddings = embeddings['users']
                # Song embeddings already loaded from CSV
            logger.info(f"Loaded user embeddings: {self.user_embeddings.shape}")
        else:
            logger.info("Generating user embeddings from listening history...")
            self._generate_user_embeddings()
        
        # Normalize embeddings if configured
        if self.config['normalize_embeddings']:
            self.user_embeddings = self._normalize(self.user_embeddings)
            self.song_embeddings = self._normalize(self.song_embeddings)
            logger.info("Normalized embeddings")
    
    def _generate_user_embeddings(self):
        """Generate user embeddings from their listening history"""
        user_embed_dim = self.config['user_embed_dim']
        
        if self.user_histories:
            # Average embeddings of songs user listened to
            num_users = len(self.user_ids)
            self.user_embeddings = np.zeros((num_users, user_embed_dim))
            
            for user_idx, user_id in enumerate(self.user_ids):
                history = self.user_histories[user_id]
                # Get embeddings of listened tracks
                track_indices = [h['track_idx'] for h in history if h['outcome'] >= 1]
                if track_indices:
                    # Average song embeddings
                    avg_embedding = self.song_embeddings[track_indices].mean(axis=0)
                    # Pad or truncate to user_embed_dim
                    if len(avg_embedding) < user_embed_dim:
                        self.user_embeddings[user_idx, :len(avg_embedding)] = avg_embedding
                    else:
                        self.user_embeddings[user_idx] = avg_embedding[:user_embed_dim]
                else:
                    # Random embedding for users with no valid history
                    self.user_embeddings[user_idx] = np.random.randn(user_embed_dim) * 0.1
        else:
            # Generate random user embeddings
            num_users = 1000  # Default number of synthetic users
            self.user_embeddings = np.random.randn(num_users, user_embed_dim)
            self.user_ids = [f"user_{i}" for i in range(num_users)]
            logger.warning(f"Generated {num_users} synthetic users")
    
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms
    
    def _setup_spaces(self):
        """Define action and observation spaces"""
        num_songs = len(self.songs_df)
        
        # Action Space
        if self.mode == "discrete":
            self.action_space = spaces.Discrete(num_songs)
            logger.info(f"Discrete action space: {num_songs} songs")
        elif self.mode == "continuous":
            self.action_space = spaces.Box(
                low=-1.0, 
                high=1.0, 
                shape=(self.config['song_embed_dim'],),
                dtype=np.float32
            )
            logger.info(f"Continuous action space: {self.config['song_embed_dim']}-dim")
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Use 'discrete' or 'continuous'")
        
        # Observation Space
        if self.config.get('obs_format') == 'sequential':
            # Format: (seq_len, feature_dim)
            # feature_dim = song_embed (8) + user_embed (32) + context (4) = 44
            
            seq_len = self.config['num_recent_tracks']
            feature_dim = self.config['song_embed_dim'] + self.config['user_embed_dim']
            
            if self.config['use_context_features']:
                feature_dim += 4
                
            self.observation_space = spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(seq_len, feature_dim), 
                dtype=np.float32
            )
            logger.info(f"Observation space: ({seq_len}, {feature_dim}) sequential")
            
        else:
            # Flat format
            obs_dim = self._calculate_obs_dim()
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32
            )
            logger.info(f"Observation space: {obs_dim}-dim")
    
    def _calculate_obs_dim(self) -> int:
        """Calculate total observation dimension (for flat mode)"""
        dim = 0
        
        # User embedding
        dim += self.config['user_embed_dim']
        
        # Recent tracks (each track has song_embed_dim features)
        dim += self.config['num_recent_tracks'] * self.config['song_embed_dim']
        
        # Context features
        if self.config['use_context_features']:
            dim += 4  # hour_of_day (normalized), session_position, session_progress, time_since_last
        
        return dim
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment for new episode
        
        Returns:
            observation: Initial state
            info: Dictionary with episode metadata
        """
        super().reset(seed=seed)
        
        # Select random user
        self.current_user_idx = self.np_random.integers(0, len(self.user_embeddings))
        self.current_user_id = self.user_ids[self.current_user_idx]
        
        # Initialize episode state
        self.user_history = []
        self.session_position = 0
        self.episode_rewards = []
        self.episode_songs = []
        self.current_hour = self.np_random.integers(0, 24)
        self.last_action_time = 0
        
        # Get initial observation
        obs = self._get_observation()
        
        info = {
            'user_id': self.current_user_id,
            'user_idx': self.current_user_idx,
            'session_position': self.session_position
        }
        
        logger.debug(f"Episode reset | User: {self.current_user_id} | Hour: {self.current_hour}")
        
        return obs, info
    
    def _get_observation(self) -> np.ndarray:
        """Build current state observation vector"""
        
        # 1. User embedding
        user_embed = self.user_embeddings[self.current_user_idx]
        
        # 2. Recent tracks embeddings
        num_recent = self.config['num_recent_tracks']
        song_embed_dim = self.config['song_embed_dim']
        
        if len(self.user_history) > 0:
            recent_indices = [h['track_idx'] for h in self.user_history[-num_recent:]]
            recent_embeds = self.song_embeddings[recent_indices]
            
            # Pad if needed
            if len(recent_embeds) < num_recent:
                padding = np.zeros((num_recent - len(recent_embeds), song_embed_dim))
                recent_embeds = np.vstack([padding, recent_embeds])
        else:
            recent_embeds = np.zeros((num_recent, song_embed_dim))
            
        # 3. Context features
        if self.config['use_context_features']:
            max_session = self.config['max_session_length']
            context = np.array([
                self.current_hour / 24.0,  # Normalized hour
                self.session_position / max_session,  # Session progress
                min(self.session_position, max_session) / max_session,  # Capped progress
                min(self.session_position - self.last_action_time, 10) / 10.0  # Time since last
            ])
        else:
            context = np.array([])
            
        # Format output
        if self.config.get('obs_format') == 'sequential':
            # Create a sequence of [song_embed, user_embed, context] for each timestep
            # Shape: (num_recent, features)
            
            # Repeat user and context for each sequence step
            user_expanded = np.tile(user_embed, (num_recent, 1))
            
            if len(context) > 0:
                context_expanded = np.tile(context, (num_recent, 1))
                # Stack: (num_recent, song+user+context)
                observation = np.hstack([recent_embeds, user_expanded, context_expanded])
            else:
                observation = np.hstack([recent_embeds, user_expanded])
                
            return observation.astype(np.float32)
            
        else:
            # Flat format
            obs_parts = [user_embed, recent_embeds.flatten()]
            if len(context) > 0:
                obs_parts.append(context)
            
            observation = np.concatenate(obs_parts).astype(np.float32)
            return observation
    
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Song index (discrete) or embedding vector (continuous)
        
        Returns:
            observation: Next state
            reward: Reward for this step
            terminated: Whether episode ended naturally
            truncated: Whether episode was cut off
            info: Additional information
        """
        # Convert action to song index
        if self.mode == "continuous":
            song_idx = self._continuous_to_discrete(action)
        else:
            song_idx = int(action)
        
        # Validate action
        if song_idx < 0 or song_idx >= len(self.songs_df):
            logger.warning(f"Invalid song index: {song_idx}, clipping")
            song_idx = np.clip(song_idx, 0, len(self.songs_df) - 1)
        
        track_id = self.idx_to_track_id[song_idx]
        
        # Simulate user reaction
        user_reaction = self._simulate_user_reaction(song_idx)
        
        # Calculate reward
        reward = self._calculate_reward(song_idx, user_reaction)
        
        # Update state
        self.user_history.append({
            'track_idx': song_idx,
            'track_id': track_id,
            'outcome': user_reaction,
            'session_position': self.session_position
        })
        self.episode_songs.append(song_idx)
        self.episode_rewards.append(reward)
        self.last_action_time = self.session_position
        self.session_position += 1
        
        # Check termination conditions
        terminated = self._check_termination(user_reaction)
        truncated = self.session_position >= self.config['max_session_length']
        
        # Get next observation
        obs = self._get_observation()
        
        # Build info dict
        info = {
            'song_idx': song_idx,
            'track_id': track_id,
            'user_reaction': user_reaction,
            'reward_components': self._get_reward_components(song_idx, user_reaction),
            'session_position': self.session_position,
            'episode_length': len(self.episode_songs)
        }
        
        # Log step
        reaction_str = ['skip', 'partial', 'full'][user_reaction]
        logger.debug(
            f"Step {self.session_position} | Song: {song_idx} | "
            f"Reaction: {reaction_str} | Reward: {reward:.3f}"
        )
        
        # Log episode summary if done
        if terminated or truncated:
            self._log_episode_summary(terminated, truncated)
        
        return obs, reward, terminated, truncated, info
    
    def _continuous_to_discrete(self, action_embedding: np.ndarray) -> int:
        """
        Convert continuous action (embedding) to discrete song index
        using nearest neighbor search
        """
        action_embedding = np.array(action_embedding).reshape(1, -1)
        
        # Compute cosine similarity with all song embeddings
        similarities = cosine_similarity(action_embedding, self.song_embeddings)[0]
        
        # Return index of most similar song
        song_idx = int(np.argmax(similarities))
        
        return song_idx
    
    def _simulate_user_reaction(self, song_idx: int) -> int:
        """
        Simulate user reaction to recommended song
        
        Returns:
            0: Skip
            1: Partial listen
            2: Full listen
        """
        # Calculate similarity between song and user preference
        user_embed = self.user_embeddings[self.current_user_idx]
        song_embed = self.song_embeddings[song_idx]
        
        # Pad embeddings to same size for comparison
        min_dim = min(len(user_embed), len(song_embed))
        user_embed_trunc = user_embed[:min_dim]
        song_embed_trunc = song_embed[:min_dim]
        
        # Cosine similarity
        similarity = np.dot(user_embed_trunc, song_embed_trunc) / (
            np.linalg.norm(user_embed_trunc) * np.linalg.norm(song_embed_trunc) + 1e-8
        )
        
        # Adjust similarity based on recent history (avoid repetition)
        if len(self.user_history) > 0:
            recent_songs = [h['track_idx'] for h in self.user_history[-5:]]
            if song_idx in recent_songs:
                similarity *= 0.5  # Penalize recently played songs
        
        # Use similarity to determine outcome probabilistically
        skip_threshold = self.config['skip_threshold']
        partial_threshold = self.config['partial_threshold']
        
        # Add some randomness
        noise = self.np_random.normal(0, 0.1)
        similarity_noisy = similarity + noise
        
        if similarity_noisy < skip_threshold:
            return 0  # Skip
        elif similarity_noisy < partial_threshold:
            return 1  # Partial
        else:
            return 2  # Full listen
    
    def _calculate_reward(self, song_idx: int, user_reaction: int) -> float:
        """
        Calculate multi-component reward
        
        Components:
        1. Listening completion reward
        2. Diversity bonus
        3. Personalization bonus
        4. Session engagement bonus
        """
        reward = 0.0
        
        # 1. Base reward from listening outcome
        if user_reaction == 0:  # Skip
            reward += self.config['reward_skip']
        elif user_reaction == 1:  # Partial
            reward += self.config['reward_partial_listen']
        else:  # Full listen
            reward += self.config['reward_full_listen']
        
        # 2. Diversity bonus (avoid repetition)
        diversity_bonus = self._calculate_diversity_bonus(song_idx)
        reward += diversity_bonus * self.config['reward_diversity_weight']
        
        # 3. Personalization bonus (match user preferences)
        personalization_bonus = self._calculate_personalization_bonus(song_idx)
        reward += personalization_bonus * self.config['reward_personalization_weight']
        
        # 4. Session engagement bonus (longer sessions)
        if self.session_position > 5:  # After initial songs
            engagement_bonus = min(self.session_position / 20.0, 1.0)
            reward += engagement_bonus * self.config['reward_session_bonus_weight']
        
        return float(reward)
    
    def _calculate_diversity_bonus(self, song_idx: int) -> float:
        """Calculate diversity bonus based on recent history"""
        if len(self.episode_songs) == 0:
            return 0.0
        
        window = self.config['diversity_window']
        recent_songs = self.episode_songs[-window:]
        
        if song_idx in recent_songs:
            return -1.0  # Penalty for exact repetition
        
        # Check similarity with recent songs
        song_embed = self.song_embeddings[song_idx]
        recent_embeds = self.song_embeddings[recent_songs]
        
        similarities = cosine_similarity(song_embed.reshape(1, -1), recent_embeds)[0]
        max_similarity = np.max(similarities) if len(similarities) > 0 else 0.0
        
        # Bonus if sufficiently different from recent songs
        threshold = self.config['diversity_threshold']
        if max_similarity < threshold:
            return 1.0
        else:
            return -0.5
    
    def _calculate_personalization_bonus(self, song_idx: int) -> float:
        """Calculate how well song matches user preferences"""
        user_embed = self.user_embeddings[self.current_user_idx]
        song_embed = self.song_embeddings[song_idx]
        
        # Truncate to same dimension
        min_dim = min(len(user_embed), len(song_embed))
        similarity = np.dot(user_embed[:min_dim], song_embed[:min_dim]) / (
            np.linalg.norm(user_embed[:min_dim]) * np.linalg.norm(song_embed[:min_dim]) + 1e-8
        )
        
        # Scale to [0, 1]
        return float((similarity + 1) / 2)
    
    def _calculate_serendipity(self, song_idx: int) -> float:
        """Calculate serendipity (relevant but unexpected)"""
        # Relevance (User match)
        relevance = self._calculate_personalization_bonus(song_idx)
        
        # Unexpectedness (Inverse of similarity to recent history)
        if not self.episode_songs:
            unexpectedness = 1.0
        else:
            window = self.config['diversity_window']
            recent_songs = self.episode_songs[-window:]
            
            song_embed = self.song_embeddings[song_idx]
            recent_embeds = self.song_embeddings[recent_songs]
            
            similarities = cosine_similarity(song_embed.reshape(1, -1), recent_embeds)[0]
            max_sim = np.max(similarities) if len(similarities) > 0 else 0.0
            
            unexpectedness = 1.0 - max(0, float(max_sim))
            
        return relevance * unexpectedness
    
    def _get_reward_components(self, song_idx: int, user_reaction: int) -> Dict[str, float]:
        """Get detailed breakdown of reward components for logging"""
        base_rewards = {
            0: self.config['reward_skip'],
            1: self.config['reward_partial_listen'],
            2: self.config['reward_full_listen']
        }
        
        return {
            'base': base_rewards[user_reaction],
            'diversity': self._calculate_diversity_bonus(song_idx) * self.config['reward_diversity_weight'],
            'personalization': self._calculate_personalization_bonus(song_idx) * self.config['reward_personalization_weight'],
            'engagement': (min(self.session_position / 20.0, 1.0) * self.config['reward_session_bonus_weight']) if self.session_position > 5 else 0.0,
            'serendipity': self._calculate_serendipity(song_idx)
        }
    
    def _check_termination(self, user_reaction: int) -> bool:
        """Check if episode should terminate"""
        # User leaves after multiple skips
        if user_reaction == 0:
            recent_outcomes = [h['outcome'] for h in self.user_history[-3:]]
            if len(recent_outcomes) == 3 and all(o == 0 for o in recent_outcomes):
                logger.debug("Episode terminated: 3 consecutive skips")
                return True
        
        # Random dropout
        if self.np_random.random() < self.config['dropout_probability']:
            logger.debug("Episode terminated: random dropout")
            return True
        
        return False
    
    def _log_episode_summary(self, terminated: bool, truncated: bool):
        """Log summary statistics for completed episode"""
        total_reward = sum(self.episode_rewards)
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        
        # Count reactions
        reactions = [h['outcome'] for h in self.user_history]
        skip_rate = reactions.count(0) / len(reactions) if reactions else 0
        completion_rate = reactions.count(2) / len(reactions) if reactions else 0
        
        termination_reason = "natural" if terminated else "truncated"
        
        logger.info(
            f"Episode ended ({termination_reason}) | "
            f"User: {self.current_user_id} | "
            f"Length: {self.session_position} | "
            f"Total reward: {total_reward:.2f} | "
            f"Avg reward: {avg_reward:.3f} | "
            f"Skip rate: {skip_rate:.2%} | "
            f"Completion rate: {completion_rate:.2%}"
        )
    
    def render(self):
        """Render environment (not implemented for this environment)"""
        pass
    
    def close(self):
        """Clean up resources"""
        logger.info("Closing MusicEnv")
        pass


# Example usage
if __name__ == "__main__":
    # Test the environment
    env = MusicEnv(data_path="data", mode="discrete")
    
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Run a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}")
        
        if terminated or truncated:
            break
    
    env.close()