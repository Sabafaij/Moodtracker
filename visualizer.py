import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import BytesIO


class Visualizer:
    """
    Creates visualizations for facial, voice, and mental health analysis results.
    """
    
    def __init__(self):
        """Initialize the visualizer with sophisticated, classy styling settings."""
        # Set up elegant styling for matplotlib
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Set default figure size to be smaller (better performance and prevents decompression bomb errors)
        plt.rcParams['figure.figsize'] = (7, 4)
        plt.rcParams['figure.dpi'] = 90
        
        # Use serif fonts for a classier look
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Serif']
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 13
        
        # Define an elegant color palette with subdued, sophisticated colors
        self.color_palette = {
            'primary': '#3a506b',    # Deep blue-gray
            'secondary': '#5d7b9d',  # Medium blue-gray
            'tertiary': '#1e2a38',   # Dark blue-gray
            'quaternary': '#6c757d', # Slate gray
            'quinary': '#5c6b7a',    # Muted blue
            'senary': '#4e5d6c',     # Steel blue
            
            # Emotion-specific colors with more subdued, sophisticated tones
            'happy': '#4b6584',      # Muted blue
            'sad': '#778ca3',        # Slate blue
            'angry': '#8e44ad',      # Muted purple
            'surprised': '#3867d6',  # Royal blue
            'fearful': '#8854d0',    # Medium purple
            'neutral': '#a5b1c2',    # Light gray-blue
        }
    
    def plot_emotions(self, emotions):
        """
        Create a bar chart of detected emotions.
        
        Args:
            emotions: Dictionary of emotion probabilities
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Sort emotions by probability
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        labels = [item[0] for item in sorted_emotions]
        values = [item[1] for item in sorted_emotions]
        
        # Create colors list based on emotion names
        colors = [self.color_palette.get(emotion, self.color_palette['primary']) for emotion in labels]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, values, color=colors, alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Probability')
        ax.set_title('Detected Emotions', fontsize=14, pad=20)
        
        plt.tight_layout()
        return fig
    
    def plot_pitch_variance(self, mean_pitch, pitch_variance):
        """
        Create a visualization of pitch and variance with elegant styling.
        
        Args:
            mean_pitch: Mean pitch value
            pitch_variance: Pitch variance value
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Use smaller figure size to prevent decompression bomb errors
        fig, ax = plt.subplots(figsize=(7, 4))
        
        # Ensure our variance is reasonable for visualization (prevent excessive calculations)
        safe_variance = min(pitch_variance, 500)
        
        # Create a gauge-like visualization with elegant styling
        # Reduce number of points for better performance
        x = np.linspace(mean_pitch - 2 * np.sqrt(safe_variance),
                       mean_pitch + 2 * np.sqrt(safe_variance), 500)
        
        # Handle edge case where variance is very small
        if safe_variance < 1:
            safe_variance = 1
            
        y = np.exp(-(x - mean_pitch)**2 / (2 * safe_variance)) / np.sqrt(2 * np.pi * safe_variance)
        
        # Plot with refined styling
        ax.plot(x, y, color=self.color_palette['primary'], linewidth=1.5)
        ax.fill_between(x, y, color=self.color_palette['primary'], alpha=0.15)
        
        # Add reference pitch ranges with subtle styling
        ax.axvspan(80, 165, color=self.color_palette['tertiary'], alpha=0.05, label='Male Range')
        ax.axvspan(165, 255, color=self.color_palette['secondary'], alpha=0.05, label='Female Range')
        
        # Mark the mean with elegant styling
        ax.axvline(mean_pitch, color=self.color_palette['tertiary'], linestyle='--', linewidth=1, 
                  label=f'Mean: {mean_pitch:.1f} Hz')
        
        # Variance indicator with sophisticated styling
        variance_text = "Low" if pitch_variance < 50 else "Moderate" if pitch_variance < 200 else "High"
        ax.text(0.95, 0.92, f"Variance: {variance_text}", transform=ax.transAxes, 
                ha='right', va='top', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3', 
                         edgecolor=self.color_palette['quaternary'], linewidth=0.5))
        
        # Refined axis labels with elegant styling
        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel('Distribution', fontsize=10, color=self.color_palette['tertiary'])
        ax.set_title('Voice Pitch Profile', fontsize=13, pad=10, color=self.color_palette['tertiary'])
        
        # Elegant legend with refined styling
        ax.legend(loc='upper right', frameon=True, framealpha=0.7, fontsize=9, 
                 edgecolor=self.color_palette['quaternary'])
        
        # Remove y-axis ticks for cleaner look
        ax.set_yticks([])
        
        # Style the spines for a more refined look
        for spine in ax.spines.values():
            spine.set_color(self.color_palette['quaternary'])
            spine.set_linewidth(0.5)
        
        # Use tight layout but with controlled padding
        fig.tight_layout(pad=1.2)
        return fig
    
    def plot_energy(self, energy_values):
        """
        Create a visualization of voice energy/volume with elegant styling.
        
        Args:
            energy_values: List of energy values
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Use smaller figure size to prevent decompression bomb errors
        fig, ax = plt.subplots(figsize=(7, 4))
        
        # Ensure we're not plotting too many points (for performance)
        if len(energy_values) > 50:
            # Downsample to 50 points for efficiency
            indices = np.linspace(0, len(energy_values)-1, 50, dtype=int)
            energy_values = [energy_values[i] for i in indices]
        
        # Create x-values (time) - reduced number of points
        x = np.linspace(0, len(energy_values) / 10, len(energy_values))  # Assuming ~10 frames per second
        
        # Plot the energy values with refined styling
        ax.plot(x, energy_values, color=self.color_palette['secondary'], linewidth=1.5, alpha=0.9)
        
        # Fill area between curve and minimum value with subtle gradient
        min_energy = min(energy_values)
        ax.fill_between(x, energy_values, min_energy, color=self.color_palette['secondary'], alpha=0.1)
        
        # Add reference levels with elegant styling
        ax.axhline(-20, color=self.color_palette['quaternary'], linestyle='--', alpha=0.3, 
                  linewidth=0.8, label='Typical')
        ax.axhline(-10, color=self.color_palette['tertiary'], linestyle='--', alpha=0.3, 
                  linewidth=0.8, label='Loud')
        ax.axhline(-30, color=self.color_palette['quinary'], linestyle='--', alpha=0.3, 
                  linewidth=0.8, label='Soft')
        
        # Refined axis labels with elegant styling
        ax.set_xlabel('Time (s)', fontsize=10, color=self.color_palette['tertiary'])
        ax.set_ylabel('Energy (dB)', fontsize=10, color=self.color_palette['tertiary'])
        ax.set_title('Voice Energy Profile', fontsize=13, pad=10, color=self.color_palette['tertiary'])
        
        # Elegant legend with refined styling
        ax.legend(loc='lower right', frameon=True, framealpha=0.7, fontsize=9,
                 edgecolor=self.color_palette['quaternary'], title='Speech Levels')
        
        # Style the axes for a classy look
        for spine in ax.spines.values():
            spine.set_color(self.color_palette['quaternary'])
            spine.set_linewidth(0.5)
            
        # Style the ticks for elegance
        ax.tick_params(axis='both', which='major', labelsize=9, colors=self.color_palette['tertiary'])
        
        # Use tight layout with controlled padding
        fig.tight_layout(pad=1.2)
        return fig
    
    def plot_indicators(self, indicators, title):
        """
        Create a simplified horizontal bar chart for indicators.
        
        Args:
            indicators: Dictionary of indicator values
            title: Title for the plot
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Get labels and values, ensuring they're in a specific order for consistency
        labels = list(indicators.keys())
        values = [indicators[label] for label in labels]
        
        # Create the plot - use a smaller, more appropriate size
        fig, ax = plt.subplots(figsize=(5, 3))
        
        # Create horizontal bar chart
        bars = ax.barh(
            [label.replace('_', ' ').title() for label in labels], 
            values,
            color=[self.color_palette['primary']] * len(labels),
            height=0.5
        )
        
        # Add value labels to the right of bars
        for bar in bars:
            width = bar.get_width()
            ax.text(
                min(width + 0.02, 0.98),
                bar.get_y() + bar.get_height()/2,
                f"{width:.2f}",
                va='center',
                fontsize=8,
                color=self.color_palette['tertiary']
            )
        
        # Set x-axis limit
        ax.set_xlim(0, 1.0)
        
        # Add gridlines for readability
        ax.xaxis.grid(True, linestyle='--', alpha=0.6)
        
        # Add title
        ax.set_title(title, fontsize=11, pad=10, color=self.color_palette['tertiary'])
        
        # Style the axes for a cleaner look
        for spine in ax.spines.values():
            spine.set_color(self.color_palette['quaternary'])
            spine.set_linewidth(0.5)
            
        # Add labels with modest styling
        ax.set_xlabel('Score (0-1)', fontsize=8, color=self.color_palette['tertiary'])
        
        # Style the ticks
        ax.tick_params(axis='both', which='major', labelsize=8, colors=self.color_palette['tertiary'])
        
        # Add reference lines
        ax.axvline(0.4, color='gray', linestyle=':', alpha=0.7, linewidth=0.8)
        ax.axvline(0.6, color='gray', linestyle=':', alpha=0.7, linewidth=0.8)
        
        # Add subtle markers for low/moderate/high ranges
        ax.text(0.2, -0.5, 'Low', ha='center', va='top', fontsize=7, alpha=0.7, color=self.color_palette['tertiary'])
        ax.text(0.5, -0.5, 'Moderate', ha='center', va='top', fontsize=7, alpha=0.7, color=self.color_palette['tertiary'])
        ax.text(0.8, -0.5, 'High', ha='center', va='top', fontsize=7, alpha=0.7, color=self.color_palette['tertiary'])
        
        # Use tight layout with controlled padding
        fig.tight_layout(pad=1.0)
        return fig
