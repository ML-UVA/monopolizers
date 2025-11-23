"""
Pygame-based renderer for Monopoly game visualization.

This module provides a graphical interface to visualize the Monopoly game state,
including the board, player positions, property ownership, houses/hotels, and game statistics.
"""

import pygame
import sys
from typing import Optional, Tuple, Dict, List
import numpy as np
from ..state import GameState, PlayerStatus
from ..board import Board, TileKind
from ..property import PropertySpec

# Color definitions
COLORS = {
    'background': (240, 248, 255),
    'board': (220, 245, 220),
    'board_center': (200, 230, 200),
    'board_edge': (50, 100, 50),
    'text': (0, 0, 0),
    'text_light': (100, 100, 100),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'red': (220, 20, 20),
    'green': (34, 139, 34),
    'blue': (30, 144, 255),
    'yellow': (255, 215, 0),
    'orange': (255, 140, 0),
    'purple': (147, 112, 219),
    'light_blue': (135, 206, 250),
    'pink': (255, 192, 203),
    'dark_blue': (0, 0, 139),
    'gray': (169, 169, 169),
    'player_colors': [
        (255, 50, 50),    # Red
        (50, 150, 255),   # Blue
        (50, 205, 50),    # Green
        (255, 165, 0),    # Orange
        (255, 20, 147),   # Pink
        (138, 43, 226),   # Purple
        (255, 215, 0),    # Gold
        (0, 255, 255),    # Cyan
    ]
}

# Property group colors
GROUP_COLORS = {
    'Purple': (147, 112, 219),
    'Light Blue': (135, 206, 250),
    'Pink': (255, 105, 180),
    'Orange': (255, 140, 0),
    'Red': (220, 20, 60),
    'Yellow': (255, 215, 0),
    'Green': (34, 139, 34),
    'Dark Blue': (0, 0, 139),
    'Railroad': (50, 50, 50),
    'Utility': (200, 200, 200),
}


class MonopolyRenderer:
    """Pygame-based renderer for Monopoly game visualization."""
    
    def __init__(self, board: Board, property_specs: List[PropertySpec], width: int = 1200, height: int = 900):
        """
        Initialize the pygame renderer.
        
        Args:
            board: The game board
            property_specs: List of property specifications
            width: Window width in pixels
            height: Window height in pixels
        """
        pygame.init()
        
        self.board = board
        self.property_specs = property_specs
        self.width = width
        self.height = height
        
        # Create display
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Monopoly Game Visualization")
        
        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        self.font_tiny = pygame.font.Font(None, 14)
        
        # Board layout (square board with tiles on edges)
        self.board_margin = 50
        self.board_size = min(width - 300, height - 100) - 2 * self.board_margin
        self.tile_width = self.board_size // 11
        self.tile_height = self.tile_width
        
        # Calculate tile positions (clockwise from bottom-right corner)
        self.tile_positions = self._calculate_tile_positions()
        
        # Animation state
        self.clock = pygame.time.Clock()
        self.fps = 30
        
    def _calculate_tile_positions(self) -> List[Tuple[int, int, int, int]]:
        """Calculate screen positions for all 40 tiles. Returns list of (x, y, width, height)."""
        positions = []
        margin = self.board_margin
        tile_w = self.tile_width
        tile_h = self.tile_height
        board_size = self.board_size
        # We'll place tiles inside the square [margin, margin+board_size]
        origin_x = margin
        origin_y = margin

        # Bottom row (tiles 0-10): right to left. Keep tiles fully inside board.
        bottom_y = origin_y + board_size - tile_h
        for i in range(11):
            x = origin_x + board_size - tile_w - (i * tile_w)
            y = bottom_y
            positions.append((int(x), int(y), int(tile_w), int(tile_h)))

        # Left column (tiles 11-19): bottom to top (excluding corners)
        left_x = origin_x
        for i in range(1, 10):
            x = left_x
            y = origin_y + board_size - tile_h - (i * tile_h)
            positions.append((int(x), int(y), int(tile_w), int(tile_h)))

        # Top row (tiles 20-30): left to right
        top_y = origin_y
        for i in range(11):
            x = origin_x + (i * tile_w)
            y = top_y
            positions.append((int(x), int(y), int(tile_w), int(tile_h)))

        # Right column (tiles 31-39): top to bottom (excluding corners)
        right_x = origin_x + board_size - tile_w
        for i in range(1, 10):
            x = right_x
            y = origin_y + (i * tile_h)
            positions.append((int(x), int(y), int(tile_w), int(tile_h)))
        
        return positions
    
    def render(self, state: GameState, show_stats: bool = True):
        """
        Render the current game state.
        
        Args:
            state: Current game state
            show_stats: Whether to show player statistics panel
        """
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # Clear screen
        self.screen.fill(COLORS['background'])
        
        # Draw board
        self._draw_board(state)
        
        # Draw player positions
        self._draw_players(state)
        
        # Draw stats panel
        if show_stats:
            self._draw_stats_panel(state)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def _draw_board(self, state: GameState):
        """Draw the game board with properties, houses, and ownership."""
        # Draw center area
        center_x = self.board_margin + self.tile_width
        center_y = self.board_margin + self.tile_height
        center_size = self.board_size - 2 * self.tile_width
        pygame.draw.rect(self.screen, COLORS['board_center'], 
                        (center_x, center_y, center_size, center_size))
        
        # Draw title in center
        title = self.font_large.render("MONOPOLY", True, COLORS['text'])
        title_rect = title.get_rect(center=(center_x + center_size // 2, center_y + center_size // 2))
        self.screen.blit(title, title_rect)
        
        # Draw each tile
        for i, tile in enumerate(self.board.tiles):
            self._draw_tile(i, tile, state)
    
    def _draw_tile(self, tile_idx: int, tile, state: GameState):
        """Draw a single tile."""
        x, y, w, h = self.tile_positions[tile_idx]
        
        # Determine background color based on property group
        bg_color = COLORS['white']
        if tile.property_idx is not None:
            prop_spec = self.property_specs[tile.property_idx]
            bg_color = GROUP_COLORS.get(prop_spec.group, COLORS['white'])
        
        # Draw tile background
        pygame.draw.rect(self.screen, bg_color, (x, y, w, h))
        pygame.draw.rect(self.screen, COLORS['board_edge'], (x, y, w, h), 2)
        
        # Draw property color bar at top of tile
        if tile.property_idx is not None:
            prop_spec = self.property_specs[tile.property_idx]
            color_bar_height = h // 6
            color = GROUP_COLORS.get(prop_spec.group, COLORS['gray'])
            pygame.draw.rect(self.screen, color, (x + 2, y + 2, w - 4, color_bar_height))
        
        # Draw tile name (shortened)
        name = tile.name
        if len(name) > 12:
            name = name[:10] + "."
        
        # Rotate text for side tiles
        if 11 <= tile_idx <= 19:  # Left side
            name_surface = self.font_tiny.render(name, True, COLORS['text'])
            name_surface = pygame.transform.rotate(name_surface, 90)
            self.screen.blit(name_surface, (x + 5, y + h - 15))
        elif 21 <= tile_idx <= 29:  # Top side
            name_surface = self.font_tiny.render(name, True, COLORS['text'])
            self.screen.blit(name_surface, (x + 5, y + 15))
        elif 31 <= tile_idx <= 39:  # Right side
            name_surface = self.font_tiny.render(name, True, COLORS['text'])
            name_surface = pygame.transform.rotate(name_surface, 270)
            self.screen.blit(name_surface, (x + w - 15, y + 15))
        else:  # Bottom side
            name_surface = self.font_tiny.render(name, True, COLORS['text'])
            self.screen.blit(name_surface, (x + 5, y + h - 15))
        
        # Draw ownership indicator and houses
        if tile.property_idx is not None:
            prop_state = state.properties[tile.property_idx]
            
            # Draw owner indicator (small colored circle)
            if prop_state.owner is not None:
                owner_color = COLORS['player_colors'][prop_state.owner % len(COLORS['player_colors'])]
                owner_x = x + w - 15
                owner_y = y + h // 2
                pygame.draw.circle(self.screen, owner_color, (owner_x, owner_y), 6)
            
            # Draw houses/hotels
            if prop_state.houses_count > 0:
                house_size = 8
                house_spacing = 10
                house_y = y + h // 2 + 10
                
                if prop_state.houses_count < 5:
                    # Draw houses
                    for i in range(prop_state.houses_count):
                        house_x = x + 10 + (i * house_spacing)
                        pygame.draw.rect(self.screen, COLORS['green'], 
                                       (house_x, house_y, house_size, house_size))
                else:
                    # Draw hotel
                    pygame.draw.rect(self.screen, COLORS['red'], 
                                   (x + 10, house_y, house_size * 2, house_size))
                    hotel_text = self.font_tiny.render("H", True, COLORS['white'])
                    self.screen.blit(hotel_text, (x + 12, house_y))
            
            # Draw mortgage indicator
            if prop_state.mortgaged:
                pygame.draw.line(self.screen, COLORS['red'], 
                               (x + 5, y + 5), (x + w - 5, y + h - 5), 3)
                pygame.draw.line(self.screen, COLORS['red'], 
                               (x + w - 5, y + 5), (x + 5, y + h - 5), 3)
        
        # Special tiles
        if tile.kind == TileKind.GO:
            go_text = self.font_small.render("GO", True, COLORS['green'])
            self.screen.blit(go_text, (x + w // 4, y + h // 3))
        elif tile.kind == TileKind.JAIL:
            jail_text = self.font_small.render("JAIL", True, COLORS['red'])
            self.screen.blit(jail_text, (x + 10, y + h // 3))
        elif tile.kind == TileKind.GO_TO_JAIL:
            gtj_text = self.font_small.render("→JAIL", True, COLORS['red'])
            self.screen.blit(gtj_text, (x + 10, y + h // 3))
        elif tile.kind == TileKind.FREE_PARKING:
            fp_text = self.font_small.render("FREE", True, COLORS['blue'])
            self.screen.blit(fp_text, (x + 15, y + h // 3))
    
    def _draw_players(self, state: GameState):
        """Draw player tokens on their current positions."""
        # Group players by position
        position_counts: Dict[int, List[int]] = {}
        for player in state.players:
            if player.status == PlayerStatus.ACTIVE:
                if player.position not in position_counts:
                    position_counts[player.position] = []
                position_counts[player.position].append(player.id)
        
        # Draw each player
        for position, player_ids in position_counts.items():
            x, y, w, h = self.tile_positions[position]
            
            # Arrange multiple players on same tile
            num_players = len(player_ids)
            for idx, player_id in enumerate(player_ids):
                player = state.players[player_id]
                
                # Calculate player token position
                if num_players == 1:
                    px = x + w // 2
                    py = y + h // 2
                elif num_players == 2:
                    px = x + w // 3 + (idx * w // 3)
                    py = y + h // 2
                else:
                    # Arrange in grid
                    row = idx // 2
                    col = idx % 2
                    px = x + w // 3 + (col * w // 3)
                    py = y + h // 3 + (row * h // 3)
                
                # Draw player token (circle with number)
                player_color = COLORS['player_colors'][player_id % len(COLORS['player_colors'])]
                # ensure integers
                px_i = int(px)
                py_i = int(py)
                token_radius = 10
                pygame.draw.circle(self.screen, player_color, (px_i, py_i), token_radius)
                pygame.draw.circle(self.screen, COLORS['black'], (px_i, py_i), token_radius, 2)
                
                # Draw player number
                player_text = self.font_small.render(str(player_id), True, COLORS['white'])
                text_rect = player_text.get_rect(center=(px_i, py_i))
                self.screen.blit(player_text, text_rect)
                
                # Draw jail indicator
                if player.jail_turns > 0:
                    jail_indicator = self.font_tiny.render("🔒", True, COLORS['red'])
                    self.screen.blit(jail_indicator, (px_i - 8, py_i - 20))
    
    def _draw_stats_panel(self, state: GameState):
        """Draw player statistics panel on the right side."""
        panel_x = self.board_margin + self.board_size + 30
        panel_y = self.board_margin
        panel_width = max(220, self.width - panel_x - 20)
        panel_height = self.height - 2 * self.board_margin
        
        # Draw panel background
        pygame.draw.rect(self.screen, COLORS['white'], 
                        (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, COLORS['board_edge'], 
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Draw title
        title = self.font_medium.render("Game Stats", True, COLORS['text'])
        self.screen.blit(title, (panel_x + 10, panel_y + 10))

        # Spacing metrics
        header_h = self.font_small.get_height() + 6
        line_h = self.font_tiny.get_height() + 4

        # Draw turn number
        turn_text = self.font_small.render(f"Turn: {state.turn_number}", True, COLORS['text'])
        self.screen.blit(turn_text, (panel_x + 10, panel_y + 10 + header_h))

        # Draw current player indicator
        current_text = self.font_small.render(f"Current: P{state.current_player}", True, COLORS['text'])
        self.screen.blit(current_text, (panel_x + 10, panel_y + 10 + header_h + line_h))

        # Draw dice roll
        y_cursor = panel_y + 10 + header_h + 2 * line_h
        if state.last_roll:
            dice_text = self.font_small.render(f"Last Roll: {state.last_roll[0]} + {state.last_roll[1]} = {sum(state.last_roll)}", 
                                               True, COLORS['text'])
            self.screen.blit(dice_text, (panel_x + 10, y_cursor))
            y_cursor += line_h

        # Draw bank resources
        bank_text = self.font_small.render(f"Houses: {state.bank_houses_left}  Hotels: {state.bank_hotels_left}", 
                                           True, COLORS['text_light'])
        self.screen.blit(bank_text, (panel_x + 10, y_cursor))
        y_cursor += line_h

        # Draw separator
        sep_y = y_cursor + 4
        pygame.draw.line(self.screen, COLORS['gray'], 
                        (panel_x + 10, sep_y), 
                        (panel_x + panel_width - 10, sep_y), 1)
        y_cursor = sep_y + 8

        # Draw player info
        for i, player in enumerate(state.players):
            player_color = COLORS['player_colors'][i % len(COLORS['player_colors'])]

            # Player header
            player_header = f"Player {i} {'(YOU)' if i == 0 else ''}"
            if player.status == PlayerStatus.BANKRUPT:
                player_header += " [BANKRUPT]"
                header_color = COLORS['red']
            elif i == state.current_player:
                header_color = COLORS['green']
            else:
                header_color = COLORS['text']

            header_text = self.font_small.render(player_header, True, header_color)
            self.screen.blit(header_text, (panel_x + 10, y_cursor))

            # Player color indicator
            indicator_x = panel_x + panel_width - 26
            pygame.draw.circle(self.screen, player_color, 
                             (indicator_x, y_cursor + header_h // 2), 8)

            y_cursor += header_h

            # Player stats (only for active players)
            if player.status == PlayerStatus.ACTIVE:
                # Cash
                cash_text = self.font_tiny.render(f"Cash: ${player.cash}", True, COLORS['text'])
                self.screen.blit(cash_text, (panel_x + 10, y_cursor))
                y_cursor += line_h

                # Position
                pos_name = self.board.get_tile(player.position).name
                if len(pos_name) > 20:
                    pos_name = pos_name[:18] + ".."
                pos_text = self.font_tiny.render(f"Pos: {pos_name}", True, COLORS['text'])
                self.screen.blit(pos_text, (panel_x + 10, y_cursor))
                y_cursor += line_h

                # Properties
                prop_text = self.font_tiny.render(f"Properties: {len(player.properties_owned)}", True, COLORS['text'])
                self.screen.blit(prop_text, (panel_x + 10, y_cursor))
                y_cursor += line_h

                # Jail status
                if player.jail_turns > 0:
                    jail_text = self.font_tiny.render(f"In Jail: {player.jail_turns} turns", True, COLORS['red'])
                    self.screen.blit(jail_text, (panel_x + 10, y_cursor))
                    y_cursor += line_h

                # Get out of jail cards
                if player.get_out_of_jail_cards > 0:
                    card_text = self.font_tiny.render(f"Jail Cards: {player.get_out_of_jail_cards}", True, COLORS['green'])
                    self.screen.blit(card_text, (panel_x + 10, y_cursor))
                    y_cursor += line_h

            y_cursor += line_h // 2  # Space between players
        
        # Draw legend at bottom
        legend_y = panel_y + panel_height - 80
        pygame.draw.line(self.screen, COLORS['gray'], 
                        (panel_x + 10, legend_y - 10), 
                        (panel_x + panel_width - 10, legend_y - 10), 1)
        
        legend_title = self.font_tiny.render("Legend:", True, COLORS['text'])
        self.screen.blit(legend_title, (panel_x + 10, legend_y))
        
        legend_items = [
            ("🟩 = House", legend_y + 15),
            ("🟥 = Hotel", legend_y + 30),
            ("❌ = Mortgaged", legend_y + 45),
        ]
        
        for text, y in legend_items:
            item_text = self.font_tiny.render(text, True, COLORS['text'])
            self.screen.blit(item_text, (panel_x + 10, y))
    
    def close(self):
        """Clean up pygame resources."""
        pygame.quit()
    
    def wait_for_key(self, duration_ms: Optional[int] = None):
        """
        Wait for a key press or specified duration.
        
        Args:
            duration_ms: If provided, wait for this many milliseconds instead of waiting for key
        """
        if duration_ms:
            pygame.time.wait(duration_ms)
        else:
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        waiting = False
                        break
                self.clock.tick(30)
