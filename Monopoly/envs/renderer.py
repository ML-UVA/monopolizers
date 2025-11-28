"""
Pygame-based renderer for Monopoly game visualization.

This module provides an authentic Monopoly-style graphical interface with:
- Classic green board background and proper property colors
- Larger corner tiles, proper color bands
- 3D houses and hotels
- Iconic player tokens
- Dice visualization
- Enhanced stats panel with net worth
"""

import pygame
import sys
import math
from typing import Optional, Tuple, Dict, List
from ..state import GameState, PlayerStatus
from ..board import Board, TileKind
from ..property import PropertySpec

# ============================================================================
# AUTHENTIC MONOPOLY COLOR PALETTE
# ============================================================================
COLORS = {
    # Board colors
    'board_green': (195, 225, 194),
    'board_center': (205, 230, 195),
    'board_border': (0, 100, 0),
    'tile_bg': (217, 230, 193),
    'table_light': (156, 110, 74),
    'table_dark': (92, 62, 44),
    'center_ribbon': (204, 24, 34),
    'center_ribbon_shadow': (150, 12, 24),
    
    # UI colors  
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'cream': (255, 253, 240),
    'panel_shadow': (0, 0, 0, 70),
    'tile_glow': (255, 255, 255, 60),
    
    # Special tile colors
    'go_red': (220, 36, 31),
    'jail_orange': (242, 142, 43),
    'go_to_jail_blue': (0, 114, 187),
    'chance_orange': (255, 102, 0),
    'community_blue': (170, 224, 250),
    'tax_diamond': (255, 215, 0),
    
    # Text colors
    'text_dark': (30, 30, 30),
    'text_light': (100, 100, 100),
    'text_red': (200, 0, 0),
    'text_green': (0, 150, 0),
    
    # Building colors
    'house_green': (0, 155, 72),
    'house_dark': (0, 100, 50),
    'hotel_red': (207, 32, 46),
    'hotel_dark': (150, 20, 30),
    
    # Player token colors
    'token_colors': [
        (192, 192, 192),
        (255, 215, 0),
        (139, 90, 43),
        (70, 130, 180),
        (220, 20, 60),
        (50, 205, 50),
        (255, 140, 0),
        (138, 43, 226),
    ],
}

# Property group colors - authentic Monopoly
GROUP_COLORS = {
    'Purple': (89, 49, 95),
    'Light Blue': (170, 224, 250),
    'Pink': (217, 58, 150),
    'Orange': (247, 148, 29),
    'Red': (237, 27, 36),
    'Yellow': (254, 242, 0),
    'Green': (31, 178, 90),
    'Dark Blue': (0, 114, 187),
    'Railroad': (40, 40, 40),
    'Utility': (200, 200, 200),
}


class MonopolyRenderer:
    """Authentic Monopoly-style pygame renderer."""
    
    def __init__(self, board: Board, property_specs: List[PropertySpec], 
                 width: int = 1400, height: int = 900):
        pygame.init()
        
        self.board = board
        self.property_specs = property_specs
        self.width = width
        self.height = height
        
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("MONOPOLY")
        
        self._init_fonts()
        self.background_surface = self._create_background_surface(width, height)
        
        self.board_margin = 30
        self.board_size = min(width - 380, height - 60)
        self.corner_size = int(self.board_size / 9.5)
        self.tile_width = int((self.board_size - 2 * self.corner_size) / 9)
        self.tile_height = self.corner_size
        
        self.tile_positions = self._calculate_tile_positions()
        self.clock = pygame.time.Clock()
        self.fps = 30
        
    def _init_fonts(self):
        """Initialize fonts."""
        try:
            # Use Verdana for better readability on small text
            self.font_title = pygame.font.SysFont('Georgia', 50, bold=True)
            self.font_large = pygame.font.SysFont('Verdana', 30, bold=True)
            self.font_medium = pygame.font.SysFont('Verdana', 20, bold=True)
            self.font_small = pygame.font.SysFont('Verdana', 16, bold=True)
            self.font_tiny = pygame.font.SysFont('Verdana', 12, bold=True)
            self.font_price = pygame.font.SysFont('Verdana', 11)
        except:
            self.font_title = pygame.font.Font(None, 54)
            self.font_large = pygame.font.Font(None, 36)
            self.font_medium = pygame.font.Font(None, 24)
            self.font_small = pygame.font.Font(None, 20)
            self.font_tiny = pygame.font.Font(None, 16)
            self.font_price = pygame.font.Font(None, 14)
    
    def _create_background_surface(self, width: int, height: int) -> pygame.Surface:
        """Generate a warm wood table gradient background once."""
        surface = pygame.Surface((width, height))
        top = COLORS['table_light']
        bottom = COLORS['table_dark']
        for y in range(height):
            t = y / max(1, height - 1)
            r = int(top[0] * (1 - t) + bottom[0] * t)
            g = int(top[1] * (1 - t) + bottom[1] * t)
            b = int(top[2] * (1 - t) + bottom[2] * t)
            pygame.draw.line(surface, (r, g, b), (0, y), (width, y))
        
        # Subtle vignette for depth
        vignette = pygame.Surface((width, height), pygame.SRCALPHA)
        for radius in range(50, max(width, height), 40):
            alpha = max(0, 140 - radius // 3)
            pygame.draw.circle(
                vignette,
                (0, 0, 0, alpha),
                (width // 2, height // 2),
                radius,
                4,
            )
        surface.blit(vignette, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)
        return surface
    
    def _calculate_tile_positions(self) -> List[Dict]:
        """Calculate positions for all 40 tiles with proper corner sizing."""
        positions = []
        m = self.board_margin
        cs = self.corner_size
        tw = self.tile_width
        bs = self.board_size
        
        # Tile 0: GO (bottom-right corner)
        positions.append({'x': m + bs - cs, 'y': m + bs - cs, 'w': cs, 'h': cs, 'side': 'corner', 'corner': 'go'})
        
        # Bottom row (tiles 1-9)
        for i in range(9):
            positions.append({'x': m + cs + (8 - i) * tw, 'y': m + bs - cs, 'w': tw, 'h': cs, 'side': 'bottom'})
        
        # Tile 10: Jail
        positions.append({'x': m, 'y': m + bs - cs, 'w': cs, 'h': cs, 'side': 'corner', 'corner': 'jail'})
        
        # Left column (tiles 11-19)
        for i in range(9):
            positions.append({'x': m, 'y': m + cs + (8 - i) * tw, 'w': cs, 'h': tw, 'side': 'left'})
        
        # Tile 20: Free Parking
        positions.append({'x': m, 'y': m, 'w': cs, 'h': cs, 'side': 'corner', 'corner': 'free_parking'})
        
        # Top row (tiles 21-29)
        for i in range(9):
            positions.append({'x': m + cs + i * tw, 'y': m, 'w': tw, 'h': cs, 'side': 'top'})
        
        # Tile 30: Go To Jail
        positions.append({'x': m + bs - cs, 'y': m, 'w': cs, 'h': cs, 'side': 'corner', 'corner': 'go_to_jail'})
        
        # Right column (tiles 31-39)
        for i in range(9):
            positions.append({'x': m + bs - cs, 'y': m + cs + i * tw, 'w': cs, 'h': tw, 'side': 'right'})
        
        return positions
    
    def render(self, state: GameState, show_stats: bool = True):
        """Render the complete game state."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        self.screen.blit(self.background_surface, (0, 0))
        self._draw_board_shadow()
        self._draw_board(state)
        self._draw_center_area(state)
        self._draw_all_tiles(state)
        self._draw_players(state)
        
        if show_stats:
            self._draw_stats_panel(state)
        
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def _draw_board_shadow(self):
        """Draw drop shadow under board."""
        m = self.board_margin
        bs = self.board_size
        shadow_surf = pygame.Surface((bs + 10, bs + 10), pygame.SRCALPHA)
        shadow_surf.fill((0, 0, 0, 80))
        self.screen.blit(shadow_surf, (m + 8, m + 8))
    
    def _draw_board(self, state: GameState):
        """Draw the main board background."""
        m = self.board_margin
        bs = self.board_size
        pygame.draw.rect(self.screen, COLORS['board_green'], (m, m, bs, bs))
        pygame.draw.rect(self.screen, COLORS['board_border'], (m, m, bs, bs), 4)
    
    def _draw_center_area(self, state: GameState):
        """Draw the center area with logo and dice."""
        m = self.board_margin
        cs = self.corner_size
        bs = self.board_size
        cx = m + cs
        cy = m + cs
        cw = bs - 2 * cs
        ch = bs - 2 * cs
        
        pygame.draw.rect(self.screen, COLORS['board_center'], (cx, cy, cw, ch))
        
        center_x = cx + cw // 2
        center_y = cy + ch // 2
        
        self._draw_center_logo(center_x, center_y)
        
        if state.last_roll:
            self._draw_dice(center_x, center_y + 50, state.last_roll)
        
        self._draw_card_decks(cx, cy, cw, ch)

    def _draw_center_logo(self, center_x: int, center_y: int):
        """Draw the classic red Monopoly ribbon and mascot moustache."""
        ribbon_w = 320
        ribbon_h = 70
        ribbon_rect = pygame.Rect(center_x - ribbon_w // 2, center_y - ribbon_h // 2 - 20, ribbon_w, ribbon_h)
        shadow_rect = ribbon_rect.inflate(10, 8)
        pygame.draw.rect(self.screen, COLORS['center_ribbon_shadow'], shadow_rect, border_radius=10)
        pygame.draw.rect(self.screen, COLORS['center_ribbon'], ribbon_rect, border_radius=10)
        title = self.font_title.render("MONOPOLY", True, COLORS['white'])
        title_rect = title.get_rect(center=ribbon_rect.center)
        self.screen.blit(title, title_rect)
        pygame.draw.rect(self.screen, COLORS['white'], ribbon_rect, 2, border_radius=10)
        
        # Draw Mr. Monopoly moustache icon underneath
        mustache_y = ribbon_rect.bottom + 18
        left = pygame.Rect(center_x - 50, mustache_y, 40, 20)
        right = pygame.Rect(center_x + 10, mustache_y, 40, 20)
        pygame.draw.ellipse(self.screen, COLORS['black'], left)
        pygame.draw.ellipse(self.screen, COLORS['black'], right)
        pygame.draw.circle(self.screen, COLORS['black'], (center_x - 5, mustache_y + 18), 10)
        pygame.draw.circle(self.screen, COLORS['black'], (center_x + 15, mustache_y + 18), 10)
    
    def _draw_dice(self, cx: int, cy: int, roll: Tuple[int, int]):
        """Draw two dice showing the last roll."""
        die_size = 40
        spacing = 15
        
        for i, value in enumerate(roll):
            dx = cx + (i * (die_size + spacing)) - die_size - spacing // 2
            dy = cy - die_size // 2
            die_rect = pygame.Rect(dx, dy, die_size, die_size)
            pygame.draw.rect(self.screen, COLORS['white'], die_rect, border_radius=6)
            pygame.draw.rect(self.screen, COLORS['black'], die_rect, 2, border_radius=6)
            self._draw_die_pips(dx, dy, die_size, value)
    
    def _draw_die_pips(self, x: int, y: int, size: int, value: int):
        """Draw pips on a die face."""
        pip_r = size // 10
        c = COLORS['black']
        cx, cy = x + size // 2, y + size // 2
        off = size // 4
        
        pip_map = {
            1: [(cx, cy)],
            2: [(cx - off, cy - off), (cx + off, cy + off)],
            3: [(cx - off, cy - off), (cx, cy), (cx + off, cy + off)],
            4: [(cx - off, cy - off), (cx + off, cy - off), (cx - off, cy + off), (cx + off, cy + off)],
            5: [(cx - off, cy - off), (cx + off, cy - off), (cx, cy), (cx - off, cy + off), (cx + off, cy + off)],
            6: [(cx - off, cy - off), (cx + off, cy - off), (cx - off, cy), (cx + off, cy), (cx - off, cy + off), (cx + off, cy + off)],
        }
        
        for px, py in pip_map.get(value, []):
            pygame.draw.circle(self.screen, c, (int(px), int(py)), pip_r)
    
    def _draw_card_decks(self, cx: int, cy: int, cw: int, ch: int):
        """Draw Chance and Community Chest card decks."""
        deck_w, deck_h = 60, 40
        
        chance_x = cx + cw - deck_w - 40
        chance_y = cy + 40
        self._draw_card_deck(chance_x, chance_y, deck_w, deck_h, COLORS['chance_orange'], "CHANCE")
        
        cc_x = cx + 40
        cc_y = cy + ch - deck_h - 40
        self._draw_card_deck(cc_x, cc_y, deck_w, deck_h, COLORS['community_blue'], "COMMUNITY")
    
    def _draw_card_deck(self, x: int, y: int, w: int, h: int, color: Tuple, label: str):
        """Draw a single card deck."""
        for i in range(3):
            offset = (2 - i) * 2
            rect = pygame.Rect(x + offset, y + offset, w, h)
            pygame.draw.rect(self.screen, COLORS['white'], rect, border_radius=3)
            pygame.draw.rect(self.screen, color, rect, 2, border_radius=3)
        
        inner_rect = pygame.Rect(x + 5, y + 5, w - 10, h - 10)
        pygame.draw.rect(self.screen, color, inner_rect, border_radius=2)
        
        text = self.font_price.render(label, True, COLORS['white'])
        text_rect = text.get_rect(center=(x + w // 2, y + h // 2))
        self.screen.blit(text, text_rect)
    
    def _draw_all_tiles(self, state: GameState):
        """Draw all 40 tiles."""
        for i, tile in enumerate(self.board.tiles):
            self._draw_tile(i, tile, state)
    
    def _draw_tile(self, idx: int, tile, state: GameState):
        """Draw a single tile based on its type."""
        pos = self.tile_positions[idx]
        x, y, w, h = pos['x'], pos['y'], pos['w'], pos['h']
        side = pos['side']
        
        pygame.draw.rect(self.screen, COLORS['tile_bg'], (x, y, w, h))
        pygame.draw.rect(self.screen, COLORS['black'], (x, y, w, h), 1)
        current_position = state.players[state.current_player].position
        if idx == current_position:
            glow = pygame.Surface((w, h), pygame.SRCALPHA)
            glow.fill(COLORS['tile_glow'])
            self.screen.blit(glow, (x, y))
        
        if side == 'corner':
            self._draw_corner_tile(idx, tile, pos, state)
        elif tile.kind == TileKind.PROPERTY:
            self._draw_property_tile(idx, tile, pos, state)
        elif tile.kind == TileKind.RAILROAD:
            self._draw_railroad_tile(idx, tile, pos, state)
        elif tile.kind == TileKind.UTILITY:
            self._draw_utility_tile(idx, tile, pos, state)
        elif tile.kind == TileKind.CHANCE:
            self._draw_chance_tile(idx, tile, pos)
        elif tile.kind == TileKind.COMMUNITY:
            self._draw_community_tile(idx, tile, pos)
        elif tile.kind == TileKind.TAX:
            self._draw_tax_tile(idx, tile, pos)
    
    def _draw_corner_tile(self, idx: int, tile, pos: Dict, state: GameState):
        """Draw a corner tile."""
        x, y, w, h = pos['x'], pos['y'], pos['w'], pos['h']
        corner = pos.get('corner', '')
        
        if corner == 'go':
            self._draw_go_tile(x, y, w, h)
        elif corner == 'jail':
            self._draw_jail_tile(x, y, w, h)
        elif corner == 'free_parking':
            self._draw_free_parking_tile(x, y, w, h)
        elif corner == 'go_to_jail':
            self._draw_go_to_jail_tile(x, y, w, h)
    
    def _draw_go_tile(self, x: int, y: int, w: int, h: int):
        """Draw the GO corner tile."""
        go_text = self.font_large.render("GO", True, COLORS['go_red'])
        go_rect = go_text.get_rect(center=(x + w // 2, y + h // 2 - 15))
        self.screen.blit(go_text, go_rect)
        
        arrow_y = y + h // 2 + 15
        arrow_points = [(x + w // 4, arrow_y), (x + 3 * w // 4, arrow_y), (x + 3 * w // 4 - 10, arrow_y - 10), (x + 3 * w // 4, arrow_y), (x + 3 * w // 4 - 10, arrow_y + 10)]
        pygame.draw.lines(self.screen, COLORS['go_red'], False, arrow_points, 4)
        
        collect = self.font_tiny.render("COLLECT $200", True, COLORS['text_dark'])
        collect_rect = collect.get_rect(center=(x + w // 2, y + h - 15))
        self.screen.blit(collect, collect_rect)
    
    def _draw_jail_tile(self, x: int, y: int, w: int, h: int):
        """Draw the Jail tile."""
        jail_size = w // 2
        jail_x = x + w - jail_size - 5
        jail_y = y + 5
        
        pygame.draw.rect(self.screen, COLORS['jail_orange'], (jail_x, jail_y, jail_size, jail_size))
        
        for i in range(4):
            bx = jail_x + (i + 1) * jail_size // 5
            pygame.draw.line(self.screen, COLORS['black'], (bx, jail_y), (bx, jail_y + jail_size), 2)
        
        in_text = self.font_tiny.render("IN", True, COLORS['white'])
        self.screen.blit(in_text, (jail_x + jail_size // 2 - 6, jail_y + 5))
        
        jail_text = self.font_small.render("JAIL", True, COLORS['black'])
        jail_rect = jail_text.get_rect(center=(jail_x + jail_size // 2, jail_y + jail_size - 12))
        self.screen.blit(jail_text, jail_rect)
        
        visit = self.font_tiny.render("JUST VISITING", True, COLORS['text_dark'])
        visit_rot = pygame.transform.rotate(visit, 45)
        self.screen.blit(visit_rot, (x + 5, y + h - 35))
    
    def _draw_free_parking_tile(self, x: int, y: int, w: int, h: int):
        """Draw the Free Parking corner with improved car icon."""
        free = self.font_medium.render("FREE", True, COLORS['go_red'])
        free_rect = free.get_rect(center=(x + w // 2, y + h // 4))
        self.screen.blit(free, free_rect)
        
        park = self.font_medium.render("PARKING", True, COLORS['go_red'])
        park_rect = park.get_rect(center=(x + w // 2, y + 3 * h // 4))
        self.screen.blit(park, park_rect)
        
        # Draw a better car
        cx, cy = x + w // 2, y + h // 2
        car_color = COLORS['black']
        
        # Car body (sedan shape)
        # Bottom part
        pygame.draw.rect(self.screen, car_color, (cx - 20, cy, 40, 12), border_radius=4)
        # Top part (cabin)
        pygame.draw.rect(self.screen, car_color, (cx - 10, cy - 8, 20, 10), border_top_left_radius=4, border_top_right_radius=4)
        
        # Wheels
        pygame.draw.circle(self.screen, car_color, (cx - 12, cy + 12), 5)
        pygame.draw.circle(self.screen, car_color, (cx + 12, cy + 12), 5)
        pygame.draw.circle(self.screen, COLORS['white'], (cx - 12, cy + 12), 2)
        pygame.draw.circle(self.screen, COLORS['white'], (cx + 12, cy + 12), 2)
    
    def _draw_go_to_jail_tile(self, x: int, y: int, w: int, h: int):
        """Draw the Go To Jail corner with policeman icon."""
        goto = self.font_small.render("GO TO", True, COLORS['go_to_jail_blue'])
        goto_rect = goto.get_rect(center=(x + w // 2, y + h // 4))
        self.screen.blit(goto, goto_rect)
        
        jail = self.font_large.render("JAIL", True, COLORS['go_to_jail_blue'])
        jail_rect = jail.get_rect(center=(x + w // 2, y + h // 2))
        self.screen.blit(jail, jail_rect)
        
        # Draw Policeman Icon
        cx, cy = x + w // 2, y + 3 * h // 4
        
        # Hat
        pygame.draw.rect(self.screen, COLORS['go_to_jail_blue'], (cx - 12, cy - 10, 24, 8)) # Brim
        pygame.draw.rect(self.screen, COLORS['go_to_jail_blue'], (cx - 10, cy - 18, 20, 10), border_top_left_radius=5, border_top_right_radius=5) # Top
        
        # Badge
        pygame.draw.circle(self.screen, (255, 215, 0), (cx, cy - 13), 3)
        
        # Head
        pygame.draw.circle(self.screen, (255, 200, 180), (cx, cy), 8)
        
        # Shoulders
        pygame.draw.arc(self.screen, COLORS['go_to_jail_blue'], (cx - 16, cy, 32, 20), 0, 3.14, 10)
    
    def _draw_property_tile(self, idx: int, tile, pos: Dict, state: GameState):
        """Draw a property tile with color band."""
        x, y, w, h = pos['x'], pos['y'], pos['w'], pos['h']
        side = pos['side']
        
        prop_spec = self.property_specs[tile.property_idx]
        color = GROUP_COLORS.get(prop_spec.group, COLORS['white'])
        band_size = h // 5 if side in ['top', 'bottom'] else w // 5
        
        if side == 'bottom':
            pygame.draw.rect(self.screen, color, (x + 1, y + 1, w - 2, band_size))
            pygame.draw.line(self.screen, COLORS['black'], (x, y + band_size), (x + w, y + band_size), 1)
            self._draw_property_content(x, y + band_size, w, h - band_size, prop_spec, tile, state, 'bottom')
        elif side == 'top':
            pygame.draw.rect(self.screen, color, (x + 1, y + h - band_size - 1, w - 2, band_size))
            pygame.draw.line(self.screen, COLORS['black'], (x, y + h - band_size), (x + w, y + h - band_size), 1)
            self._draw_property_content(x, y, w, h - band_size, prop_spec, tile, state, 'top')
        elif side == 'left':
            pygame.draw.rect(self.screen, color, (x + w - band_size - 1, y + 1, band_size, h - 2))
            pygame.draw.line(self.screen, COLORS['black'], (x + w - band_size, y), (x + w - band_size, y + h), 1)
            self._draw_property_content(x, y, w - band_size, h, prop_spec, tile, state, 'left')
        elif side == 'right':
            pygame.draw.rect(self.screen, color, (x + 1, y + 1, band_size, h - 2))
            pygame.draw.line(self.screen, COLORS['black'], (x + band_size, y), (x + band_size, y + h), 1)
            self._draw_property_content(x + band_size, y, w - band_size, h, prop_spec, tile, state, 'right')
        
        self._draw_buildings(idx, tile, pos, state)
        self._draw_ownership(tile, pos, state)
    
    def _draw_property_content(self, x: int, y: int, w: int, h: int, prop_spec, tile, state: GameState, side: str):
        """Draw property name and price with improved visibility."""
        name = prop_spec.name
        words = name.split()
        
        # Use smaller font for fitting
        font = self.font_price
        
        if side in ['bottom', 'top']:
            # Center text in the white area
            # Calculate available height for text
            text_area_h = h - 16  # Leave room for price
            
            if len(words) > 1:
                # Split into two lines
                if len(words) == 2:
                    line1 = words[0]
                    line2 = words[1]
                else:
                    mid = len(words) // 2
                    line1 = ' '.join(words[:mid+1])
                    line2 = ' '.join(words[mid+1:])
                
                # Truncate if needed
                max_chars = w // 6
                line1 = line1[:max_chars]
                line2 = line2[:max_chars]
                
                t1 = font.render(line1, True, COLORS['text_dark'])
                t2 = font.render(line2, True, COLORS['text_dark'])
                
                # Stack vertically centered
                total_text_h = t1.get_height() + t2.get_height() + 2
                start_y = y + (text_area_h - total_text_h) // 2
                
                self.screen.blit(t1, (x + (w - t1.get_width()) // 2, start_y))
                self.screen.blit(t2, (x + (w - t2.get_width()) // 2, start_y + t1.get_height() + 2))
            else:
                max_chars = w // 6
                t = font.render(name[:max_chars], True, COLORS['text_dark'])
                self.screen.blit(t, (x + (w - t.get_width()) // 2, y + (text_area_h - t.get_height()) // 2))
            
            price = self.font_price.render(f"${prop_spec.price}", True, COLORS['text_dark'])
            self.screen.blit(price, (x + (w - price.get_width()) // 2, y + h - 13))
        else:
            # Side tiles - rotate text to read from outside
            # Left side: text reads bottom-to-top (rotate 90)
            # Right side: text reads top-to-bottom (rotate -90)
            angle = 90 if side == 'left' else -90
            
            # Truncate name to fit
            max_chars = h // 7
            short_name = name[:max_chars] if len(name) > max_chars else name
            
            t = font.render(short_name, True, COLORS['text_dark'])
            t_rot = pygame.transform.rotate(t, angle)
            
            # Center rotated text in available area
            text_area_w = w - 14  # Leave room for price
            cx = x + text_area_w // 2
            cy = y + h // 2
            self.screen.blit(t_rot, (cx - t_rot.get_width() // 2, cy - t_rot.get_height() // 2))
            
            # Price near the board edge
            price = self.font_price.render(f"${prop_spec.price}", True, COLORS['text_dark'])
            price_rot = pygame.transform.rotate(price, angle)
            
            if side == 'left':
                px = x + w - 8
            else:
                px = x + 8
            
            self.screen.blit(price_rot, (px - price_rot.get_width() // 2, cy - price_rot.get_height() // 2))
    
    def _draw_buildings(self, idx: int, tile, pos: Dict, state: GameState):
        """Draw houses or hotel on property."""
        if tile.property_idx is None:
            return
        prop_state = state.properties[tile.property_idx]
        if prop_state.houses_count == 0:
            return
        
        x, y, w, h = pos['x'], pos['y'], pos['w'], pos['h']
        side = pos['side']
        
        if prop_state.houses_count == 5:
            self._draw_hotel(x, y, w, h, side)
        else:
            self._draw_houses(x, y, w, h, side, prop_state.houses_count)
    
    def _draw_houses(self, x: int, y: int, w: int, h: int, side: str, count: int):
        """Draw green houses on a property, evenly spaced."""
        house_w, house_h = 8, 6
        
        for i in range(count):
            # Calculate evenly spaced positions
            if side == 'bottom':
                spacing = (w - 8) // max(count, 1)
                hx = x + 4 + i * spacing + (spacing - house_w) // 2
                hy = y + 2
                # Draw house facing down (roof at top)
                roof_points = [(hx + house_w // 2, hy), (hx, hy + 4), (hx + house_w, hy + 4)]
                pygame.draw.polygon(self.screen, COLORS['house_dark'], roof_points)
                pygame.draw.rect(self.screen, COLORS['house_green'], (hx, hy + 4, house_w, house_h - 2))
            elif side == 'top':
                spacing = (w - 8) // max(count, 1)
                hx = x + 4 + i * spacing + (spacing - house_w) // 2
                hy = y + h - house_h - 2
                # Draw house facing up (roof at bottom)
                roof_points = [(hx + house_w // 2, hy + house_h + 2), (hx, hy + house_h - 2), (hx + house_w, hy + house_h - 2)]
                pygame.draw.polygon(self.screen, COLORS['house_dark'], roof_points)
                pygame.draw.rect(self.screen, COLORS['house_green'], (hx, hy, house_w, house_h - 2))
            elif side == 'left':
                spacing = (h - 8) // max(count, 1)
                hx = x + w - house_h - 2
                hy = y + 4 + i * spacing + (spacing - house_w) // 2
                # Draw house facing left (roof on right)
                roof_points = [(hx + house_h + 2, hy + house_w // 2), (hx + house_h - 2, hy), (hx + house_h - 2, hy + house_w)]
                pygame.draw.polygon(self.screen, COLORS['house_dark'], roof_points)
                pygame.draw.rect(self.screen, COLORS['house_green'], (hx, hy, house_h - 2, house_w))
            else:  # right
                spacing = (h - 8) // max(count, 1)
                hx = x + 2
                hy = y + 4 + i * spacing + (spacing - house_w) // 2
                # Draw house facing right (roof on left)
                roof_points = [(hx, hy + house_w // 2), (hx + 4, hy), (hx + 4, hy + house_w)]
                pygame.draw.polygon(self.screen, COLORS['house_dark'], roof_points)
                pygame.draw.rect(self.screen, COLORS['house_green'], (hx + 4, hy, house_h - 2, house_w))
    
    def _draw_hotel(self, x: int, y: int, w: int, h: int, side: str):
        """Draw red hotel on a property, centered on the color band."""
        hotel_w, hotel_h = 14, 10
        
        if side == 'bottom':
            hx = x + w // 2 - hotel_w // 2
            hy = y + 2
            pygame.draw.rect(self.screen, COLORS['hotel_dark'], (hx + 1, hy + 1, hotel_w, hotel_h))
            pygame.draw.rect(self.screen, COLORS['hotel_red'], (hx, hy, hotel_w, hotel_h))
        elif side == 'top':
            hx = x + w // 2 - hotel_w // 2
            hy = y + h - hotel_h - 2
            pygame.draw.rect(self.screen, COLORS['hotel_dark'], (hx + 1, hy + 1, hotel_w, hotel_h))
            pygame.draw.rect(self.screen, COLORS['hotel_red'], (hx, hy, hotel_w, hotel_h))
        elif side == 'left':
            hx = x + w - hotel_h - 2
            hy = y + h // 2 - hotel_w // 2
            pygame.draw.rect(self.screen, COLORS['hotel_dark'], (hx + 1, hy + 1, hotel_h, hotel_w))
            pygame.draw.rect(self.screen, COLORS['hotel_red'], (hx, hy, hotel_h, hotel_w))
        else:  # right
            hx = x + 2
            hy = y + h // 2 - hotel_w // 2
            pygame.draw.rect(self.screen, COLORS['hotel_dark'], (hx + 1, hy + 1, hotel_h, hotel_w))
            pygame.draw.rect(self.screen, COLORS['hotel_red'], (hx, hy, hotel_h, hotel_w))
        
        # Add 'H' label
        h_text = self.font_price.render("H", True, COLORS['white'])
        if side in ['bottom', 'top']:
            self.screen.blit(h_text, (hx + hotel_w // 2 - h_text.get_width() // 2, hy + hotel_h // 2 - h_text.get_height() // 2))
        else:
            self.screen.blit(h_text, (hx + hotel_h // 2 - h_text.get_width() // 2, hy + hotel_w // 2 - h_text.get_height() // 2))
    
    def _draw_ownership(self, tile, pos: Dict, state: GameState):
        """Draw ownership indicator as a small colored bar on the outer edge."""
        if tile.property_idx is None:
            return
        prop_state = state.properties[tile.property_idx]
        if prop_state.owner is None:
            return
        
        x, y, w, h = pos['x'], pos['y'], pos['w'], pos['h']
        side = pos['side']
        owner_color = COLORS['token_colors'][prop_state.owner % len(COLORS['token_colors'])]
        
        # Draw ownership bar on the outer edge (board perimeter side)
        bar_thickness = 4
        if side == 'bottom':
            pygame.draw.rect(self.screen, owner_color, (x + 2, y + h - bar_thickness - 1, w - 4, bar_thickness))
        elif side == 'top':
            pygame.draw.rect(self.screen, owner_color, (x + 2, y + 1, w - 4, bar_thickness))
        elif side == 'left':
            pygame.draw.rect(self.screen, owner_color, (x + 1, y + 2, bar_thickness, h - 4))
        elif side == 'right':
            pygame.draw.rect(self.screen, owner_color, (x + w - bar_thickness - 1, y + 2, bar_thickness, h - 4))
        
        # Draw mortgage X overlay
        if prop_state.mortgaged:
            pygame.draw.line(self.screen, COLORS['text_red'], (x + 3, y + 3), (x + w - 3, y + h - 3), 2)
            pygame.draw.line(self.screen, COLORS['text_red'], (x + w - 3, y + 3), (x + 3, y + h - 3), 2)
    
    def _draw_railroad_tile(self, idx: int, tile, pos: Dict, state: GameState):
        """Draw a railroad tile with a simple train icon."""
        x, y, w, h = pos['x'], pos['y'], pos['w'], pos['h']
        side = pos['side']
        cx, cy = x + w // 2, y + h // 2
        icon_color = COLORS['black']
        
        # Draw simple train icon - a locomotive silhouette
        def draw_train(tx, ty, horizontal=True):
            """Draw train at position, horizontal or vertical."""
            if horizontal:
                # Horizontal train
                pygame.draw.rect(self.screen, icon_color, (tx - 12, ty - 4, 18, 10))  # Body
                pygame.draw.rect(self.screen, icon_color, (tx + 6, ty - 8, 8, 12))    # Cab
                pygame.draw.rect(self.screen, icon_color, (tx - 8, ty - 10, 4, 6))    # Smokestack
                pygame.draw.circle(self.screen, icon_color, (tx - 6, ty + 8), 4)      # Wheel 1
                pygame.draw.circle(self.screen, icon_color, (tx + 6, ty + 8), 4)      # Wheel 2
            else:
                # Vertical train (rotated 90 degrees)
                pygame.draw.rect(self.screen, icon_color, (tx - 4, ty - 12, 10, 18))  # Body
                pygame.draw.rect(self.screen, icon_color, (tx - 8, ty + 6, 12, 8))    # Cab
                pygame.draw.rect(self.screen, icon_color, (tx - 10, ty - 8, 6, 4))    # Smokestack
                pygame.draw.circle(self.screen, icon_color, (tx + 8, ty - 6), 4)      # Wheel 1
                pygame.draw.circle(self.screen, icon_color, (tx + 8, ty + 6), 4)      # Wheel 2
        
        name = tile.name.replace(' Railroad', '').replace(' R.R.', '')[:12]
        text = self.font_price.render(name, True, COLORS['text_dark'])
        price = self.font_price.render("$200", True, COLORS['text_dark'])
        
        if side == 'bottom':
            draw_train(cx, y + 22, horizontal=True)
            self.screen.blit(text, (x + (w - text.get_width()) // 2, y + h - 24))
            self.screen.blit(price, (x + (w - price.get_width()) // 2, y + h - 12))
        elif side == 'top':
            draw_train(cx, y + h - 22, horizontal=True)
            self.screen.blit(text, (x + (w - text.get_width()) // 2, y + 4))
            self.screen.blit(price, (x + (w - price.get_width()) // 2, y + 16))
        elif side == 'left':
            draw_train(x + w - 22, cy, horizontal=False)
            angle = 90
            text_rot = pygame.transform.rotate(text, angle)
            price_rot = pygame.transform.rotate(price, angle)
            self.screen.blit(text_rot, (x + 4, cy - text_rot.get_height() // 2))
            self.screen.blit(price_rot, (x + 16, cy - price_rot.get_height() // 2))
        else:  # right
            draw_train(x + 22, cy, horizontal=False)
            angle = -90
            text_rot = pygame.transform.rotate(text, angle)
            price_rot = pygame.transform.rotate(price, angle)
            self.screen.blit(text_rot, (x + w - text_rot.get_width() - 4, cy - text_rot.get_height() // 2))
            self.screen.blit(price_rot, (x + w - price_rot.get_width() - 16, cy - price_rot.get_height() // 2))
        
        self._draw_ownership(tile, pos, state)
    
    def _draw_utility_tile(self, idx: int, tile, pos: Dict, state: GameState):
        """Draw utility tile with icons properly oriented per side."""
        x, y, w, h = pos['x'], pos['y'], pos['w'], pos['h']
        side = pos['side']
        cx, cy = x + w // 2, y + h // 2
        is_electric = 'Electric' in tile.name
        
        # Adjust icon center based on side to leave room for text
        if side == 'bottom':
            icon_cx, icon_cy = cx, y + 25
        elif side == 'top':
            icon_cx, icon_cy = cx, y + h - 25
        elif side == 'left':
            icon_cx, icon_cy = x + w - 25, cy
        else:  # right
            icon_cx, icon_cy = x + 25, cy
        
        if is_electric:
            # Draw simple lightbulb
            bulb_color = (255, 240, 100)
            pygame.draw.circle(self.screen, bulb_color, (icon_cx, icon_cy - 3), 10)
            pygame.draw.circle(self.screen, COLORS['black'], (icon_cx, icon_cy - 3), 10, 1)
            pygame.draw.rect(self.screen, COLORS['text_light'], (icon_cx - 4, icon_cy + 6, 8, 6))
            pygame.draw.rect(self.screen, COLORS['black'], (icon_cx - 4, icon_cy + 6, 8, 6), 1)
            # Rays
            for angle in range(0, 360, 60):
                rad = math.radians(angle)
                p1x = icon_cx + int(12 * math.cos(rad))
                p1y = icon_cy - 3 + int(12 * math.sin(rad))
                p2x = icon_cx + int(16 * math.cos(rad))
                p2y = icon_cy - 3 + int(16 * math.sin(rad))
                pygame.draw.line(self.screen, COLORS['black'], (p1x, p1y), (p2x, p2y), 1)
        else:
            # Draw simple water drop
            drop_color = (100, 150, 255)
            # Teardrop shape
            pygame.draw.circle(self.screen, drop_color, (icon_cx, icon_cy + 4), 8)
            pygame.draw.polygon(self.screen, drop_color, [(icon_cx, icon_cy - 10), (icon_cx - 8, icon_cy + 4), (icon_cx + 8, icon_cy + 4)])
            pygame.draw.circle(self.screen, COLORS['black'], (icon_cx, icon_cy + 4), 8, 1)
            pygame.draw.polygon(self.screen, COLORS['black'], [(icon_cx, icon_cy - 10), (icon_cx - 8, icon_cy + 4), (icon_cx + 8, icon_cy + 4)], 1)
        
        name = "ELECTRIC" if is_electric else "WATER"
        text = self.font_price.render(name, True, COLORS['text_dark'])
        price = self.font_price.render("$150", True, COLORS['text_dark'])
        
        if side == 'bottom':
            self.screen.blit(text, (x + (w - text.get_width()) // 2, y + h - 24))
            self.screen.blit(price, (x + (w - price.get_width()) // 2, y + h - 12))
        elif side == 'top':
            self.screen.blit(text, (x + (w - text.get_width()) // 2, y + 4))
            self.screen.blit(price, (x + (w - price.get_width()) // 2, y + 16))
        elif side == 'left':
            angle = 90
            text_rot = pygame.transform.rotate(text, angle)
            price_rot = pygame.transform.rotate(price, angle)
            self.screen.blit(text_rot, (x + 4, cy - text_rot.get_height() // 2))
            self.screen.blit(price_rot, (x + 16, cy - price_rot.get_height() // 2))
        else:  # right
            angle = -90
            text_rot = pygame.transform.rotate(text, angle)
            price_rot = pygame.transform.rotate(price, angle)
            self.screen.blit(text_rot, (x + w - text_rot.get_width() - 4, cy - text_rot.get_height() // 2))
            self.screen.blit(price_rot, (x + w - price_rot.get_width() - 16, cy - price_rot.get_height() // 2))
        
        self._draw_ownership(tile, pos, state)
    
    def _draw_chance_tile(self, idx: int, tile, pos: Dict):
        """Draw Chance tile with question mark oriented per side."""
        x, y, w, h = pos['x'], pos['y'], pos['w'], pos['h']
        side = pos['side']
        cx, cy = x + w // 2, y + h // 2
        
        # Create question mark
        font_q = pygame.font.SysFont('Georgia', 32, bold=True)
        q = font_q.render("?", True, COLORS['chance_orange'])
        text = self.font_price.render("CHANCE", True, COLORS['text_dark'])
        
        if side == 'bottom':
            q_rect = q.get_rect(center=(cx, cy - 5))
            self.screen.blit(q, q_rect)
            self.screen.blit(text, (x + (w - text.get_width()) // 2, y + h - 12))
        elif side == 'top':
            q_rect = q.get_rect(center=(cx, cy + 5))
            self.screen.blit(q, q_rect)
            self.screen.blit(text, (x + (w - text.get_width()) // 2, y + 3))
        elif side == 'left':
            q_rot = pygame.transform.rotate(q, 90)
            q_rect = q_rot.get_rect(center=(cx + 5, cy))
            self.screen.blit(q_rot, q_rect)
            text_rot = pygame.transform.rotate(text, 90)
            self.screen.blit(text_rot, (x + 3, cy - text_rot.get_height() // 2))
        else:  # right
            q_rot = pygame.transform.rotate(q, -90)
            q_rect = q_rot.get_rect(center=(cx - 5, cy))
            self.screen.blit(q_rot, q_rect)
            text_rot = pygame.transform.rotate(text, -90)
            self.screen.blit(text_rot, (x + w - text_rot.get_width() - 3, cy - text_rot.get_height() // 2))
    
    def _draw_community_tile(self, idx: int, tile, pos: Dict):
        """Draw Community Chest tile with chest icon oriented per side."""
        x, y, w, h = pos['x'], pos['y'], pos['w'], pos['h']
        side = pos['side']
        cx, cy = x + w // 2, y + h // 2
        
        # Determine icon position based on side
        if side == 'bottom':
            icon_cx, icon_cy = cx, y + 25
        elif side == 'top':
            icon_cx, icon_cy = cx, y + h - 25
        elif side == 'left':
            icon_cx, icon_cy = x + w - 25, cy
        else:  # right
            icon_cx, icon_cy = x + 25, cy
        
        # Draw simple chest icon
        chest_w, chest_h = 20, 14
        chest_x, chest_y = icon_cx - chest_w // 2, icon_cy - chest_h // 2
        
        # Chest body
        pygame.draw.rect(self.screen, (139, 69, 19), (chest_x, chest_y + 4, chest_w, chest_h - 4))
        pygame.draw.rect(self.screen, COLORS['black'], (chest_x, chest_y + 4, chest_w, chest_h - 4), 1)
        # Lid
        pygame.draw.arc(self.screen, (160, 82, 45), (chest_x, chest_y - 2, chest_w, 10), 0, 3.14, 4)
        # Lock
        pygame.draw.circle(self.screen, (255, 215, 0), (icon_cx, chest_y + 5), 2)
        
        # Text
        text = self.font_price.render("COMMUNITY", True, COLORS['text_dark'])
        text2 = self.font_price.render("CHEST", True, COLORS['text_dark'])
        
        if side == 'bottom':
            self.screen.blit(text, (x + (w - text.get_width()) // 2, y + h - 22))
            self.screen.blit(text2, (x + (w - text2.get_width()) // 2, y + h - 11))
        elif side == 'top':
            self.screen.blit(text, (x + (w - text.get_width()) // 2, y + 2))
            self.screen.blit(text2, (x + (w - text2.get_width()) // 2, y + 13))
        elif side == 'left':
            angle = 90
            t_rot = pygame.transform.rotate(text, angle)
            t2_rot = pygame.transform.rotate(text2, angle)
            self.screen.blit(t_rot, (x + 2, cy - t_rot.get_height() // 2))
            self.screen.blit(t2_rot, (x + 2 + t_rot.get_width(), cy - t2_rot.get_height() // 2))
        else:  # right
            angle = -90
            t_rot = pygame.transform.rotate(text, angle)
            t2_rot = pygame.transform.rotate(text2, angle)
            self.screen.blit(t2_rot, (x + w - t2_rot.get_width() - 2, cy - t2_rot.get_height() // 2))
            self.screen.blit(t_rot, (x + w - t_rot.get_width() - 2 - t2_rot.get_width(), cy - t_rot.get_height() // 2))
    
    def _draw_tax_tile(self, idx: int, tile, pos: Dict):
        """Draw Tax tile with icons - tax tiles are on bottom row in standard Monopoly."""
        x, y, w, h = pos['x'], pos['y'], pos['w'], pos['h']
        side = pos['side']
        is_income = 'Income' in tile.name
        cx, cy = x + w // 2, y + h // 2
        
        # Icon position
        icon_cy = cy if side in ['left', 'right'] else (y + 28 if side == 'bottom' else y + h - 28)
        icon_cx = cx if side in ['bottom', 'top'] else (x + w - 28 if side == 'left' else x + 28)
        
        if is_income:
            # Draw Diamond shape for Income Tax
            diamond_size = 10
            points = [(icon_cx, icon_cy - diamond_size), (icon_cx + diamond_size, icon_cy), 
                      (icon_cx, icon_cy + diamond_size), (icon_cx - diamond_size, icon_cy)]
            pygame.draw.polygon(self.screen, (50, 50, 50), points)
            pygame.draw.polygon(self.screen, COLORS['white'], points, 1)
        else:
            # Draw Ring for Luxury Tax
            pygame.draw.circle(self.screen, (255, 215, 0), (icon_cx, icon_cy), 8, 2)
            # Small diamond on top
            d_pts = [(icon_cx, icon_cy - 6), (icon_cx + 3, icon_cy - 3), (icon_cx, icon_cy), (icon_cx - 3, icon_cy - 3)]
            pygame.draw.polygon(self.screen, (200, 240, 255), d_pts)
        
        # Text
        label = "INCOME" if is_income else "LUXURY"
        amount = "$200" if is_income else "$100"
        text1 = self.font_price.render(label, True, COLORS['text_dark'])
        text2 = self.font_price.render("TAX", True, COLORS['text_dark'])
        text3 = self.font_price.render(amount, True, COLORS['text_dark'])
        
        if side == 'bottom':
            self.screen.blit(text1, (x + (w - text1.get_width()) // 2, y + 4))
            self.screen.blit(text2, (x + (w - text2.get_width()) // 2, y + 15))
            self.screen.blit(text3, (x + (w - text3.get_width()) // 2, y + h - 12))
        elif side == 'top':
            self.screen.blit(text1, (x + (w - text1.get_width()) // 2, y + h - 26))
            self.screen.blit(text2, (x + (w - text2.get_width()) // 2, y + h - 15))
            self.screen.blit(text3, (x + (w - text3.get_width()) // 2, y + 4))
        elif side == 'left':
            angle = 90
            t1_rot = pygame.transform.rotate(text1, angle)
            t2_rot = pygame.transform.rotate(text2, angle)
            t3_rot = pygame.transform.rotate(text3, angle)
            self.screen.blit(t1_rot, (x + 2, cy - t1_rot.get_height() // 2))
            self.screen.blit(t2_rot, (x + 13, cy - t2_rot.get_height() // 2))
            self.screen.blit(t3_rot, (x + w - t3_rot.get_width() - 2, cy - t3_rot.get_height() // 2))
        else:  # right
            angle = -90
            t1_rot = pygame.transform.rotate(text1, angle)
            t2_rot = pygame.transform.rotate(text2, angle)
            t3_rot = pygame.transform.rotate(text3, angle)
            self.screen.blit(t1_rot, (x + w - t1_rot.get_width() - 2, cy - t1_rot.get_height() // 2))
            self.screen.blit(t2_rot, (x + w - t2_rot.get_width() - 13, cy - t2_rot.get_height() // 2))
            self.screen.blit(t3_rot, (x + 2, cy - t3_rot.get_height() // 2))
    
    def _draw_players(self, state: GameState):
        """Draw player tokens at their positions."""
        position_players: Dict[int, List[int]] = {}
        for player in state.players:
            if player.status == PlayerStatus.ACTIVE:
                pos = player.position
                if pos not in position_players:
                    position_players[pos] = []
                position_players[pos].append(player.id)
        
        for position, player_ids in position_players.items():
            self._draw_tokens_at_position(position, player_ids, state)
    
    def _draw_tokens_at_position(self, position: int, player_ids: List[int], state: GameState):
        """Draw multiple player tokens at a position."""
        pos = self.tile_positions[position]
        x, y, w, h = pos['x'], pos['y'], pos['w'], pos['h']
        num = len(player_ids)
        
        for i, pid in enumerate(player_ids):
            if num == 1:
                ox, oy = w // 2, h // 2
            elif num == 2:
                ox, oy = w // 3 + (i % 2) * w // 3, h // 2
            else:
                ox, oy = w // 4 + (i % 2) * w // 2, h // 4 + (i // 2) * h // 2
            
            self._draw_token(x + ox, y + oy, pid, state.players[pid])
    
    def _draw_token(self, x: int, y: int, player_id: int, player):
        """Draw a single player token."""
        color = COLORS['token_colors'][player_id % len(COLORS['token_colors'])]
        size = 14
        shape = player_id % 4
        
        if shape == 0:
            pygame.draw.ellipse(self.screen, color, (x - size, y - size // 2, size * 2, size))
            pygame.draw.circle(self.screen, COLORS['black'], (x - size // 2, y + size // 2), 3)
            pygame.draw.circle(self.screen, COLORS['black'], (x + size // 2, y + size // 2), 3)
        elif shape == 1:
            pygame.draw.circle(self.screen, color, (x, y), size // 2)
            pygame.draw.ellipse(self.screen, color, (x - size // 2, y - size, size, size // 2))
            pygame.draw.circle(self.screen, COLORS['black'], (x - 3, y - 2), 2)
            pygame.draw.circle(self.screen, COLORS['black'], (x + 3, y - 2), 2)
        elif shape == 2:
            pygame.draw.rect(self.screen, color, (x - size // 2, y - size // 3, size, size // 2))
            pygame.draw.rect(self.screen, color, (x - size, y + size // 6, size * 2, size // 3))
        else:
            pygame.draw.rect(self.screen, color, (x - size // 3, y - size, size // 2, size))
            pygame.draw.rect(self.screen, color, (x - size // 3, y, size, size // 3))
        
        pygame.draw.circle(self.screen, COLORS['black'], (x, y), size + 2, 2)
        num = self.font_tiny.render(str(player_id), True, COLORS['white'])
        num_rect = num.get_rect(center=(x, y))
        self.screen.blit(num, num_rect)
        
        if player.jail_turns > 0:
            pygame.draw.circle(self.screen, COLORS['hotel_red'], (x + size, y - size), 6)
            j = self.font_price.render("J", True, COLORS['white'])
            self.screen.blit(j, (x + size - 3, y - size - 4))
    
    def _draw_stats_panel(self, state: GameState):
        """Draw the player statistics panel."""
        panel_x = self.board_margin + self.board_size + 20
        panel_y = self.board_margin
        panel_w = self.width - panel_x - 20
        panel_h = self.height - 2 * self.board_margin
        
        shadow_surf = pygame.Surface((panel_w + 4, panel_h + 4), pygame.SRCALPHA)
        shadow_surf.fill(COLORS['panel_shadow'])
        self.screen.blit(shadow_surf, (panel_x + 4, panel_y + 4))
        
        pygame.draw.rect(self.screen, COLORS['cream'], (panel_x, panel_y, panel_w, panel_h), border_radius=8)
        pygame.draw.rect(self.screen, COLORS['board_border'], (panel_x, panel_y, panel_w, panel_h), 2, border_radius=8)
        
        title = self.font_medium.render("GAME STATUS", True, COLORS['text_dark'])
        self.screen.blit(title, (panel_x + 15, panel_y + 12))
        
        y = panel_y + 45
        turn = self.font_small.render(f"Turn: {state.turn_number}", True, COLORS['text_dark'])
        self.screen.blit(turn, (panel_x + 15, y))
        
        if state.last_roll:
            dice_text = f"Dice: {state.last_roll[0]} + {state.last_roll[1]} = {sum(state.last_roll)}"
            dice = self.font_small.render(dice_text, True, COLORS['text_dark'])
            self.screen.blit(dice, (panel_x + 100, y))
        
        y += 25
        bank = self.font_tiny.render(f"Bank: {state.bank_houses_left} houses, {state.bank_hotels_left} hotels", True, COLORS['text_light'])
        self.screen.blit(bank, (panel_x + 15, y))
        
        y += 25
        pygame.draw.line(self.screen, COLORS['text_light'], (panel_x + 10, y), (panel_x + panel_w - 10, y), 1)
        y += 10
        
        card_h = 100
        card_w = panel_w - 30
        net_worths = [self._compute_player_net_worth(p, state) for p in state.players]
        max_net = max(1, max(net_worths))
        
        for i, player in enumerate(state.players):
            if y + card_h > panel_y + panel_h - 20:
                break
            self._draw_player_card(panel_x + 15, y, card_w, card_h, i, player, state, net_worths[i], max_net)
            y += card_h + 10
    
    def _draw_player_card(self, x: int, y: int, w: int, h: int, player_id: int, player, state: GameState,
                          net_worth: int, max_net_worth: int):
        """Draw a player info card."""
        color = COLORS['token_colors'][player_id % len(COLORS['token_colors'])]
        is_current = player_id == state.current_player
        is_bankrupt = player.status == PlayerStatus.BANKRUPT
        
        bg_color = (255, 240, 240) if is_bankrupt else (240, 255, 240) if is_current else COLORS['white']
        pygame.draw.rect(self.screen, bg_color, (x, y, w, h), border_radius=5)
        
        border_color = COLORS['text_green'] if is_current else COLORS['text_light']
        border_width = 3 if is_current else 1
        pygame.draw.rect(self.screen, border_color, (x, y, w, h), border_width, border_radius=5)
        
        pygame.draw.rect(self.screen, color, (x + 5, y + 5, 8, h - 10), border_radius=3)
        
        label = f"Player {player_id}"
        if player_id == 0:
            label += " (RL)"
        if is_bankrupt:
            label += " [BANKRUPT]"
        
        label_color = COLORS['text_red'] if is_bankrupt else COLORS['text_dark']
        name = self.font_small.render(label, True, label_color)
        self.screen.blit(name, (x + 20, y + 8))
        
        if is_bankrupt:
            return
        
        ty = y + 28
        cash = self.font_tiny.render(f"Cash: ${player.cash:,}", True, COLORS['text_dark'])
        self.screen.blit(cash, (x + 20, ty))
        
        nw = self.font_tiny.render(f"Net Worth: ${net_worth:,}", True, COLORS['text_green'])
        self.screen.blit(nw, (x + 120, ty))
        # Visual meter for quick comparison
        bar_w = w - 40
        bar_rect = pygame.Rect(x + 20, ty + 16, bar_w, 8)
        pygame.draw.rect(self.screen, (230, 230, 230), bar_rect, border_radius=4)
        fill_ratio = min(1.0, net_worth / max(1, max_net_worth))
        if fill_ratio > 0:
            filled = bar_rect.copy()
            filled.width = max(4, int(bar_w * fill_ratio))
            pygame.draw.rect(self.screen, COLORS['text_green'], filled, border_radius=4)
        
        ty += 26
        pos_name = self.board.get_tile(player.position).name
        pos_name = pos_name[:20] + ".." if len(pos_name) > 20 else pos_name
        pos = self.font_tiny.render(f"Position: {pos_name}", True, COLORS['text_light'])
        self.screen.blit(pos, (x + 20, ty))
        
        ty += 16
        props = self.font_tiny.render(f"Properties: {len(player.properties_owned)}", True, COLORS['text_dark'])
        self.screen.blit(props, (x + 20, ty))
        
        houses = sum(state.properties[p].houses_count for p in player.properties_owned if state.properties[p].houses_count < 5)
        hotels = sum(1 for p in player.properties_owned if state.properties[p].houses_count == 5)
        bld = self.font_tiny.render(f"H:{houses} Ho:{hotels}", True, COLORS['text_dark'])
        self.screen.blit(bld, (x + 120, ty))
        
        ty += 16
        if player.jail_turns > 0:
            jail = self.font_tiny.render(f"IN JAIL ({player.jail_turns} turns)", True, COLORS['text_red'])
            self.screen.blit(jail, (x + 20, ty))
        
        if player.get_out_of_jail_cards > 0:
            cards = self.font_tiny.render(f"Jail Cards: {player.get_out_of_jail_cards}", True, COLORS['text_green'])
            self.screen.blit(cards, (x + 120, ty))

    def _compute_player_net_worth(self, player, state: GameState) -> int:
        """Aggregate player liquid cash, property values, and structures."""
        total = player.cash
        for prop_idx in player.properties_owned:
            prop_spec = self.property_specs[prop_idx]
            prop_state = state.properties[prop_idx]
            if prop_state.mortgaged:
                total += prop_spec.mortgage_value
            else:
                total += prop_spec.price
                total += prop_state.houses_count * prop_spec.house_cost
        return max(0, total)
    
    def close(self):
        """Clean up pygame resources."""
        pygame.quit()
    
    def wait_for_key(self, duration_ms: Optional[int] = None):
        """Wait for key press or duration."""
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
                self.clock.tick(30)
