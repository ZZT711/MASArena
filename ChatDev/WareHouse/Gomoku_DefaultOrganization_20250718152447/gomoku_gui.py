'''
This module contains the GUI for the Gomoku game.
'''
import tkinter as tk
from tkinter import messagebox
class GomokuGUI:
    def __init__(self, master, game):
        self.master = master
        self.game = game
        self.canvas = tk.Canvas(master, width=600, height=600)
        self.canvas.pack()
        self.status_label = tk.Label(master, text="Current Player: X")
        self.status_label.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self.draw_board()
    def draw_board(self):
        self.canvas.delete("all")
        for i in range(15):
            self.canvas.create_line(40 * i, 0, 40 * i, 600)
            self.canvas.create_line(0, 40 * i, 600, 40 * i)
        for x in range(15):
            for y in range(15):
                if self.game.board[x][y] == "X":
                    self.canvas.create_text(20 + 40 * y, 20 + 40 * x, text="X", font=("Arial", 24))
                elif self.game.board[x][y] == "O":
                    self.canvas.create_text(20 + 40 * y, 20 + 40 * x, text="O", font=("Arial", 24))
    def on_click(self, event):
        x, y = event.y // 40, event.x // 40
        self.game.make_move(x, y)
        self.draw_board()
        if self.game.winner:
            self.update_status(f"Player {self.game.winner} wins!")
            messagebox.showinfo("Game Over", f"Player {self.game.winner} wins!")
            self.game.reset_game()
            self.draw_board()
    def update_status(self, message):
        self.status_label.config(text=message)