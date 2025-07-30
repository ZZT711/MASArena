'''
Main entry point for the Gomoku game application.
'''
from gomoku_game import GomokuGame
from gomoku_gui import GomokuGUI
import tkinter as tk
def main():
    root = tk.Tk()
    root.title("Gomoku Game")
    game = GomokuGame()
    gui = GomokuGUI(root, game)
    root.mainloop()
if __name__ == "__main__":
    main()