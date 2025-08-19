import tkinter as tk
import random

class TicTacToe:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic-Tac-Toe Setup")

        self.themes = {
            "Classic": {"bg":"#ffffff","btn_bg":"#f0f8ff","x_color":"#0b66ff","o_color":"#d92b2b","highlight":"#b7ffb7","press":"#d9ecff"},
            "Dark Mode": {"bg":"#2b2b2b","btn_bg":"#3a3a3a","x_color":"#00e6ff","o_color":"#ffd500","highlight":"#2f8f6f","press":"#4a4a4a"},
            "Pastel": {"bg":"#fff8f2","btn_bg":"#fff1e6","x_color":"#ff6fa3","o_color":"#1fbfa6","highlight":"#fff1a8","press":"#ffe9d9"},
            "High Contrast": {"bg":"#000000","btn_bg":"#000000","x_color":"#39ff14","o_color":"#ff6a00","highlight":"#ffd400","press":"#222222"},
            "Ocean": {"bg":"#e8f6ff","btn_bg":"#d7eefc","x_color":"#073b5a","o_color":"#ff7f66","highlight":"#a6ddff","press":"#cde7ff"},
            "Forest": {"bg":"#eef7ec","btn_bg":"#e2f0df","x_color":"#0b6b3a","o_color":"#7a4200","highlight":"#c8f7b2","press":"#dff0d9"}
        }

        self.selected_theme = tk.StringVar(value="Classic")
        self.first_var = tk.StringVar(value="Human")
        self.sym_var = tk.StringVar(value="X")

        self.build_setup()

    def build_setup(self):
        tk.Label(self.root, text="Choose Theme:").pack()
        for theme in self.themes:
            tk.Radiobutton(self.root, text=theme, variable=self.selected_theme, value=theme).pack(anchor="w")

        tk.Label(self.root, text="Who plays first?").pack()
        tk.Radiobutton(self.root, text="Human", variable=self.first_var, value="Human").pack(anchor="w")
        tk.Radiobutton(self.root, text="AI", variable=self.first_var, value="AI").pack(anchor="w")

        tk.Label(self.root, text="Your Symbol:").pack()
        tk.Radiobutton(self.root, text="X", variable=self.sym_var, value="X").pack(anchor="w")
        tk.Radiobutton(self.root, text="O", variable=self.sym_var, value="O").pack(anchor="w")

        tk.Button(self.root, text="Start Game", command=self.start_game).pack(pady=10)

    def start_game(self):
        self.theme = self.themes[self.selected_theme.get()]
        self.player = self.sym_var.get()
        self.ai = "O" if self.player == "X" else "X"

        for widget in self.root.winfo_children():
            widget.destroy()

        self.root.configure(bg=self.theme["bg"])

        self.board = [""] * 9
        self.buttons = []
        self.game_over = False

        frame = tk.Frame(self.root, bg=self.theme["bg"])
        frame.pack()

        for i in range(9):
            btn = tk.Button(frame, text="", font=("Arial", 24, "bold"), width=5, height=2,
                            bg=self.theme["btn_bg"], command=lambda i=i: self.on_click(i))
            btn.grid(row=i//3, column=i%3, padx=5, pady=5)
            self.buttons.append(btn)

        tk.Button(self.root, text="Restart", command=self.restart, bg=self.theme["btn_bg"]).pack(pady=10)

        if self.first_var.get() == "AI":
            self.root.after(300, self.ai_move)

    def restart(self):
        self.start_game()

    def on_click(self, index):
        if self.board[index] == "" and not self.game_over:
            self.buttons[index].config(text=self.player, fg=self.theme["x_color"] if self.player == "X" else self.theme["o_color"],
                                       bg=self.theme["press"])
            self.board[index] = self.player

            if self.check_winner(self.player):
                self.end_game(self.player)
                return
            elif "" not in self.board:
                self.end_game("Draw")
                return

            self.root.after(300, self.ai_move)

    def ai_move(self):
        if self.game_over:
            return

        best_score = -float("inf")
        best_move = None
        for i in range(9):
            if self.board[i] == "":
                self.board[i] = self.ai
                score = self.minimax(False, -float("inf"), float("inf"))
                self.board[i] = ""
                if score > best_score:
                    best_score = score
                    best_move = i

        if best_move is not None:
            self.buttons[best_move].config(text=self.ai, fg=self.theme["x_color"] if self.ai == "X" else self.theme["o_color"],
                                           bg=self.theme["press"])
            self.board[best_move] = self.ai

            if self.check_winner(self.ai):
                self.end_game(self.ai)
            elif "" not in self.board:
                self.end_game("Draw")

    def minimax(self, is_maximizing, alpha, beta):
        if self.check_winner(self.ai):
            return 1
        elif self.check_winner(self.player):
            return -1
        elif "" not in self.board:
            return 0

        if is_maximizing:
            max_eval = -float("inf")
            for i in range(9):
                if self.board[i] == "":
                    self.board[i] = self.ai
                    eval = self.minimax(False, alpha, beta)
                    self.board[i] = ""
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            return max_eval
        else:
            min_eval = float("inf")
            for i in range(9):
                if self.board[i] == "":
                    self.board[i] = self.player
                    eval = self.minimax(True, alpha, beta)
                    self.board[i] = ""
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
            return min_eval

    def check_winner(self, player):
        win_combos = [(0,1,2),(3,4,5),(6,7,8),
                      (0,3,6),(1,4,7),(2,5,8),
                      (0,4,8),(2,4,6)]
        for a, b, c in win_combos:
            if self.board[a] == self.board[b] == self.board[c] == player:
                self.buttons[a].config(bg=self.theme["highlight"])
                self.buttons[b].config(bg=self.theme["highlight"])
                self.buttons[c].config(bg=self.theme["highlight"])
                return True
        return False

    def end_game(self, winner):
        self.game_over = True
        if winner != "Draw":
            print(f"{winner} wins!")
        else:
            print("It's a draw!")

root = tk.Tk()
game = TicTacToe(root)
root.mainloop()

