+++
title = '2024 Heuristic Agent'
date = 2023-10-02T17:31:02+02:00
draft = false
tags = ["ai", "minimax"]
+++
![screenshot](/20231008154014.png)
For our AI lab at ZHAW, we were tasked with programming a heuristic agent for the game [2048](https://play2048.co/). In this post, I will document my approach and what I learned.
## Initial Approach: Heuristics for Improved Play
### Finding the right heuristics
Our first goal was to design heuristics and an algorithm that could outperform a random-move agent. I started by implementing two key heuristics:

1. **Number of Empty Tiles:** This metric represents the number of available moves on the board, with more empty tiles indicating more possible actions.
2. **Corner Bonus:** Online resources suggested that keeping high-value tiles in the corners was an effective strategy in 2048. To reward this behavior, I created a weight matrix, assigning higher scores to tiles in the top-left corner. Fine-tuning these weights was a trial-and-error process, and I settled on the following matrix:

![matrix](/matrix.png)

I incorporated these heuristics into a utility function:
```python
def compute_utility(board):
    empty_tiles = len(board[board == 0]) * 10
    corner_score = calc_corner_score(board)
    return empty_tiles + corner_score

def calc_corner_score(board):
    return np.sum(board * POSITION_WEIGHTS)
```
This initial approach consistently outperformed the random-move agent, but I was eager to explore further possibilities.
## Implementing Minimax with Alpha-Beta Pruning
I turned to the classic Minimax algorithm, which I had encountered in my algorithms and data structures class. Although typically used for two-player games, I adapted it to 2048, where the random tile acted as the opposing player trying to minimize the AI's chances of winning.

Minimax involves evaluating possible moves by considering both maximizing and minimizing strategies, effectively creating a decision tree. To manage the vast number of possible moves, I introduced alpha-beta pruning, a technique that prunes branches of the tree that are unlikely to yield better results than already explored branches.

![tree](/20231008204958.png)
Here's my Python implementation:
```python
def minimax(board, depth, alpha, beta, maximizing):
    if depth == 0 or game_over(board):
        return compute_utility(board), -1
    move = None
    if maximizing:
        v = -np.inf
        for a in [UP, DOWN, LEFT, RIGHT]:
            board_to_evaluate = execute_move(a, board)
            if not board_equals(board, board_to_evaluate):
                v2, _ = minimax(board_to_evaluate, depth - 1, alpha, beta, False)
                if v2 > v:
                    move = a
                    v = v2
                alpha = max(alpha, v2)
                if beta <= alpha:
                    break
        return v, move
    else:
        v = np.inf
        available_tiles = find_empty_tiles(board)
        for tile in available_tiles:
            board_to_evaluate = board.copy()
            i, j = tile
            board_to_evaluate[i][j] = 4 if random.random() > 0.9 else 2
            v2, a = minimax(board_to_evaluate, depth - 1, alpha, beta, True)
            if v2 < v:
                move = a
                v = v2
            beta = min(beta, v2)
            if beta <= alpha:
                break
        return v, move
```
## Introducing the Third Heuristic: Smoothness
I noticed that the AI often struggled when tiles were spread out, making it challenging to merge them. To address this issue, I introduced the "smoothness" heuristic. This heuristic penalizes the AI if neighboring tiles have different scores, with a higher penalty for larger score differences. I calculated these differences both vertically and horizontally and subtracted the resulting "smoothness" score from the overall utility.
```python
def smoothness(board):
	smooth = 0
	row, col = len(board), len(board[0]) if len(board) > 0 else 0
	for r in board:
		for i in range(col - 1):
			smooth += abs(r[i] - r[i + 1])
	for j in range(row):
		for k in range(col - 1):
			smooth += abs(board[k][j] - board[k + 1][j])
	return smooth
```
Combining these heuristics with the minimax algorithm consistently enabled the AI to achieve scores exceeding 10,000 and occasionally even reach the coveted 2048 tile.
## Exploring the Depth-Efficiency Balance
![scatter](/scatter.png)
To determine the optimal balance between search depth and computational efficiency, I conducted several test runs with varying depths. I plotted the achieved scores against the time taken for each depth. The results showed a correlation between search depth and higher scores, but with each increase in depth, the computation time grew exponentially.

I also calculated the average score achieved per second by interpolating the data points. The findings indicated that a depth of 6 resulted in a significant performance boost. Going beyond a depth of 8 was not feasible due to the computational demands.
![interpolated](/interpolated.png)
## Strategies for Enhanced Effectiveness with the First Approach
Given the exponential increase in possible moves, simply increasing the depth of the minimax algorithm was not a practical solution. However, two alternative strategies could potentially improve its performance:

1. **Enhancing Computational Efficiency:** Exploring different data structures, such as encoding the entire board as a single 64-bit integer, as proposed in a [Stackoverflow discussion](https://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048), could significantly reduce move evaluation times, allowing for more computations per second.
2. **Refining Heuristics:** Improving heuristics to make smarter moves could lead to higher scores even with a shallower search depth.
### Conclusions from first approach
Overall, the results suggest that there is a trade-off between search depth and efficiency. While increasing the search depth can lead to higher scores, it also significantly increases the time it takes to compute. Therefore, it is important to find the optimal balance between search depth and efficiency to achieve the best performance. By improving the heuristics and using more efficient data structures, the minimax algorithm can be made to reach a higher score in the game 2048.
