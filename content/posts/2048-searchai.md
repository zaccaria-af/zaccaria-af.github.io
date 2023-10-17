+++
title = '2048 Part 2 - Expectimax'
date = 2023-10-17T15:03:02+02:00
draft = false
tags = ["ai", "minimax"]
+++
## Writing the algorithm
Our second task in a university assignment was to build an agent for 2048, and this time, by using Expectimax. This algorithm is similar to the Minimax algorithm I used in the first task but introduces a new element – the chance node. In Minimax, we work with maximizer and minimizer nodes, but in Expectimax, we deal with a maximizer and a chance node. The chance node calculates the expected value of various stochastic outcomes, such as randomly placing either a _3_ or a _4_ on an available empty space in 2048, with probabilities of _0.9_ and _0.1_, respectively.

The key distinction lies in the way Expectimax considers not only the worst-case scenario but the expected outcome. This leads the algorithm to make "riskier" moves based on expected values.

```python
def expectimax(board, depth, max_depth, node):
    move = -1
    if depth == max_depth or game_over(board):
        return evaluate_position(board), move
    if node == MAX:
        v = -np.inf
        for a in [UP, DOWN, LEFT, RIGHT]:
            board_to_evaluate = execute_move(a, board)
            if not board_equals(board, board_to_evaluate):
                v2, _ = expectimax(board_to_evaluate, depth, max_depth, CHANCE)
                if v2 > v:
                    move = a
                    v = v2
        return v, move
    else:
        v = 0
        available_tiles = find_empty_tiles(board)
        num_tiles = len(available_tiles)
        for tile in available_tiles:
            board_to_evaluate = board.copy()
            i, j = tile
            board_to_evaluate[i][j] = 2
            v += expectimax(board_to_evaluate, depth+1, max_depth, MAX)[0] * PROB_2
            board_to_evaluate[i][j] = 4
            v += expectimax(board_to_evaluate, depth+1, max_depth, MAX)[0] * PROB_4
        return v / num_tiles, move
```

## Finding the right weights
### Assessing performance
For heuristics, I initially used the same set of metrics that served me well in the Minimax approach:

- Number of empty tiles
- Encouraging high-value tiles to gather in the corners (using a weight matrix)
- Promoting tile smoothness, rewarding minimal differences between neighboring tiles

However, these heuristics didn't perform as expected with Expectimax, prompting me to introduce weighting factors, represented as _w0_, _w1_, and _w2_. The branching factor in Expectimax is significantly higher, necessitating a reduction in the search depth. For instance, I had to limit it to just 2.

To fine-tune the weights, I wrote a script that ran the AI multiple times in different weight configurations and then exported the results into a CSV file. The goal was to optimize the weights for superior performance.

![first-plot](/0_2_plot.png)

In the plot we can see that the weight setting *w0=board.max(), w1=1, w2=1* performed the best. I noticed in the first run that the higher the tiles got, the less the AI would care about empty spaces. That is why I tried to calculate this bonus dynamically by multiplying it with the highest tile currently on the board.
### Dynamic Search Depth
A conversation with fellow students led to a clever addition to the algorithm. I increased the search depth when the board neared full capacity. This move aimed to leverage lower computational demands when the board is less cluttered, allowing us to search deeper without compromising performance.

```python
    if depth == 0 and len(find_empty_tiles(board)) <= 1:
        max_depth += 1
```

I did a few more runs and plotted the results:

![second-plot](/5_7_plot.png)
### Introducing the Snake Pattern
Through collaborative discussions, I realized that a snake pattern in the weight matrix might be more effective than the linear pattern I initially used:
```python
POSITION_WEIGHTS = [[16 ** 4, 15 ** 4, 14 ** 4, 13 ** 4],
                       [9 ** 4, 10 ** 4, 11 ** 4, 12 ** 4],
                       [8 ** 4, 7 ** 4, 6 ** 4, 5 ** 4],
                       [1 ** 4, 2 ** 4, 3 ** 4, 4 ** 4]]
```
Several variations of this pattern were tested, and the one above proved to be the most effective.
Next, I experimented with different weight combinations using this snake pattern, running the AI multiple times (100 times for better accuracy). The final choice for weight values that delivered the best performance was _w0=board.max(), w1=board.max(), w2=1_ in combination with the snake pattern.
![boxplot](/boxplot.png)
## Further improvements
Although I have invested considerable time and effort into this assignment, there remain several avenues for potential enhancement. Here are some ideas for future exploration:
- **Binary Board Representation:** Representing the board state using bytes instead of numpy matrices could dramatically improve efficiency. Bitwise operations are highly efficient, enabling deeper searches.
- **Monte Carlo Search:** Incorporating Monte Carlo search techniques could introduce greater adaptability and strategy into the AI's decision-making process.
- **Machine Learning:** Utilizing machine learning to determine optimal weight values (_w0_, _w1_, _w2_) could potentially lead to superior performance. Reinforcement learning or other machine learning techniques could be explored in this context.
In conclusion, my journey to master 2048 with Expectimax has been both challenging and rewarding. Despite some setbacks, I've achieved significant progress in building a capable AI agent. I'm excited to see where this journey in artificial intelligence will lead.