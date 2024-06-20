import random

import numpy as np
import abc
import util
from game import Agent, Action


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """

        # Useful information you can extract from a GameState (game_state.py)

        successor_game_state = current_game_state.generate_successor(action=action)
        board = successor_game_state.board
        max_tile = successor_game_state.max_tile
        score = successor_game_state.score
        num_of_empty_cells = np.count_nonzero(board == 0)

        "*** YOUR CODE HERE ***"

        # Add to the evaluation to check if adjacent tiles are equal, which will be more likely for a higher score
        # later on.
        same_number_score = 0
        for row in board:
            for i in range(len(row) - 1):
                if row[i] == row[i + 1]:
                    same_number_score += 1

        for col in range(board.shape[1]):
            for i in range(board.shape[0] - 1):
                if board[i, col] == board[i + 1, col]:
                    same_number_score += 1

        # The evaluation will consist of the: score, number of empty cells (more empty cells means that the game is
        # probably in a better state), and the evaluation of the adjacent tiles equality.
        return score + max_tile + num_of_empty_cells + same_number_score


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        """*** YOUR CODE HERE ***"""
        value, action = self._Max_value(game_state, 0)
        return action

    def _Max_value(self, game_state, current_depth):
        """
        This function acts as the Max player.
        This player is the one who moves the tiles.
        :param game_state: The current game state of the board.
        :param current_depth: The current depth of the minimax algorithm.
        :return: The max value of the possible successor states according to the evaluation function.
        """
        if current_depth == self.depth or len(game_state.get_legal_actions(0)) == 0:
            return self.evaluation_function(game_state), Action.STOP
        value = float('-inf')
        best_move = None
        for action in game_state.get_legal_actions(0):
            successor_value, successor_action = self._Min_value(
                game_state.generate_successor(0, action), current_depth)
            if successor_value > value:
                value = successor_value
                best_move = action
        return value, best_move

    def _Min_value(self, game_state, current_depth):
        """
        This function acts as the Min player.
        This player is the one who places new tiles in each of his turns.
        :param game_state: The current state of the game.
        :param current_depth:  The current depth of the minimax algorithm.
        :return: The minimum value of the possible successor states according to the evaluation function.
        """
        value = float('inf')
        best_move = None
        for action in game_state.get_legal_actions(1):
            successor_value, successor_action = self._Max_value(
                game_state.generate_successor(1, action), current_depth + 1)
            if successor_value < value:
                value = successor_value
                best_move = action
        return value, best_move



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        """*** YOUR CODE HERE ***"""
        value, action = self._Max_value(game_state, 0, float('-inf'), float('inf'))
        return action

    def _Max_value(self, game_state, current_depth, alpha, beta):
        """
        This function acts as the Max player.
        This player is the one who moves the tiles.
        :param game_state: The current state of the game.
        :param current_depth: The current depth of the minimax algorithm.
        :param alpha: The current best Max value.
        :param beta: The current best Min value.
        :return: The max value of the possible successor states according to the evaluation function.
        """
        if current_depth == self.depth or len(game_state.get_legal_actions(0)) == 0:
            return self.evaluation_function(game_state), Action.STOP
        value = float('-inf')
        best_move = None
        for action in game_state.get_legal_actions(0):
            successor_value, successor_action = self._Min_value(
                game_state.generate_successor(0, action), current_depth, alpha, beta)
            if successor_value > value:
                value = successor_value
                best_move = action
                # Assign to alpha the value if it's larger:
                alpha = max(alpha, value)
            # If the value is larger than beta, the algorithm won't pick this path, so we end now.
            if value >= beta:
                return value, best_move
        return value, best_move

    def _Min_value(self, game_state, current_depth, alpha, beta):
        """
        This function acts as the Min player.
        This player is the one who places new tiles in each of his turns.
        :param game_state: The current state of the game.
        :param current_depth: The current depth of the minimax algorithm.
        :param alpha: The current best Max value.
        :param beta: The current best Min value.
        :return: The max value of the possible successor states according to the evaluation function.
        """
        value = float('inf')
        best_move = None
        for action in game_state.get_legal_actions(1):
            successor_value, successor_action = self._Max_value(
                game_state.generate_successor(1, action), current_depth + 1, alpha, beta)
            if successor_value < value:
                value = successor_value
                best_move = action
                # Assign to beta the value if it's smaller:
                beta = min(beta, value)
            # If the value is smaller than alpha, the algorithm won't pick this path, so we end now.
            if value <= alpha:
                return value, best_move
        return value, best_move



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        """*** YOUR CODE HERE ***"""
        value, action = self._Max_value(game_state, 0)
        return action

    def _Max_value(self, game_state, current_depth):
        """
        This function acts as the Max player.
        This player is the one who moves the tiles.
        :param game_state: The current state of the game.
        :param current_depth: The current depth of the minimax algorithm.
        :param alpha: The current best Max value.
        :param beta: The current best Min value.
        :return: The max value of the possible successor states according to the evaluation function.
        """
        if current_depth == self.depth or len(game_state.get_legal_actions(0)) == 0:
            return self.evaluation_function(game_state), Action.STOP
        value = float('-inf')
        best_move = None
        for action in game_state.get_legal_actions(0):
            successor_value = self._expectation_value(
                game_state.generate_successor(0, action), current_depth)
            if successor_value > value:
                value = successor_value
                best_move = action
        return value, best_move

    def _expectation_value(self, game_state, current_depth):
        """
        This function evaluates the expectation value of a game state.
        :param game_state: The game state to evaluate.
        :param current_depth: The depth of the evaluation.
        :return: The value of the expectation
        """
        value = 0
        legal_actions = game_state.get_legal_actions(1)
        for action in legal_actions:
            successor_value, successor_action = self._Max_value(
                game_state.generate_successor(1, action), current_depth + 1)
            value += successor_value
        return value / len(legal_actions)


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = better_evaluation_function
