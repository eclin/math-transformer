import torch
import numpy as np
from typing import List, Tuple


DATA_SIZE = 100000

MIN_INT = -99
MAX_INT = 99
operations = ["+", "-", "*"]

TOKENS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', ' ', '-', '+', '*', '<PAD>', '<BOS>', '<END>']

def equation_to_token_ids(row, max_length: int, with_padding=True) -> torch.tensor:
    token_ids = [TOKENS.index(c) for c in row[0]]
    if with_padding:
        token_ids += [TOKENS.index('<PAD>')] * (max_length - len(token_ids))
    return torch.LongTensor(token_ids)

def answer_to_token_ids(row, max_length: int, with_padding=True) -> torch.tensor:
    token_ids = [TOKENS.index('<BOS>')] + [TOKENS.index(c) for c in row[0]]
    if with_padding:
        token_ids += [TOKENS.index('<PAD>')] * (max_length - len(token_ids) - 1)
    token_ids += [TOKENS.index('<END>')]
    return torch.LongTensor(token_ids)

class MathData():
    def __init__(
        self,
        tokens: List[int],
        num_tokens: int,
        max_equation_length: int,
        max_answer_length: int,
        train_equations: torch.tensor,
        train_answers: torch.tensor,
        test_equations: torch.tensor,
        test_answers: torch.tensor
    ):
        self.tokens = tokens
        self.num_tokens = num_tokens
        self.max_equation_length = max_equation_length
        self.max_answer_length = max_answer_length

        self.train_equations = train_equations
        self.train_answers = train_answers
        self.test_equations = test_equations
        self.test_answers = test_answers

class DataGenerator():
    # Generate different strings representing math problems
    def __init__(self):
        pass

    def create_equation_answer_string(self, row) -> Tuple[str, str]:
        op = np.random.choice(operations)
        result = 0
        if op == "+":
            result = row[0] + row[1]
        elif op == "-":
            result = row[0] - row[1]
        elif op == "*":
            result = row[0] * row[1]
        eq = f"{row[0]} {op} {row[1]}"
        ans = f"{result}"

        return eq, ans

    def generate_data(self, verbose: bool=False) -> MathData:
        data_arr = np.random.randint(MIN_INT, MAX_INT + 1, size=(int(DATA_SIZE), 2))

        # Generate equations
        data_arr = [self.create_equation_answer_string(row) for row in data_arr]
        equations = [row[0] for row in data_arr]
        answers = [row[1] for row in data_arr]

        max_equation_length = 0
        for i in range(len(equations)):
            max_equation_length = max(max_equation_length, len(equations[i]))
        max_answer_length = 0
        for i in range(len(answers)):
            max_answer_length = max(max_answer_length, len(answers[i]))
        max_answer_length += 2  # For <BOS> and final <PAD> tokens


        equation_tokens = torch.LongTensor(np.apply_along_axis(equation_to_token_ids, 1, np.expand_dims(equations, axis=1), max_length=max_equation_length, with_padding=True))
        answer_tokens = torch.LongTensor(np.apply_along_axis(answer_to_token_ids, 1, np.expand_dims(answers, axis=1), max_length=max_answer_length, with_padding=True))

        split_idx = int(DATA_SIZE * 0.9)
        train_equations = equation_tokens[:split_idx]
        test_equations = equation_tokens[split_idx:]

        train_answers = answer_tokens[:split_idx]
        test_answers = answer_tokens[split_idx:]

        num_tokens = len(TOKENS)

        if verbose:
            print(f"Equations: {len(equations)}")
            print(f"Answers: {len(answers)}")
            print(f"Max Equation Length: {max_equation_length}")
            print(f"Max Answer Length: {max_answer_length}")

            print(equations[:10])
            print(answers[:10])

            print(f"Tokens: {TOKENS}")
            print(f"Number of Tokens: {num_tokens}")
            print(f"Equation Tokens Shape: {equation_tokens.shape}")
            print(f"Example equation tokens: {equation_tokens[0:10]}")
            print(f"Answer Tokens Shape: {answer_tokens.shape}")
            print(f"Example answer tokens: {answer_tokens[0:10]}")

        math_data = MathData(TOKENS, num_tokens, max_equation_length, max_answer_length, train_equations, train_answers, test_equations, test_answers)
        return math_data
