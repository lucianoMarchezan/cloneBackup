import textwrap
from src.config import FUNCTION_NAME

SYSTEM_PROMPT_MINIMAL = f"""
You are a Python generation engine.
You produce Python code based on the information given. 
"""

SYSTEM_PROMPT_COMPLETE = f"""You are a careful Python refactoring engine.
You produce an alternative solution to the given function.
Rules:
- Output ONLY Python code in a single fenced block.
- Define exactly one function named `{FUNCTION_NAME}` with the correct signature for the tests.
- Keep the same external behavior, side effects.
- Do NOT hardcode any test data or specific URLs or values from tests.
- Keep I/O contract identical (same return types, shapes, and exceptions).
"""

MANDATORY_HINTS = """
- Do NOT output explanations, reasoning, or any text, **only valid Python code**
- Do NOT add default values to function parameters
- Do NOT add comments or multiline comments to the function 
- Do NOT add **print()** statements to your code
- If needed, library imports should be added before the function definition.
- Generate ONLY the code in a single ```python``` fenced block. 
- If you cannot generate code, output an empty function stub instead.
"""

REFACTORING = {
  "refac_1": ( # Algorithmic reimplementation
    "Generate an alternative solution using a different algorithmic strategy. You may use built-in functions, comprehensions, or alternative logic constructs."
 ),

  "refac_2": ( # Library exchange
    "Generate an alternative solution using different libraries or external APIs. None of the libraries from the example solution should be used."
 ),

  "refac_3": ( # Data representation shift
    "Generate an alternative solution using different data structures or representations. For example, replace lists with dictionaries, tuples, or sets where appropriate."
 ),

  "refac_4": ( # Stylistic and formatting variation
    "Generate an alternative solution with a different coding style and layout. Change indentation, whitespace, comment placement, and formatting. You should add docstrings or rearrange statements."
 ),

  "refac_5": ( # Side-effect isolation and purity
    "Generate an alternative solution that isolates side effects (e.g., I/O operations, state mutations) from pure computations. You should refactor the function to separate concerns."
 ),

  "refac_6": ( # Security and robustness enhancement
    "Generate an alternative solution that enhances security and robustness. You may add input validation, error handling, or logging."
 ),

  "refac_7": ( # Bad smells
    "Generate an alternative solution that introduces common 'bad smells' in code, such as long methods, duplicated code, or others."
 )
}

def build_user_prompt_test(strategy: str, description: str, tests_snippet: str, params: str, return_text: str, refacs: list[str]) -> str:
 return f"""

You will be shown:
1) A short description of the task.
2) An excerpt of the unit tests.

Your task:
- Generate a solution named `{FUNCTION_NAME}` with the following arguments: {params}. 
- Your solution must PASS all these unit tests:
```
{textwrap.dedent(tests_snippet).strip()}
```
- Your solution must correspond to this description: {description}.
- Your solution must return something based on this text: {return_text}.
- To ensure syntactic and structural differences on your solution, you MUST:
  {get_combined_refacs(refacs)}


- In addition, make sure that:
 {MANDATORY_HINTS}

{STRATEGIES[strategy]}
"""

def build_user_prompt_complete( strategy: str, original_body: str, description: str, tests_snippet: str, refacs: list[str]) -> str:
 return f"""

You will be shown:
1) A short description and a possible solution.
2) An excerpt of the unit tests.

Your task:
- Generate an alternative solution named `{FUNCTION_NAME}.
- Your solution must correspond to this description: {description}.
- To ensure syntactic and structural differences on your solution, you MUST:
{get_combined_refacs(refacs)}
- Your solution must PASS all these unit tests:
```
{textwrap.dedent(tests_snippet).strip()}
```
- Example solution (you must not use it):
```
{textwrap.dedent(original_body).strip()}
```

- In addition, make sure that:
 {MANDATORY_HINTS}
{STRATEGIES[strategy]}
"""

def build_user_prompt_code(strategy: str, original_body: str, description: str, params: str, return_text: str, refacs: list[str]) -> str:
 return f"""

You will be shown:
1) A short description.
2) The original function.

- Your solution must correspond to this description: {description}.
- Example solution (you must not use it):
```
{textwrap.dedent(original_body).strip()}
```
Your task:
- Generate an alternative solution named `{FUNCTION_NAME}` with the following arguments: {params}. 
- Make sure the syntax and structure of your solution are as different as possible from the Example solution BODY.
- Your solution must return something based on this text: {return_text}.
- To ensure syntactic and structural differences on your solution, you MUST:
{get_combined_refacs(refacs)}

- In addition, make sure that:
 {MANDATORY_HINTS}

{STRATEGIES[strategy]}
"""

def build_user_prompt_ast(strategy: str, gen_ast: str, description: str, params: str, return_text: str,refacs: list[str])-> str:
 return f"""
 You will be shown:
1) A short description.
2) The abstract syntax tree (AST) of an example solution.

- Your solution must correspond to this description: {description}.
- Make sure the AST your solution is as different as possible from this Example solution AST:
{gen_ast}.

Your task:
- Generate an alternative solution named `{FUNCTION_NAME}` with the following arguments: {params}. 
- Your solution must return something based on this text: {return_text}.
- To ensure syntactic and structural differences on your solution, you MUST:
{get_combined_refacs(refacs)}

- In addition, make sure that:
 {MANDATORY_HINTS}

{STRATEGIES[strategy]} 
 """

def build_user_prompt_reprompt( clone_code: str, params: str, return_text: str, tests_snippet: str, failing_tests: list[str]) -> str:
  return f"""
  You will be shown:
  1) A solution for a task that does not pass all tests.
  2) An excerpt of the unit tests.

  Your task:
  - Generate a new solution based on the given one that passes all the tests.
  - Generate a solution named `{FUNCTION_NAME}` with the following arguments: {params}. 
  - The solution must return something based on this text: {return_text}.
  - This is the current solution (modify as little as possible):
  ```
  {textwrap.dedent(clone_code).strip()}
  ```
  - Your solution must PASS all these unit tests:
  ```
  {textwrap.dedent(tests_snippet).strip()}
  ```
  Currently, these tests fail for this solution:
  {failing_tests}

  - In addition, make sure that:
  {MANDATORY_HINTS} 
  """

STRATEGIES = {
"zero-shot": """""",
"cot": """
Here are some examples of how to create alternative solutions in Python. 
Example:
```
# Original code: compute factorial correctly for all non-negative integers
def task_func(n):
 if n == 0 or n == 1:
 return 1
 return n * task_func(n - 1)
```
Now we generate an alternative solution by changing the implementation style. We have to make sure that:
- The I/O contract (input: integer n, output: factorial of n) is identical.
- The change is structural (different AST, control flow) but not semantic.
- Since the original used recursion, we can use an iterative approach (with a for loop).

Alternative solution:
```
# Alternative solution: checks whether a string s is equal to its reverse (s[::-1]). This is a palindrome check.
def task_func(n):
 result = 1
 for i in range(2, n + 1):
 result *= i
 return result
```
Here is another example.
```
# Original code:
def task_func(s):
 return s == s[::-1]
```
Now we generate an alternative solution by changing the implementation style. We have to make sure that:
- Instead of slicing with [::-1], Python also provides the reversed() built-in function.
- reversed(s) returns an iterator of the string in reverse order.
- Joining it back into a string with "".join(...) gives the reversed string.

Alternative solution:
```
# Alternative solution:
def task_func(s):
 return s == ''.join(reversed(s))
```

You may reason step-by-step internally to produce a correct solution.
DO NOT output your internal reasoning. Output ONLY the final code snippet (see instructions below).
"""
}

context_builders = { # add more builders to extend the supported contexts
 "ast": lambda **kwargs: (
  SYSTEM_PROMPT_MINIMAL,
  build_user_prompt_ast(kwargs["strategy"], kwargs["gen_ast"], kwargs["description"], kwargs["params"], kwargs["return_text"], kwargs["refacs"])
 ),
 "code": lambda **kwargs: (
  SYSTEM_PROMPT_MINIMAL,
  build_user_prompt_code(kwargs["strategy"], kwargs["original_body"], kwargs["description"], kwargs["params"], kwargs["return_text"], kwargs["refacs"])
 ),
 "complete": lambda **kwargs: ( # code + tests
  SYSTEM_PROMPT_COMPLETE,
  build_user_prompt_complete(kwargs["strategy"], kwargs["original_body"], kwargs["description"], kwargs["tests_snippet"], kwargs["refacs"])
 ),
"test": lambda **kwargs: (
    SYSTEM_PROMPT_MINIMAL,
  build_user_prompt_test(kwargs["strategy"], kwargs["description"], kwargs["tests_snippet"], kwargs["params"], kwargs["return_text"], kwargs["refacs"])
 )
 
} 
def get_combined_refacs(refacs: list[str]) -> str:
 combined_refacs = "\n".join([f"- {REFACTORING[key]}" for key in refacs if key in REFACTORING])
 return combined_refacs