import json
import random


def generate_math_jsonl(output_file, num_samples=10000):
    # Specialized Math Components
    latex_blocks = [
        "\\frac{x}{y} + \\sqrt{z^2}",
        "\\sum_{i=1}^{n} i^2",
        "\\int_{0}^{\\infty} e^{-x^2} dx",
        "\\lim_{x \\to 0} \\frac{\\sin x}{x} = 1",
        "\\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\epsilon_0}"
    ]

    logic_steps = [
        "[THOUGHT] Let's analyze the symmetry of the equation.",
        "[STEP] Substituting the values into the formula.",
        "[RESULT] Therefore, the solution set is \\emptyset.",
        "[VERIFY] Checking the boundary conditions at x=0."
    ]

    sentences = [
        "Suppose that $V$ is a vector space over $\\mathbb{R}$.",
        "Let \\alpha and \\beta be the roots of the quadratic equation.",
        "The derivative with respect to x is given by \\frac{d}{dx}.",
        "Using the Pythagorean theorem: a^2 + b^2 = c^2."
    ]

    with open(output_file, 'w', encoding='utf-8') as f:
        for _ in range(num_samples):
            # Mix natural language with LaTeX and Logic tokens
            content = f"{random.choice(sentences)} {random.choice(logic_steps)} " \
                      f"Consider the expression: {random.choice(latex_blocks)}."

            # Create the JSON object for this line
            entry = {
                "text": content,
                "metadata": {
                    "has_latex": True,
                    "language": "en",
                    "domain": random.choice(["calculus", "algebra", "physics"])
                }
            }

            # Write as a single line in the JSONL file
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


generate_math_jsonl("tokenizer_data.jsonl")
print("Successfully created tokenizer_data.jsonl with TeX tokens.")