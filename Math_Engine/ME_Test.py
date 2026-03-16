from MathEngine import MathEngine

ME = MathEngine()


test_cases_sanitize = {
        
        "Standard Spaced": "s i n ( x )",
        
    
        "Logarithmic": "l n l o g",
        
       
        "No brackets": "s i n x",
        
        "Hyper-regular": "s i n x + s i n h = 0",
        
        "Embedded Words": "b a s i n = 5",
        
        "Long Equation": "s i n ^ 2 x + c s c x + l n x - s i n h x = 5 + t a n h x + t a n x",

        "1st order ODE" : "2\\frac{d y}{d x} + y ( x ) = 0",

        "2nd order ODE" : "2\\frac{d ^ { 2 } y}{d x ^ 2} + \\frac{d y}{d x} + 2 = 0"
    }

print(f"{'Test Case':<20} | {'Raw Input':<15} | {'Sanitized Output'}")
print("-" * 60)
for name, raw in test_cases_sanitize.items():
    sanitized = ME.sanitize_latex(raw)
    print(f"{name:<15} | {raw:<30} | {sanitized}")



test_cases_Solver = {

    "Regular Equation": "x + 2 = 5",

    "Summation Problem": "2 \\sum _ { i = 1 } ^ 10 2 i",

    "Summation with letter": "\\sum_{ i = 1} ^ n 2 ^ i",

    "Basic Arithmetic" : "2 + 2",

    "Quadratic Equation": "x ^ 2 + 2 x + 3",

    "Quintic Equation": "x ^ 5 + 1 = 10",

    "Transcendental Equation": "2 l n x + 1 = 0",

    "Product Problem": "\\prod _ { i = 1 } ^ 10 2 ^ i",

    "Derivative Problem": "\\frac { d } { d x } c o s h x",

    "Summation and Product Problem": "2 \\sum _ { i = 1 } ^ 10 2 i - \\prod _ { i = 1 } ^ 10 2 ^ i",

    "Definite Integral": "\\int _ { 0 } ^ { 2 \pi } s i n x",

    "Indefinite Integral" : "\\int s e c x",


}

print(f"{'Test Case':<20} | {'Problem':<15} | {'Answer'}")
print("-" * 60)
answers = {}
for name, problem in test_cases_Solver.items():
    sanitized = ME.sanitize_latex(problem)
    answer = ME.solve_equation(ME.parse_equation(sanitized))
    answers[name] = answer
    print(f"{name:<15} | {problem:<30} | {answer}")


print(f"{'Test Case':<20} | {'Answer':<15} | {'Latex'}")
print("-" * 60)
for name, answer in test_cases_Solver.items():
    latex = ME.format_to_latex(answers[name])
    print(f"{name:<15} | {answer:<30} | {latex}")