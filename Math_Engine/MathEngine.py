import re
import sympy
from sympy import solve
from latex2sympy2 import latex2sympy

class MathEngine:
    def __init__(self):
        # Possible trigonometric and logarithmic functions that can appear. 
        self.trig_funcs = ['sinh', 'cosh', 'tanh', 'sin', 'cos', 'tan', 'sec', 'csc', 'cot']
        self.log_funcs = ['log', 'ln']
        self.all_funcs = self.trig_funcs + self.log_funcs

    def sanitize_latex(self, raw_latex):
        clean_str = raw_latex

        for func in self.all_funcs:
            pattern = r'\s+'.join(list(func))
            clean_str = re.sub(pattern, rf'\\{func}', clean_str)
        
        # Standardize remaining inline Leibniz formatting (for integrals and derivatives)
        clean_str = re.sub(r'd\s+([a-zA-Z])', r'd\1', clean_str)

        # Matrix sanitization
        clean_str = re.sub(r'm\s+a\s+t\s+r\s+i\s+x', 'matrix', clean_str)
        
        return clean_str.strip()
    
    def format_to_latex(self, result):
        # 1. If it's an error message (string), just return the text
        if isinstance(result, str):
            return result
        
        # 2. If it's a list of multiple answers (e.g., from an algebraic quadratic)
        if isinstance(result, list):
            # Convert each answer to LaTeX and join them with a comma
            return r", ".join([sympy.latex(res) for res in result])
        
        # 3. For single objects (Equations, numbers, derivatives, integrals)
        return sympy.latex(result)
    
    def parse_equation(self, sanitized_latex):

        try:

            return latex2sympy(sanitized_latex)

        except Exception as e:

            print(f"Latex2SymPy Parsing Error: {e}")

            return None

    

    def solve_equation(self, sympy_obj):
        if sympy_obj is None:
            return "\\text{Cannot solve: Invalid parsing.}"

       
        # If a list somehow makes it here, extract the first math element
        if isinstance(sympy_obj, list):
            if len(sympy_obj) > 0:
                sympy_obj = sympy_obj[0]
            else:
                return "Cannot solve: Empty input."
                
        # Also, if it evaluates to a raw Python number (int/float), convert it to SymPy Integer/Float
        if isinstance(sympy_obj, (int, float)):
            sympy_obj = sympy.sympify(sympy_obj)
        

        # 1. CALCULUS, SUMMATION, PRODUCT routing (Integrals and Derivatives only for calculus)
        if sympy_obj.has(sympy.Integral, sympy.Derivative, sympy.Limit, sympy.Sum, sympy.Product):
            try:
                evaluated = sympy_obj.doit()
                # If everything collapsed into pure numbers after evaluating, simplify it down
                if hasattr(evaluated, 'free_symbols') and not evaluated.free_symbols:
                    return sympy.simplify(evaluated)
                return evaluated
            except Exception as e:
                return f"Calculus Evaluation failed: {e}"

        # 2. BASIC ARITHMETIC ROUTING
        if not isinstance(sympy_obj, sympy.Eq):
            if hasattr(sympy_obj, 'free_symbols') and not sympy_obj.free_symbols:
                # sympy.simplify() forces unevaluated expressions like '2 + 2' to become '4'
                return sympy.simplify(sympy_obj)

        # 3. FORMATTING FOR SOLVERS
        if not isinstance(sympy_obj, sympy.Eq):
            sympy_obj = sympy.Eq(sympy_obj, 0)

        

        # 4. ALGEBRAIC ROUTING
        try:
            return solve(sympy_obj)
        except Exception as e:
            return f"Solver failed: {e}"


