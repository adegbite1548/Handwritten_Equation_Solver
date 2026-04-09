import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import cv2
import numpy as np
import json
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms
import re
import multiprocessing # Changed to multiprocessing

from HMER_CAN.Watcher import DenseNetEncoder
from HMER_CAN.Parser import WAPDecoderCAN
from HMER_CAN.Weakly_Supervised_Counter import WSCM

from Math_Engine.MathEngine import MathEngine 


def worker_solve(latex_str, return_dict):
    try:
        engine = MathEngine()
        
        # 1. Sanitize inside the worker
        sanitized = engine.sanitize_latex(latex_str)
        return_dict['sanitized'] = sanitized # Save it so the UI can display it
        
        # 2. Parse inside the worker (in case the parser itself hangs)
        sympy_obj = engine.parse_equation(sanitized)
        
        # 3. Solve
        sol = engine.solve_equation(sympy_obj)
        return_dict['result'] = engine.format_to_latex(sol)
    except Exception as e:
        return_dict['error'] = str(e)


class MathRecognizerApp:
    def __init__(self, checkpoint_path, vocab_path='HMER_CAN/vocab.json'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading App on {self.device}...")

        with open(vocab_path, 'r') as f:
            self.vocab_dict = json.load(f)
        self.idx_to_token = {v: k for k, v in self.vocab_dict.items()}

        VOCAB_SIZE = 231
        EMBED_DIM = 256
        DECODER_DIM = 512
        ENCODER_DIM = 1024

        self.encoder = DenseNetEncoder().to(self.device)
        self.wscm = WSCM(encoder_dim=ENCODER_DIM, vocab_size=VOCAB_SIZE).to(self.device)
        self.decoder = WAPDecoderCAN(embed_dim=EMBED_DIM, decoder_dim=DECODER_DIM, vocab_size=VOCAB_SIZE).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.wscm.load_state_dict(checkpoint['wscm_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])

        self.encoder.eval()
        self.wscm.eval()
        self.decoder.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Keep one engine in the main class as a fallback
        self.engine = MathEngine()

    def preprocess_image(self, pil_image):
        
        img = np.array(pil_image)
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        # Use a global threshold (OTSU finds the perfect split automatically)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        clean_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

        target_h, target_w = 128, 728
        h, w = clean_img.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(clean_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255

        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return Image.fromarray(canvas)

    def translate_tokens(self, token_ids):
        words = []
        for token in token_ids:
            if token == 1: 
                break
            if token not in [0, 2]: 
                words.append(self.idx_to_token.get(token, '<UNK>'))

        latex =  " ".join(words)
    
        # Remove spaces ONLY between consecutive digits.
        fixed_latex = re.sub(r'(?<=\d)\s+(?=\d)', '', latex)
        
        return fixed_latex

    def predict(self, input_image):
        if input_image is None:
            return "Please upload an image.", "", ""

        processed_pil = self.preprocess_image(input_image)
        image_tensor = self.transform(processed_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            encoder_features = self.encoder(image_tensor)
            current_counts = self.wscm(encoder_features) 
            
            b, c, h, w = encoder_features.size()
            encoder_features = encoder_features.view(b, c, -1).permute(0, 2, 1)
            
            decoder_hidden = torch.zeros(1, self.decoder.decoder_dim).to(self.device)
            current_token = torch.tensor([0]).to(self.device) 
            
            coverage = torch.zeros(1, encoder_features.size(1)).to(self.device)
            predicted_ids = []

            for _ in range(150):
                predictions, decoder_hidden, _, coverage = self.decoder(
                    current_token, decoder_hidden, encoder_features, coverage, current_counts
                )
                
                top_token = predictions.argmax(1).item()
                predicted_ids.append(top_token)
                
                if top_token == 1: 
                    break
                    
                import torch.nn.functional as F
                pred_tensor = torch.tensor([top_token]).to(self.device)
                one_hot = F.one_hot(pred_tensor, num_classes=self.decoder.vocab_size).float()
                mask = (pred_tensor > 3).float().unsqueeze(1)
                current_counts = F.relu(current_counts - (one_hot * mask))
                current_token = pred_tensor
                
        # Get the raw prediction
        latex_str = self.translate_tokens(predicted_ids)
        
      
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        
        # Start the worker
        p = multiprocessing.Process(target=worker_solve, args=(latex_str, return_dict))
        p.start()
        
        # Block the UI for exactly 10 seconds waiting for the process
        p.join(20.0)
        
        if p.is_alive():
            # If 10 seconds passed and it's still running, KILL IT.
            p.terminate()
            p.join()
            final_solution_output = (
                "The Math Engine took too long to process this. Are you sure the equation format you entered is currently supported? "
                "If the equation recognised above is supported but different from what you wrote , try writing your equation again more clearly"
            )
            # Fallback sanitization just to display the rendered equation
            sanitized_latex_str = self.engine.sanitize_latex(latex_str)
        else:
            # Process finished naturally within 10 seconds
            sanitized_latex_str = return_dict.get('sanitized', self.engine.sanitize_latex(latex_str))
            
            if 'error' in return_dict:
                final_solution_output = f"Error solving equation:** {return_dict['error']}"
            else:
                final_solution_output = f"$${return_dict.get('result', '')}$$"

        return processed_pil, latex_str, f"$${sanitized_latex_str}$$", final_solution_output


# --- GRADIO WEB APP ---
if __name__ == "__main__":
    # Required to prevent recursive crashing when using multiprocessing
    multiprocessing.freeze_support() 
    
    app = MathRecognizerApp("HMER_CAN/checkpoints_CAN/hmer_can_checkpoint_epoch_9.pth")

    interface = gr.Interface(
        fn=app.predict,
        inputs=gr.Image(type="pil", label="Upload Math Image"),
        outputs=[
            gr.Image(type="pil", label="What the AI actually sees"),
            gr.Textbox(label="Raw Model Output (Tokens)"),
            gr.Markdown(label="Rendered Equation"),
            gr.Markdown(label="Solved Answer (SymPy)") 
        ],
        title="Handwritten Equation Solver",
        description = (
            "<u>**Instructions:**</u>\n\n"
            "Upload a handwritten equation.\n\n"
            "Currently supported are algebraic equations, transcendental equations, summation and product expressions and equations, definite and indefinite integrals as well as derivatives.\n\n"
            "For the best results, the image should be written on a tablet with a white background and cropped and zoomed in as much as possible. It will also be beneficial to space out your equation as much as possible."
        ),
        flagging_mode="never"
    )

    interface.queue().launch(share=False)