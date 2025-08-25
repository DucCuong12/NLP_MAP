# Cấu hình
class Config:
    def __init__(self, model_name = "Qwen/Qwen3-14B"):
        self.model_name = model_name
        self.debug = False
        self.save_model = True  
        self.model_dir = "./reasoning_model_output"
        self.device = "cuda"
        
        self.use_lora = True
        self.lora_r = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.05
        self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        self.lora_modules_to_save = []
        
        self.epochs = 20
        self.max_length = 384
        self.batch_size = 4
        self.per_device_eval_batch_size = 1
        self.gradient_accumulation_steps = 1
        self.eval_frequency = 1000
    
        self.optimizer_name = "AdamW"
        self.lr = 1e-4
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0
        self.warmup_pct = 0.1