class LoRAQATLinearADC(nn.Module):
    """
    ADC-LoRA: Low-Rank Adaptation for ADC quantization (Paper Section 3.4).
    
    Implements Equation 11: Y = QA(Qx(X)Qw(W + AB))
    
    Where:
    - W is the frozen pretrained weight
    - A ∈ R^{out_features × r}, B ∈ R^{r × in_features} are learnable low-rank matrices
    - r << min(out_features, in_features) is the rank
    - The effective weight (W + scaling * A @ B) is quantized through the ADC pipeline
    
    This reduces trainable parameters dramatically while maintaining ADC quantization.
    """
    
    def __init__(self,
                 base_layer: QATLinearADC,
                 r: int = 8,
                 alpha: float = 16.0,
                 dropout: float = 0.0):
        """
        Args:
            base_layer: The QATLinearADC layer to wrap with LoRA
            r: LoRA rank (dimension of low-rank matrices)
            alpha: LoRA scaling factor (scaling = alpha / r)
            dropout: Dropout rate applied to LoRA path
        """
        super().__init__()
        
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # Freeze base layer weights
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False
        
        # LoRA parameters: A (out_features, r), B (r, in_features)
        # Following standard LoRA: A initialized with Kaiming, B initialized with zeros
        # This ensures initial output is identical to base layer (A @ B = 0)
        self.lora_A = nn.Parameter(torch.zeros(out_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, in_features))
        
        # Initialize A with Kaiming uniform (standard LoRA initialization)
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        # B stays zero so initial A @ B = 0
        
        # Optional dropout for regularization
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Store dimensions for reference
        self.in_features = in_features
        self.out_features = out_features
    
    def set_quantizer_mode(self, mode: str):
        """Delegate to base layer"""
        self.base_layer.set_quantizer_mode(mode)
    
    def set_adc_bits(self, ba: int):
        """Delegate to base layer for BitAug"""
        self.base_layer.set_adc_bits(ba)
    
    def get_auxiliary_losses(self) -> dict:
        """Get kurtosis loss on effective weight (W + LoRA)."""
        effective_weight = self._get_effective_weight()
        
        if self.training and self.base_layer.use_kurtosis_loss:
            loss = self.base_layer.kurtosis_weight * compute_kurtosis_loss(
                effective_weight, self.base_layer.target_kurtosis
            )
        else:
            loss = torch.tensor(0.0, device=effective_weight.device)
        
        return {'total': loss}
    
    def _get_effective_weight(self) -> torch.Tensor:
        """Compute effective weight: W + scaling * A @ B"""
        lora_weight = self.scaling * (self.lora_A @ self.lora_B)
        return self.base_layer.weight + lora_weight
    
    def compute_reference_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reference output: Qx(X) @ Qw(W) - quantized but NO ADC, NO LoRA.
        
        This is the target for MSE warmup optimization (Paper Equation 12).
        We want to find A, B such that:
            ||Qx(X)Qw(W) - QA(Qx(X)Qw(W + AB))||^2_F is minimized
        
        Returns:
            Reference output in real domain (dequantized)
        """
        # 1) Build activation codes (same as forward)
        act_q = self.base_layer.activation_quantizer
        s_x = act_q.scale
        
        if act_q.symmetric:
            code_x = round_ste(safe_divide(x, s_x))
            code_x = torch.clamp(code_x, act_q.qmin, act_q.qmax)
        else:
            zp_x = act_q.zero_point
            code_x_temp = round_ste(safe_divide(x, s_x) + zp_x)
            code_x_temp = torch.clamp(code_x_temp, 0, act_q.qmax)
            
            if self.base_layer.ashift:
                code_x = code_x_temp - self.base_layer.C
            else:
                code_x = code_x_temp - zp_x
        
        # 2) Build weight codes using BASE weight W only (NO LoRA!)
        w_q = self.base_layer.weight_quantizer
        s_w_vec = w_q.scale
        s_w_b = s_w_vec.view(-1, 1)
        
        code_w = round_ste(safe_divide(self.base_layer.weight, s_w_b))
        code_w = torch.clamp(code_w, w_q.qmin, w_q.qmax)
        
        # 3) Integer MM (NO ADC quantization!)
        y_int = F.linear(code_x, code_w, bias=None)
        
        # 4) Dequantize back to real domain (skip ADC step)
        y_real = y_int * s_x * s_w_vec
        
        # Corrections for asymmetric quantization
        if not act_q.symmetric:
            wq_sum = code_w.sum(dim=1)
            if self.base_layer.ashift:
                zp_x = act_q.zero_point
                y_real = y_real - (zp_x * s_x) * (s_w_vec * wq_sum)
                y_real = y_real + (self.base_layer.C * s_x) * (s_w_vec * wq_sum)
        
        # Add bias
        if self.base_layer.bias is not None:
            y_real = y_real + self.base_layer.bias
        
        return y_real
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing ADC-LoRA: Y = QA(Qx(X)Qw(W + AB))
        
        The effective weight (W + scaling * A @ B) is quantized through the ADC pipeline.
        """
        # Compute effective weight with LoRA adaptation
        lora_weight = self.scaling * (self.lora_A @ self.lora_B)
        effective_weight = self.base_layer.weight + lora_weight
        
        # Apply dropout to the weight modification (during training)
        if self.training:
            # Apply dropout mask to lora contribution
            lora_weight_dropped = self.lora_dropout(lora_weight)
            if not isinstance(self.lora_dropout, nn.Identity):
                effective_weight = self.base_layer.weight + lora_weight_dropped
        
        # 1) Build activation codes (per-tensor quantizer)
        act_q = self.base_layer.activation_quantizer
        s_x = act_q.scale
        
        if act_q.symmetric:
            code_x = round_ste(safe_divide(x, s_x))
            qmin_x, qmax_x = act_q.qmin, act_q.qmax
            code_x = torch.clamp(code_x, qmin_x, qmax_x)
        else:
            zp_x = act_q.zero_point
            code_x_temp = round_ste(safe_divide(x, s_x) + zp_x)
            code_x_temp = torch.clamp(code_x_temp, 0, act_q.qmax)
            
            if self.base_layer.ashift:
                code_x = code_x_temp - self.base_layer.C
            else:
                code_x = code_x_temp - zp_x
        
        # 2) Build weight codes using EFFECTIVE weight (W + LoRA)
        w_q = self.base_layer.weight_quantizer
        s_w_vec = w_q.scale
        s_w_b = s_w_vec.view(-1, 1)
        
        # Quantize effective weight (this is the key ADC-LoRA difference!)
        code_w = round_ste(safe_divide(effective_weight, s_w_b))
        qmin_w, qmax_w = w_q.qmin, w_q.qmax
        code_w = torch.clamp(code_w, qmin_w, qmax_w)
        
        # 3) Integer MM in code domain
        y_int = F.linear(code_x, code_w, bias=None)
        
        delta = self.base_layer.delta
        na = self.base_layer.na
        pa = self.base_layer.pa
        
        # Apply ADC quantization (Paper Equation 2: uses floor)
        y_adc_codes = floor_ste(y_int / delta)
        y_adc_codes = torch.clamp(y_adc_codes, na, pa)
        
        # Dequantize back to the scale
        adc_output = y_adc_codes * delta
        
        # Store kurtosis loss on effective weight
        if self.training and self.base_layer.use_kurtosis_loss:
            kurtosis_loss = self.base_layer.kurtosis_weight * compute_kurtosis_loss(
                effective_weight, self.base_layer.target_kurtosis
            )
            self.base_layer._last_kurtosis_loss = kurtosis_loss
        else:
            self.base_layer._last_kurtosis_loss = torch.tensor(0.0, device=y_int.device, dtype=y_int.dtype)
        
        # 5) Dequantize back to real domain
        y_real = adc_output * s_x
        y_real = y_real * s_w_vec
        
        # Corrections for asymmetric quantization
        if not act_q.symmetric:
            wq_sum = code_w.sum(dim=1)
            
            if self.base_layer.ashift:
                zp_x = act_q.zero_point
                y_real = y_real - (zp_x * s_x) * (s_w_vec * wq_sum)
                y_real = y_real + (self.base_layer.C * s_x) * (s_w_vec * wq_sum)
        
        # 6) Add bias if present
        if self.base_layer.bias is not None:
            y_real = y_real + self.base_layer.bias
        
        return y_real
    
    def merge_lora_weights(self):
        """
        Merge LoRA weights into base weights permanently.
        Useful for inference after training.
        """
        with torch.no_grad():
            lora_weight = self.scaling * (self.lora_A @ self.lora_B)
            self.base_layer.weight.add_(lora_weight)
            # Reset LoRA to zero
            self.lora_A.zero_()
            self.lora_B.zero_()
    
    def get_num_trainable_params(self) -> int:
        """Return number of trainable LoRA parameters"""
        return self.lora_A.numel() + self.lora_B.numel()
    
    def get_compression_ratio(self) -> float:
        """Return compression ratio compared to full fine-tuning"""
        full_params = self.in_features * self.out_features
        lora_params = self.get_num_trainable_params()
        return full_params / lora_params


class LoRATiledLinearADC(nn.Module):
    """
    ADC-LoRA wrapper for TiledLinearADC.
    
    Applies LoRA adapters to each tile in the TiledLinearADC layer.
    Each tile gets its own LoRA A/B matrices, but they share the same rank and scaling.
    
    This maintains the tiling structure while reducing trainable parameters.
    """
    
    def __init__(self,
                 tiled_layer: TiledLinearADC,
                 r: int = 8,
                 alpha: float = 16.0,
                 dropout: float = 0.0):
        """
        Args:
            tiled_layer: The TiledLinearADC layer to wrap with LoRA
            r: LoRA rank for each tile
            alpha: LoRA scaling factor
            dropout: Dropout rate for LoRA path
        """
        super().__init__()
        
        self.tiled_layer = tiled_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.n_tiles = tiled_layer.n_tiles
        self.in_features_total = tiled_layer.in_features_total
        self.out_features = tiled_layer.out_features
        self.in_features_tile = tiled_layer.in_features_tile
        
        # Freeze all base layer weights
        for tile in self.tiled_layer.tiles:
            tile.weight.requires_grad = False
            if tile.bias is not None:
                tile.bias.requires_grad = False
        
        # Create LoRA adapters for each tile
        # Each tile has shape (out_features, in_features_tile)
        # LoRA: A (out_features, r), B (r, in_features_tile)
        self.lora_A = nn.ParameterList([
            nn.Parameter(torch.zeros(self.out_features, r))
            for _ in range(self.n_tiles)
        ])
        self.lora_B = nn.ParameterList([
            nn.Parameter(torch.zeros(r, self.in_features_tile))
            for _ in range(self.n_tiles)
        ])
        
        # Initialize A matrices with Kaiming uniform
        for lora_a in self.lora_A:
            nn.init.kaiming_uniform_(lora_a, a=5 ** 0.5)
        # B matrices stay zero for identity initialization
        
        # Dropout for regularization
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
    
    def set_quantizer_mode(self, mode: str):
        """Set mode for all quantizers in all tiles"""
        self.tiled_layer.set_quantizer_mode(mode)
    
    def set_adc_bits(self, ba: int):
        """Set ADC bits for all tiles (for BitAug)"""
        self.tiled_layer.set_adc_bits(ba)
    
    def get_adc_bits(self) -> int:
        """Get current ADC bit precision"""
        return self.tiled_layer.get_adc_bits()
    
    def get_auxiliary_losses(self) -> dict:
        """Get aggregated kurtosis loss from all tiles with LoRA-adjusted weights."""
        total = torch.tensor(0.0)
        
        for i, tile in enumerate(self.tiled_layer.tiles):
            lora_weight = self.scaling * (self.lora_A[i] @ self.lora_B[i])
            effective_weight = tile.weight + lora_weight
            
            if tile.use_kurtosis_loss:
                total = total + tile.kurtosis_weight * compute_kurtosis_loss(
                    effective_weight, tile.target_kurtosis
                )
        
        return {'total': total}
    
    def train(self, mode: bool = True):
        super().train(mode)
        self.tiled_layer.train(mode)
        return self
    
    def eval(self):
        super().eval()
        self.tiled_layer.eval()
        return self
    
    def compute_reference_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reference output: sum of Qx(X_i) @ Qw(W_i) for each tile.
        
        This is the target for MSE warmup (Paper Equation 12).
        No ADC quantization, no LoRA - just quantized activations times quantized weights.
        
        Returns:
            Reference output in real domain (dequantized)
        """
        if x.shape[-1] != self.in_features_total:
            raise ValueError(f"Expected last dim={self.in_features_total}, got {x.shape[-1]}")
        
        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_features_total)
        batch_size_2d = x2d.shape[0]
        
        y2d = torch.zeros(batch_size_2d, self.out_features, dtype=x2d.dtype, device=x2d.device)
        
        tile_in = self.in_features_tile
        for i, tile in enumerate(self.tiled_layer.tiles):
            # Get input slice for this tile
            if x2d.is_contiguous():
                xi = x2d.narrow(1, i * tile_in, tile_in)
            else:
                xi = x2d[:, i * tile_in:(i + 1) * tile_in]
            
            # Activation quantization
            act_q = tile.activation_quantizer
            s_x = act_q.scale
            
            if act_q.symmetric:
                code_x = round_ste(safe_divide(xi, s_x))
                code_x = torch.clamp(code_x, act_q.qmin, act_q.qmax)
            else:
                zp_x = act_q.zero_point
                code_x_temp = round_ste(safe_divide(xi, s_x) + zp_x)
                code_x_temp = torch.clamp(code_x_temp, 0, act_q.qmax)
                
                if tile.ashift:
                    code_x = code_x_temp - tile.C
                else:
                    code_x = code_x_temp - zp_x
            
            # Weight quantization using BASE weight only (NO LoRA!)
            w_q = tile.weight_quantizer
            s_w_vec = w_q.scale
            s_w_b = s_w_vec.view(-1, 1)
            
            code_w = round_ste(safe_divide(tile.weight, s_w_b))
            code_w = torch.clamp(code_w, w_q.qmin, w_q.qmax)
            
            # Integer MM (NO ADC quantization!)
            y_int = F.linear(code_x, code_w, bias=None)
            
            # Dequantize (skip ADC step)
            y_real = y_int * s_x * s_w_vec
            
            # Asymmetric corrections
            if not act_q.symmetric:
                wq_sum = code_w.sum(dim=1)
                if tile.ashift:
                    zp_x = act_q.zero_point
                    y_real = y_real - (zp_x * s_x) * (s_w_vec * wq_sum)
                    y_real = y_real + (tile.C * s_x) * (s_w_vec * wq_sum)
            
            # Add bias (only first tile has bias)
            if tile.bias is not None:
                y_real = y_real + tile.bias
            
            y2d.add_(y_real)
        
        y = y2d.reshape(*orig_shape[:-1], self.out_features)
        return y
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA-adapted tiles.
        
        Each tile computes: Y_i = QA(Qx(X_i)Qw(W_i + A_i @ B_i))
        """
        if x.shape[-1] != self.in_features_total:
            raise ValueError(f"Expected last dim={self.in_features_total}, got {x.shape[-1]}")
        
        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_features_total)
        batch_size_2d = x2d.shape[0]
        
        # Pre-allocate output tensor
        y2d = torch.zeros(batch_size_2d, self.out_features, dtype=x2d.dtype, device=x2d.device)
        
        tile_in = self.in_features_tile
        for i, tile in enumerate(self.tiled_layer.tiles):
            # Get input slice for this tile
            if x2d.is_contiguous():
                xi = x2d.narrow(1, i * tile_in, tile_in)
            else:
                xi = x2d[:, i * tile_in:(i + 1) * tile_in]
            
            # Compute effective weight with LoRA
            lora_weight = self.scaling * (self.lora_A[i] @ self.lora_B[i])
            if self.training:
                lora_weight = self.lora_dropout(lora_weight)
            effective_weight = tile.weight + lora_weight
            
            # ===== ADC Quantization Pipeline for this tile =====
            act_q = tile.activation_quantizer
            s_x = act_q.scale
            
            if act_q.symmetric:
                code_x = round_ste(safe_divide(xi, s_x))
                code_x = torch.clamp(code_x, act_q.qmin, act_q.qmax)
            else:
                zp_x = act_q.zero_point
                code_x_temp = round_ste(safe_divide(xi, s_x) + zp_x)
                code_x_temp = torch.clamp(code_x_temp, 0, act_q.qmax)
                
                if tile.ashift:
                    code_x = code_x_temp - tile.C
                else:
                    code_x = code_x_temp - zp_x
            
            # Weight quantization with effective weight
            w_q = tile.weight_quantizer
            s_w_vec = w_q.scale
            s_w_b = s_w_vec.view(-1, 1)
            
            code_w = round_ste(safe_divide(effective_weight, s_w_b))
            code_w = torch.clamp(code_w, w_q.qmin, w_q.qmax)
            
            # Integer MM
            y_int = F.linear(code_x, code_w, bias=None)
            
            # ADC quantization
            y_adc_codes = floor_ste(y_int / tile.delta)
            y_adc_codes = torch.clamp(y_adc_codes, tile.na, tile.pa)
            adc_output = y_adc_codes * tile.delta
            
            # Dequantize
            y_real = adc_output * s_x * s_w_vec
            
            # Asymmetric corrections
            if not act_q.symmetric:
                wq_sum = code_w.sum(dim=1)
                if tile.ashift:
                    zp_x = act_q.zero_point
                    y_real = y_real - (zp_x * s_x) * (s_w_vec * wq_sum)
                    y_real = y_real + (tile.C * s_x) * (s_w_vec * wq_sum)
            
            # Add bias (only first tile has bias)
            if tile.bias is not None:
                y_real = y_real + tile.bias
            
            y2d.add_(y_real)
        
        y = y2d.reshape(*orig_shape[:-1], self.out_features)
        return y
    
    def merge_lora_weights(self):
        """Merge all LoRA weights into base tile weights"""
        with torch.no_grad():
            for i, tile in enumerate(self.tiled_layer.tiles):
                lora_weight = self.scaling * (self.lora_A[i] @ self.lora_B[i])
                tile.weight.add_(lora_weight)
                self.lora_A[i].zero_()
                self.lora_B[i].zero_()
    
    def get_num_trainable_params(self) -> int:
        """Return total number of trainable LoRA parameters across all tiles"""
        total = 0
        for i in range(self.n_tiles):
            total += self.lora_A[i].numel() + self.lora_B[i].numel()
        return total
    
    def get_compression_ratio(self) -> float:
        """Return compression ratio compared to full fine-tuning"""
        full_params = self.in_features_total * self.out_features
        lora_params = self.get_num_trainable_params()
        return full_params / lora_params if lora_params > 0 else float('inf')
 