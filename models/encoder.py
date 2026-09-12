import types
import logging

import torch
import torch.nn.functional as F
import whisper

logger = logging.getLogger(__name__)


class WhisperWrappedEncoder:

    @classmethod
    def load(cls, model_config):
        """
        Load a Whisper encoder with optional layer pruning.

        Args:
            model_config: Config object with:
                - encoder_model: Whisper model name (e.g. "base")
                - encoder_num_layers: Number of encoder layers to keep
                  (None = all). Keeping N layers removes the top ones.

        Returns:
            Whisper encoder with variable length feature extraction.
        """
        num_layers = getattr(model_config, 'encoder_num_layers', None)

        def extract_variable_length_features(self, x: torch.Tensor):
            """
            x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
                the mel spectrogram of the audio
            """
            x = F.gelu(self.conv1(x))
            x = F.gelu(self.conv2(x))
            x = x.permute(0, 2, 1)

            # slice the positional embedding to the actual (variable) length
            x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)

            blocks_to_use = self.blocks[:self._num_layers] if self._num_layers else self.blocks
            for block in blocks_to_use:
                x = block(x)

            x = self.ln_post(x)
            return x

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        encoder = whisper.load_model(name=model_config.encoder_model, device=device).encoder

        total_layers = len(encoder.blocks)
        if num_layers is not None:
            if num_layers < 1 or num_layers > total_layers:
                raise ValueError(f"encoder_num_layers must be between 1 and {total_layers}, got {num_layers}")
            encoder._num_layers = num_layers
            logger.info(f"Whisper encoder: using {num_layers}/{total_layers} layers (pruned top {total_layers - num_layers} layers) on {device}")
        else:
            encoder._num_layers = None
            logger.info(f"Whisper encoder: using all {total_layers} layers on {device}")

        encoder.extract_variable_length_features = types.MethodType(extract_variable_length_features, encoder)

        return encoder
