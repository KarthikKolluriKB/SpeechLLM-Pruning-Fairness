import torch
import torch.nn as nn


class EncoderProjectorConcat(nn.Module):
    """
    Linear projector with concatenation-based downsampling.

    Input: [B, T, encoder_dim]
    - Concatenates k consecutive frames: [B, T//k, encoder_dim*k]
    - Projects to LLM dimension: [B, T//k, llm_dim]

    The final LayerNorm matches the scale of the LLM's text embeddings: without
    it the projected audio embeddings have a norm one to two orders of magnitude
    larger than text embeddings, and the frozen LLM receives out-of-distribution
    inputs.

    hidden_dim of 512 suits small datasets (<10K samples); 2048 suits large ones
    (>100K samples). Dropout regularizes the small-dataset case.
    """

    def __init__(self, config):
        super().__init__()
        self.k = config.projector_ds_rate
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim

        self.hidden_dim = getattr(config, 'projector_hidden_dim', 2048)
        dropout_rate = getattr(config, 'projector_dropout', 0.1)

        self.linear1 = nn.Linear(self.encoder_dim * self.k, self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.linear2 = nn.Linear(self.hidden_dim, config.llm_dim)
        self.layer_norm = nn.LayerNorm(config.llm_dim, eps=1e-5)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.layer_norm(x)
        return x


class EncoderProjectorCov1d(nn.Module):
    """
    Conv1D projector with strided convolution for downsampling.
    """

    def __init__(self, config):
        super().__init__()
        self.k = config.projector_ds_rate
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        self.conv1d = nn.Conv1d(in_channels=self.encoder_dim, out_channels=self.encoder_dim, kernel_size=self.k, stride=self.k, padding=0)
        self.linear1 = nn.Linear(self.encoder_dim, 2048)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(2048, self.llm_dim)
        self.layer_norm = nn.LayerNorm(self.llm_dim, eps=1e-5)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.layer_norm(x)
        return x


class EncoderProjectorQFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        from transformers import Blip2QFormerConfig, Blip2QFormerModel
        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = self.encoder_dim
        configuration.num_hidden_layers = config.qformer_layers

        self.query_len = int(config.get("query_len", 64))
        self.query = nn.Parameter(torch.zeros(1, self.query_len, configuration.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)

        self.linear = nn.Linear(configuration.hidden_size, self.llm_dim)
        self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)

    def forward(self, x, atts):
        query = self.query.expand(x.shape[0], -1, -1)

        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )

        query_proj = self.norm(self.linear(query_output.last_hidden_state))

        return query_proj
